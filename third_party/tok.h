/* Qwen GPT-2/ByteLevel BPE tokenizer foundation.
 *
 * The caller supplies the vocabulary, merge ranks and added tokens from GGUF.
 * Added control and user-defined tokens remain atomic during encode/decode.
 */
#ifndef TOK_H
#define TOK_H

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "tok_unicode.h"

typedef struct {
    const char *k;
    int klen;
    int v;
    int used;
} ment;

typedef struct {
    ment *e;
    int cap;
} hmap;

static uint64_t tk_fnv(const char *s, int n)
{
    uint64_t h = UINT64_C(1469598103934665603);
    for (int i = 0; i < n; ++i) {
        h ^= (unsigned char)s[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static void hm_init(hmap *m, int cap)
{
    m->cap = cap;
    m->e = (ment *)calloc((size_t)cap, sizeof(ment));
}

static int hm_put(hmap *m, const char *k, int klen, int v)
{
    if (!m || !m->e || m->cap <= 0) return 0;
    uint64_t h = tk_fnv(k, klen) & (uint64_t)(m->cap - 1);
    while (m->e[h].used) {
        if (m->e[h].klen == klen && !memcmp(m->e[h].k, k, (size_t)klen)) {
            m->e[h].v = v;
            return 1;
        }
        h = (h + 1u) & (uint64_t)(m->cap - 1);
    }
    m->e[h].k = k;
    m->e[h].klen = klen;
    m->e[h].v = v;
    m->e[h].used = 1;
    return 1;
}

static int hm_get(const hmap *m, const char *k, int klen)
{
    if (!m || !m->e || m->cap <= 0) return -1;
    uint64_t h = tk_fnv(k, klen) & (uint64_t)(m->cap - 1);
    while (m->e[h].used) {
        if (m->e[h].klen == klen && !memcmp(m->e[h].k, k, (size_t)klen))
            return m->e[h].v;
        h = (h + 1u) & (uint64_t)(m->cap - 1);
    }
    return -1;
}

typedef struct {
    char *str;
    int len;
    int id;
} Special;

typedef struct {
    hmap vocab;
    hmap merges;
    char **id2str;
    int *id_added;
    int *id_special;
    int n_ids;
    Special *sp;
    int nsp;
    uint32_t byte2cp[256];
    int byte2cp_len[256];
    char byte2str[256][3];
    int16_t cp2byte[1024];
    int failed;
} Tok;

static int u8_next(const unsigned char *s, int len, int i, uint32_t *cp)
{
    const unsigned char c = s[i];
    if (c < 0x80) {
        *cp = c;
        return 1;
    }
    if ((c >> 5) == 0x6 && i + 1 < len) {
        *cp = ((uint32_t)(c & 0x1f) << 6) | (s[i + 1] & 0x3f);
        return 2;
    }
    if ((c >> 4) == 0xe && i + 2 < len) {
        *cp = ((uint32_t)(c & 0x0f) << 12) |
              ((uint32_t)(s[i + 1] & 0x3f) << 6) | (s[i + 2] & 0x3f);
        return 3;
    }
    if ((c >> 3) == 0x1e && i + 3 < len) {
        *cp = ((uint32_t)(c & 0x07) << 18) |
              ((uint32_t)(s[i + 1] & 0x3f) << 12) |
              ((uint32_t)(s[i + 2] & 0x3f) << 6) | (s[i + 3] & 0x3f);
        return 4;
    }
    *cp = c;
    return 1;
}

static int u8_put(char *out, uint32_t cp)
{
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xc0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3f));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xe0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3f));
        out[2] = (char)(0x80 | (cp & 0x3f));
        return 3;
    }
    out[0] = (char)(0xf0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3f));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3f));
    out[3] = (char)(0x80 | (cp & 0x3f));
    return 4;
}

static void tk_build_bytemap(Tok *tokenizer)
{
    for (int i = 0; i < 1024; ++i) tokenizer->cp2byte[i] = -1;
    int direct[256];
    memset(direct, 0, sizeof(direct));
    for (int b = 33; b <= 126; ++b) direct[b] = 1;
    for (int b = 161; b <= 172; ++b) direct[b] = 1;
    for (int b = 174; b <= 255; ++b) direct[b] = 1;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        const uint32_t cp = direct[b] ? (uint32_t)b : (uint32_t)(256 + extra++);
        tokenizer->byte2cp[b] = cp;
        tokenizer->byte2cp_len[b] = u8_put(tokenizer->byte2str[b], cp);
        if (cp < 1024) tokenizer->cp2byte[cp] = (int16_t)b;
    }
}

static int cmp_sp_len(const void *left, const void *right)
{
    return ((const Special *)right)->len - ((const Special *)left)->len;
}

static void bpe_piece(Tok *tokenizer, const unsigned char *text,
                      int begin, int end, int *output, int *count, int maximum)
{
    const int bytes = end - begin;
    if (bytes <= 0 || tokenizer->failed) return;
    if (bytes > (INT_MAX - 1) / 2) {
        tokenizer->failed = 1;
        return;
    }

    char *encoded = (char *)malloc((size_t)bytes * 2u + 1u);
    if (!encoded) {
        tokenizer->failed = 1;
        return;
    }
    int encoded_length = 0;
    for (int i = begin; i < end; ++i) {
        const int byte = text[i];
        memcpy(encoded + encoded_length, tokenizer->byte2str[byte],
               (size_t)tokenizer->byte2cp_len[byte]);
        encoded_length += tokenizer->byte2cp_len[byte];
    }
    encoded[encoded_length] = '\0';

    const int whole = hm_get(&tokenizer->vocab, encoded, encoded_length);
    if (whole >= 0) {
        if (*count < maximum) output[(*count)++] = whole;
        free(encoded);
        return;
    }

    int *offsets = (int *)malloc(((size_t)encoded_length + 1u) * sizeof(int));
    int *lengths = (int *)malloc(((size_t)encoded_length + 1u) * sizeof(int));
    char *key = (char *)malloc((size_t)encoded_length * 2u + 2u);
    if (!offsets || !lengths || !key) {
        free(encoded);
        free(offsets);
        free(lengths);
        free(key);
        tokenizer->failed = 1;
        return;
    }

    int symbols = 0;
    for (int i = 0; i < encoded_length;) {
        uint32_t cp;
        const int length = u8_next((const unsigned char *)encoded,
                                   encoded_length, i, &cp);
        (void)cp;
        offsets[symbols] = i;
        lengths[symbols] = length;
        ++symbols;
        i += length;
    }
    for (;;) {
        int best = INT_MAX;
        int best_pair = -1;
        for (int i = 0; i + 1 < symbols; ++i) {
            const int left = lengths[i];
            const int right = lengths[i + 1];
            memcpy(key, encoded + offsets[i], (size_t)left);
            key[left] = '\0';
            memcpy(key + left + 1, encoded + offsets[i + 1], (size_t)right);
            const int rank = hm_get(&tokenizer->merges, key, left + 1 + right);
            if (rank >= 0 && rank < best) {
                best = rank;
                best_pair = i;
            }
        }
        if (best_pair < 0) break;
        lengths[best_pair] = offsets[best_pair + 1] +
                             lengths[best_pair + 1] - offsets[best_pair];
        for (int i = best_pair + 1; i < symbols - 1; ++i) {
            offsets[i] = offsets[i + 1];
            lengths[i] = lengths[i + 1];
        }
        --symbols;
    }
    for (int i = 0; i < symbols; ++i) {
        const int id = hm_get(&tokenizer->vocab, encoded + offsets[i], lengths[i]);
        if (id >= 0 && *count < maximum) output[(*count)++] = id;
    }
    free(encoded);
    free(offsets);
    free(lengths);
    free(key);
}

/* Qwen's GPT-2 regex alternatives are evaluated in order. */
static void pretok_chunk(Tok *tokenizer, const unsigned char *text,
                         int begin, int end, int *output, int *count, int maximum)
{
    const int bytes = end - begin;
    if (bytes <= 0 || tokenizer->failed) return;
    uint32_t *codepoints = (uint32_t *)malloc(
        ((size_t)bytes + 1u) * sizeof(uint32_t));
    int *offsets = (int *)malloc(((size_t)bytes + 2u) * sizeof(int));
    if (!codepoints || !offsets) {
        free(codepoints);
        free(offsets);
        tokenizer->failed = 1;
        return;
    }

    int codepoint_count = 0;
    for (int i = begin; i < end;) {
        uint32_t cp;
        const int length = u8_next(text, end, i, &cp);
        offsets[codepoint_count] = i;
        codepoints[codepoint_count++] = cp;
        i += length;
    }
    offsets[codepoint_count] = end;

#define Q38_IS_NEWLINE(c) ((c) == '\r' || (c) == '\n')
#define Q38_LOWER_ASCII(c) (((c) >= 'A' && (c) <= 'Z') ? ((c) + 32) : (c))
    int i = 0;
    while (i < codepoint_count && !tokenizer->failed) {
        const int start = i;
        const uint32_t current = codepoints[i];

        if (current == '\'' && i + 1 < codepoint_count) {
            const uint32_t next = Q38_LOWER_ASCII(codepoints[i + 1]);
            if (i + 2 < codepoint_count) {
                const uint32_t third = Q38_LOWER_ASCII(codepoints[i + 2]);
                if ((next == 'r' && third == 'e') ||
                    (next == 'v' && third == 'e') ||
                    (next == 'l' && third == 'l')) {
                    i += 3;
                    bpe_piece(tokenizer, text, offsets[start], offsets[i],
                              output, count, maximum);
                    continue;
                }
            }
            if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                i += 2;
                bpe_piece(tokenizer, text, offsets[start], offsets[i],
                          output, count, maximum);
                continue;
            }
        }

        {
            int j = i;
            if (!is_L(current) && !Q38_IS_NEWLINE(current) && !is_N(current)) {
                if (j + 1 < codepoint_count && is_L(codepoints[j + 1]))
                    ++j;
                else
                    j = -1;
            }
            if (j >= 0 && is_L(codepoints[j])) {
                while (j < codepoint_count && is_L(codepoints[j])) ++j;
                i = j;
                bpe_piece(tokenizer, text, offsets[start], offsets[i],
                          output, count, maximum);
                continue;
            }
        }

        if (is_N(current)) {
            ++i;
            bpe_piece(tokenizer, text, offsets[start], offsets[i],
                      output, count, maximum);
            continue;
        }

        {
            int j = i;
            if (current == ' ' && j + 1 < codepoint_count &&
                !is_S(codepoints[j + 1]) && !is_L(codepoints[j + 1]) &&
                !is_N(codepoints[j + 1])) {
                ++j;
            }
            if (j < codepoint_count && !is_S(codepoints[j]) &&
                !is_L(codepoints[j]) && !is_N(codepoints[j])) {
                while (j < codepoint_count && !is_S(codepoints[j]) &&
                       !is_L(codepoints[j]) && !is_N(codepoints[j])) ++j;
                while (j < codepoint_count && Q38_IS_NEWLINE(codepoints[j])) ++j;
                i = j;
                bpe_piece(tokenizer, text, offsets[start], offsets[i],
                          output, count, maximum);
                continue;
            }
        }

        {
            int run = i;
            while (run < codepoint_count && is_S(codepoints[run])) ++run;
            if (run > i) {
                int last_newline = -1;
                for (int j = i; j < run; ++j) {
                    if (Q38_IS_NEWLINE(codepoints[j])) last_newline = j;
                }
                if (last_newline >= 0) {
                    i = last_newline + 1;
                    bpe_piece(tokenizer, text, offsets[start], offsets[i],
                              output, count, maximum);
                    continue;
                }
                int span_end = run < codepoint_count ? run - 1 : run;
                if (span_end <= i) span_end = i + 1;
                i = span_end;
                bpe_piece(tokenizer, text, offsets[start], offsets[i],
                          output, count, maximum);
                continue;
            }
        }

        ++i;
        bpe_piece(tokenizer, text, offsets[start], offsets[i],
                  output, count, maximum);
    }
#undef Q38_IS_NEWLINE
#undef Q38_LOWER_ASCII
    free(codepoints);
    free(offsets);
}

static int tok_encode(Tok *tokenizer, const char *text, int length,
                      int *output, int maximum)
{
    if (!tokenizer || !text || !output || length < 0 || maximum < 0) return -1;
    tokenizer->failed = 0;
    const unsigned char *bytes = (const unsigned char *)text;
    int count = 0;
    int at = 0;
    while (at < length) {
        int hit_position = -1;
        int hit_length = 0;
        int hit_id = -1;
        for (int position = at; position < length && hit_position < 0;
             ++position) {
            for (int special = 0; special < tokenizer->nsp; ++special) {
                const int special_length = tokenizer->sp[special].len;
                if (special_length > 0 && position + special_length <= length &&
                    !memcmp(bytes + position, tokenizer->sp[special].str,
                            (size_t)special_length)) {
                    hit_position = position;
                    hit_length = special_length;
                    hit_id = tokenizer->sp[special].id;
                    break;
                }
            }
        }
        const int chunk_end = hit_position < 0 ? length : hit_position;
        if (chunk_end > at) {
            pretok_chunk(tokenizer, bytes, at, chunk_end,
                         output, &count, maximum);
            if (tokenizer->failed) return -1;
        }
        if (hit_position < 0) break;
        if (count < maximum) output[count++] = hit_id;
        at = hit_position + hit_length;
    }
    return count;
}

static int tok_id_of(const Tok *tokenizer, const char *content)
{
    for (int i = 0; i < tokenizer->nsp; ++i) {
        if (!strcmp(tokenizer->sp[i].str, content)) return tokenizer->sp[i].id;
    }
    return -1;
}

static int tok_decode(const Tok *tokenizer, const int *ids, int count,
                      char *output, int maximum)
{
    int bytes = 0;
    for (int i = 0; i < count; ++i) {
        const int id = ids[i];
        if (id < 0 || id >= tokenizer->n_ids || !tokenizer->id2str[id])
            continue;
        const char *encoded = tokenizer->id2str[id];
        if (tokenizer->id_added[id]) {
            const int length = (int)strlen(encoded);
            for (int j = 0; j < length && bytes < maximum; ++j)
                output[bytes++] = encoded[j];
            continue;
        }
        const int length = (int)strlen(encoded);
        for (int j = 0; j < length;) {
            uint32_t cp;
            const int consumed = u8_next((const unsigned char *)encoded,
                                         length, j, &cp);
            j += consumed;
            if (cp < 1024 && tokenizer->cp2byte[cp] >= 0 && bytes < maximum)
                output[bytes++] = (char)(unsigned char)tokenizer->cp2byte[cp];
        }
    }
    if (bytes < maximum) output[bytes] = '\0';
    return bytes;
}

#endif
