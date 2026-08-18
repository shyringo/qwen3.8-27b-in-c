#include "qwen38_tokenizer.h"

#include "qwen38_gguf.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tok.h"
#include "tok_nfc.h"

struct Q38Tokenizer {
    Tok tokenizer;
    uint32_t eos;
};

static int q38_span_contains(Q38GGUFString string, const char *needle)
{
    const size_t length = strlen(needle);
    if (length > string.length) return 0;
    for (uint64_t i = 0; i + length <= string.length; ++i) {
        if (memcmp(string.data + i, needle, length) == 0) return 1;
    }
    return 0;
}

static char *q38_copy_span(Q38GGUFString string)
{
    if (string.length > SIZE_MAX - 1) return NULL;
    char *copy = (char *)malloc((size_t)string.length + 1u);
    if (!copy) return NULL;
    memcpy(copy, string.data, (size_t)string.length);
    copy[string.length] = '\0';
    return copy;
}

static void q38_tokenizer_release(Tok *tokenizer)
{
    if (!tokenizer) return;
    if (tokenizer->id2str) {
        for (int i = 0; i < tokenizer->n_ids; ++i) free(tokenizer->id2str[i]);
    }
    if (tokenizer->merges.e) {
        for (int i = 0; i < tokenizer->merges.cap; ++i) {
            if (tokenizer->merges.e[i].used) free((void *)tokenizer->merges.e[i].k);
        }
    }
    free(tokenizer->vocab.e);
    free(tokenizer->merges.e);
    free(tokenizer->id2str);
    free(tokenizer->id_added);
    free(tokenizer->id_special);
    free(tokenizer->sp);
    memset(tokenizer, 0, sizeof(*tokenizer));
}

static int q38_build_vocab(Tok *tokenizer, const Q38GGUFMeta *tokens,
                           const Q38GGUFMeta *types)
{
    if (tokens->count > INT_MAX / 2) return 0;
    Q38GGUFString *spans = (Q38GGUFString *)calloc((size_t)tokens->count,
                                                    sizeof(*spans));
    if (!spans || !q38_gguf_meta_array_strings(tokens, spans, tokens->count)) {
        free(spans);
        return 0;
    }
    tokenizer->n_ids = (int)tokens->count;
    tokenizer->id2str = (char **)calloc((size_t)tokenizer->n_ids, sizeof(char *));
    tokenizer->id_added = (int *)calloc((size_t)tokenizer->n_ids, sizeof(int));
    tokenizer->id_special = (int *)calloc((size_t)tokenizer->n_ids, sizeof(int));
    int capacity = 1;
    while (capacity < tokenizer->n_ids * 2) capacity <<= 1;
    hm_init(&tokenizer->vocab, capacity);
    if (!tokenizer->id2str || !tokenizer->id_added || !tokenizer->id_special ||
        !tokenizer->vocab.e) {
        free(spans);
        return 0;
    }
    int special_count = 0;
    for (int id = 0; id < tokenizer->n_ids; ++id) {
        char *token = q38_copy_span(spans[id]);
        if (!token) {
            free(spans);
            return 0;
        }
        tokenizer->id2str[id] = token;
        if (!hm_put(&tokenizer->vocab, tokenizer->id2str[id],
                    (int)spans[id].length, id)) {
            free(spans);
            return 0;
        }
        int32_t type = 1;
        if (types && !q38_gguf_meta_array_i32(types, (uint64_t)id, &type)) {
            free(spans);
            return 0;
        }
        if (type == 3 || type == 4) ++special_count;
    }
    if (special_count) {
        tokenizer->sp = (Special *)calloc((size_t)special_count,
                                           sizeof(Special));
        if (!tokenizer->sp) {
            free(spans);
            return 0;
        }
    }
    for (int id = 0; id < tokenizer->n_ids; ++id) {
        int32_t type = 1;
        if (types) (void)q38_gguf_meta_array_i32(types, (uint64_t)id, &type);
        if (type != 3 && type != 4) continue;
        tokenizer->id_added[id] = 1;
        tokenizer->id_special[id] = type == 3;
        Special *special = &tokenizer->sp[tokenizer->nsp++];
        special->str = tokenizer->id2str[id];
        special->len = (int)strlen(special->str);
        special->id = id;
    }
    if (tokenizer->nsp > 1)
        qsort(tokenizer->sp, (size_t)tokenizer->nsp,
              sizeof(Special), cmp_sp_len);
    free(spans);
    return 1;
}

static int q38_build_merges(Tok *tokenizer, const Q38GGUFMeta *merges)
{
    if (!merges || merges->count > INT_MAX / 2) return 0;
    Q38GGUFString *spans = (Q38GGUFString *)calloc((size_t)merges->count,
                                                    sizeof(*spans));
    if (!spans || !q38_gguf_meta_array_strings(merges, spans, merges->count)) {
        free(spans);
        return 0;
    }
    int capacity = 1;
    while (capacity < (int)merges->count * 2) capacity <<= 1;
    hm_init(&tokenizer->merges, capacity);
    if (!tokenizer->merges.e) {
        free(spans);
        return 0;
    }
    for (uint64_t rank = 0; rank < merges->count; ++rank) {
        const char *separator = (const char *)memchr(spans[rank].data, ' ',
                                                     (size_t)spans[rank].length);
        if (!separator) {
            free(spans);
            return 0;
        }
        const size_t left = (size_t)(separator - spans[rank].data);
        const size_t right = (size_t)spans[rank].length - left - 1u;
        char *key = (char *)malloc(left + 1u + right);
        if (!key) {
            free(spans);
            return 0;
        }
        memcpy(key, spans[rank].data, left);
        key[left] = '\0';
        memcpy(key + left + 1u, separator + 1, right);
        const int key_length = (int)(left + 1u + right);
        if (hm_get(&tokenizer->merges, key, key_length) >= 0 ||
            !hm_put(&tokenizer->merges, key, key_length, (int)rank)) {
            free(key);
            free(spans);
            return 0;
        }
    }
    free(spans);
    return 1;
}

Q38Tokenizer *q38_tokenizer_open_gguf(const char *path)
{
    if (!path) return NULL;
    Q38GGUF gguf;
    if (!q38_gguf_open(&gguf, path)) return NULL;
    const Q38GGUFMeta *tokens = q38_gguf_find_meta(&gguf, "tokenizer.ggml.tokens");
    const Q38GGUFMeta *types = q38_gguf_find_meta(&gguf, "tokenizer.ggml.token_type");
    const Q38GGUFMeta *merges = q38_gguf_find_meta(&gguf, "tokenizer.ggml.merges");
    Q38Tokenizer *result = (Q38Tokenizer *)calloc(1, sizeof(*result));
    if (!result || !tokens || !merges ||
        (types && types->count != tokens->count)) {
        free(result);
        q38_gguf_close(&gguf);
        return NULL;
    }
    tk_build_bytemap(&result->tokenizer);
    Q38GGUFString pre;
    if (q38_gguf_meta_string(&gguf, "tokenizer.ggml.pre", &pre) &&
        !q38_span_contains(pre, "qwen")) {
        fprintf(stderr, "qwen38: unsupported tokenizer pre-model: %.*s\n",
                (int)pre.length, pre.data);
        q38_tokenizer_close(result);
        q38_gguf_close(&gguf);
        return NULL;
    }
    uint32_t eos = 0;
    if (!q38_gguf_meta_u32(&gguf, "tokenizer.ggml.eos_token_id", &eos) ||
        eos >= tokens->count ||
        !q38_build_vocab(&result->tokenizer, tokens, types) ||
        !q38_build_merges(&result->tokenizer, merges)) {
        q38_tokenizer_close(result);
        q38_gguf_close(&gguf);
        return NULL;
    }
    result->eos = eos;
    q38_gguf_close(&gguf);
    return result;
}

void q38_tokenizer_close(Q38Tokenizer *tokenizer)
{
    if (!tokenizer) return;
    q38_tokenizer_release(&tokenizer->tokenizer);
    free(tokenizer);
}

int q38_tokenizer_encode(Q38Tokenizer *tokenizer, const char *text,
                         size_t text_length, uint32_t *tokens,
                         size_t capacity)
{
    if (!tokenizer || !text || !tokens || text_length > INT_MAX / 2 ||
        capacity > INT_MAX) return -1;
    char *normalized = NULL;
    size_t normalized_length = 0;
    if (!tok_nfc_normalize(text, text_length, &normalized,
                           &normalized_length) ||
        normalized_length > INT_MAX) {
        free(normalized);
        return -1;
    }
    const int count = tok_encode(&tokenizer->tokenizer, normalized,
                                 (int)normalized_length,
                                 (int *)tokens, (int)capacity);
    free(normalized);
    return count;
}

int q38_tokenizer_decode_token(Q38Tokenizer *tokenizer, uint32_t token,
                               char *text, size_t capacity)
{
    if (!tokenizer || !text || capacity == 0 || capacity > INT_MAX ||
        token > INT_MAX) return -1;
    const int id = (int)token;
    return tok_decode(&tokenizer->tokenizer, &id, 1, text, (int)capacity - 1);
}

int q38_tokenizer_find(Q38Tokenizer *tokenizer, const char *text)
{
    return tokenizer && text ? tok_id_of(&tokenizer->tokenizer, text) : -1;
}

int q38_tokenizer_is_special(const Q38Tokenizer *tokenizer, uint32_t token)
{
    return tokenizer && token < (uint32_t)tokenizer->tokenizer.n_ids &&
           tokenizer->tokenizer.id_special[token];
}

uint32_t q38_tokenizer_eos(const Q38Tokenizer *tokenizer)
{
    return tokenizer ? tokenizer->eos : 0;
}

uint32_t q38_tokenizer_vocab_size(const Q38Tokenizer *tokenizer)
{
    return tokenizer ? (uint32_t)tokenizer->tokenizer.n_ids : 0;
}
