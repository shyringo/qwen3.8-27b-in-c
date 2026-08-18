#ifndef TOK_NFC_H
#define TOK_NFC_H

#include <stdint.h>
#include <stdlib.h>

#include "tok_nfc_data.h"

typedef struct {
    uint32_t *values;
    size_t count;
    size_t capacity;
} TokNfcBuffer;

static int tok_nfc_push(TokNfcBuffer *buffer, uint32_t value)
{
    if (buffer->count == buffer->capacity) {
        size_t capacity = buffer->capacity ? buffer->capacity * 2u : 32u;
        if (capacity < buffer->capacity ||
            capacity > SIZE_MAX / sizeof(uint32_t)) return 0;
        uint32_t *values = (uint32_t *)realloc(
            buffer->values, capacity * sizeof(uint32_t));
        if (!values) return 0;
        buffer->values = values;
        buffer->capacity = capacity;
    }
    buffer->values[buffer->count++] = value;
    return 1;
}

static const TokNfcDecomp *tok_nfc_find_decomposition(uint32_t code)
{
    size_t low = 0;
    size_t high = TOK_NFC_DECOMP_COUNT;
    while (low < high) {
        const size_t middle = low + (high - low) / 2u;
        const TokNfcDecomp *entry = &tok_nfc_decompositions[middle];
        if (entry->code < code)
            low = middle + 1u;
        else
            high = middle;
    }
    return low < TOK_NFC_DECOMP_COUNT &&
           tok_nfc_decompositions[low].code == code
        ? &tok_nfc_decompositions[low] : NULL;
}

static uint8_t tok_nfc_class(uint32_t code)
{
    size_t low = 0;
    size_t high = TOK_NFC_CLASS_COUNT;
    while (low < high) {
        const size_t middle = low + (high - low) / 2u;
        if (tok_nfc_classes[middle].code < code)
            low = middle + 1u;
        else
            high = middle;
    }
    return low < TOK_NFC_CLASS_COUNT && tok_nfc_classes[low].code == code
        ? tok_nfc_classes[low].value : 0u;
}

static uint32_t tok_nfc_compose(uint32_t first, uint32_t second)
{
    enum {
        SBASE = 0xac00,
        LBASE = 0x1100,
        VBASE = 0x1161,
        TBASE = 0x11a7,
        LCOUNT = 19,
        VCOUNT = 21,
        TCOUNT = 28,
        NCOUNT = VCOUNT * TCOUNT,
        SCOUNT = LCOUNT * NCOUNT
    };
    const uint32_t lindex = first - LBASE;
    if (lindex < LCOUNT) {
        const uint32_t vindex = second - VBASE;
        if (vindex < VCOUNT)
            return SBASE + (lindex * VCOUNT + vindex) * TCOUNT;
    }
    const uint32_t sindex = first - SBASE;
    if (sindex < SCOUNT && sindex % TCOUNT == 0u) {
        const uint32_t tindex = second - TBASE;
        if (tindex > 0u && tindex < TCOUNT) return first + tindex;
    }

    size_t low = 0;
    size_t high = TOK_NFC_COMPOSE_COUNT;
    while (low < high) {
        const size_t middle = low + (high - low) / 2u;
        const TokNfcCompose *entry = &tok_nfc_compositions[middle];
        if (entry->first < first ||
            (entry->first == first && entry->second < second))
            low = middle + 1u;
        else
            high = middle;
    }
    if (low < TOK_NFC_COMPOSE_COUNT) {
        const TokNfcCompose *entry = &tok_nfc_compositions[low];
        if (entry->first == first && entry->second == second)
            return entry->result;
    }
    return 0u;
}

static int tok_nfc_decompose(TokNfcBuffer *buffer, uint32_t code)
{
    enum {
        SBASE = 0xac00,
        LBASE = 0x1100,
        VBASE = 0x1161,
        TBASE = 0x11a7,
        VCOUNT = 21,
        TCOUNT = 28,
        NCOUNT = VCOUNT * TCOUNT,
        SCOUNT = 19 * NCOUNT
    };
    const uint32_t sindex = code - SBASE;
    if (sindex < SCOUNT) {
        if (!tok_nfc_push(buffer, LBASE + sindex / NCOUNT) ||
            !tok_nfc_push(buffer, VBASE + (sindex % NCOUNT) / TCOUNT))
            return 0;
        const uint32_t trail = TBASE + sindex % TCOUNT;
        return trail == TBASE || tok_nfc_push(buffer, trail);
    }
    const TokNfcDecomp *entry = tok_nfc_find_decomposition(code);
    if (!entry) return tok_nfc_push(buffer, code);
    for (uint8_t index = 0; index < entry->length; ++index) {
        if (!tok_nfc_push(buffer,
                          tok_nfc_decomp_values[entry->offset + index]))
            return 0;
    }
    return 1;
}

static int tok_nfc_utf8_next(const unsigned char *text, size_t length,
                             size_t *offset, uint32_t *code)
{
    const size_t at = *offset;
    if (at >= length) return 0;
    const unsigned char first = text[at];
    if (first < 0x80u) {
        *code = first;
        *offset = at + 1u;
        return 1;
    }
    if (first >= 0xc2u && first <= 0xdfu && at + 1u < length &&
        (text[at + 1u] & 0xc0u) == 0x80u) {
        *code = ((uint32_t)(first & 0x1fu) << 6) |
                (uint32_t)(text[at + 1u] & 0x3fu);
        *offset = at + 2u;
        return 1;
    }
    if (first >= 0xe0u && first <= 0xefu && at + 2u < length &&
        (text[at + 1u] & 0xc0u) == 0x80u &&
        (text[at + 2u] & 0xc0u) == 0x80u &&
        (first != 0xe0u || text[at + 1u] >= 0xa0u) &&
        (first != 0xedu || text[at + 1u] < 0xa0u)) {
        *code = ((uint32_t)(first & 0x0fu) << 12) |
                ((uint32_t)(text[at + 1u] & 0x3fu) << 6) |
                (uint32_t)(text[at + 2u] & 0x3fu);
        *offset = at + 3u;
        return 1;
    }
    if (first >= 0xf0u && first <= 0xf4u && at + 3u < length &&
        (text[at + 1u] & 0xc0u) == 0x80u &&
        (text[at + 2u] & 0xc0u) == 0x80u &&
        (text[at + 3u] & 0xc0u) == 0x80u &&
        (first != 0xf0u || text[at + 1u] >= 0x90u) &&
        (first != 0xf4u || text[at + 1u] < 0x90u)) {
        *code = ((uint32_t)(first & 0x07u) << 18) |
                ((uint32_t)(text[at + 1u] & 0x3fu) << 12) |
                ((uint32_t)(text[at + 2u] & 0x3fu) << 6) |
                (uint32_t)(text[at + 3u] & 0x3fu);
        *offset = at + 4u;
        return 1;
    }
    return 0;
}

static size_t tok_nfc_utf8_put(char *text, uint32_t code)
{
    if (code < 0x80u) {
        text[0] = (char)code;
        return 1u;
    }
    if (code < 0x800u) {
        text[0] = (char)(0xc0u | (code >> 6));
        text[1] = (char)(0x80u | (code & 0x3fu));
        return 2u;
    }
    if (code < 0x10000u) {
        text[0] = (char)(0xe0u | (code >> 12));
        text[1] = (char)(0x80u | ((code >> 6) & 0x3fu));
        text[2] = (char)(0x80u | (code & 0x3fu));
        return 3u;
    }
    text[0] = (char)(0xf0u | (code >> 18));
    text[1] = (char)(0x80u | ((code >> 12) & 0x3fu));
    text[2] = (char)(0x80u | ((code >> 6) & 0x3fu));
    text[3] = (char)(0x80u | (code & 0x3fu));
    return 4u;
}

static int tok_nfc_normalize(const char *input, size_t input_length,
                             char **output, size_t *output_length)
{
    if (!input || !output || !output_length) return 0;
    TokNfcBuffer buffer = {NULL, 0, 0};
    size_t offset = 0;
    while (offset < input_length) {
        uint32_t code = 0;
        if (!tok_nfc_utf8_next((const unsigned char *)input, input_length,
                               &offset, &code) ||
            !tok_nfc_decompose(&buffer, code)) {
            free(buffer.values);
            return 0;
        }
    }

    for (size_t index = 1; index < buffer.count; ++index) {
        const uint8_t combining = tok_nfc_class(buffer.values[index]);
        if (!combining) continue;
        size_t cursor = index;
        while (cursor > 0u) {
            const uint8_t previous = tok_nfc_class(buffer.values[cursor - 1u]);
            if (!previous || previous <= combining) break;
            const uint32_t temporary = buffer.values[cursor - 1u];
            buffer.values[cursor - 1u] = buffer.values[cursor];
            buffer.values[cursor] = temporary;
            --cursor;
        }
    }

    size_t composed_count = buffer.count;
    if (buffer.count) {
        size_t write = 1u;
        size_t starter = 0u;
        uint8_t last_class = tok_nfc_class(buffer.values[0]);
        for (size_t index = 1; index < buffer.count; ++index) {
            const uint32_t code = buffer.values[index];
            const uint8_t combining = tok_nfc_class(code);
            const uint32_t combined = tok_nfc_compose(
                buffer.values[starter], code);
            if (combined && (last_class < combining || last_class == 0u)) {
                buffer.values[starter] = combined;
                continue;
            }
            if (combining == 0u) starter = write;
            buffer.values[write++] = code;
            last_class = combining;
        }
        composed_count = write;
    }

    if (composed_count > (SIZE_MAX - 1u) / 4u) {
        free(buffer.values);
        return 0;
    }
    char *normalized = (char *)malloc(composed_count * 4u + 1u);
    if (!normalized) {
        free(buffer.values);
        return 0;
    }
    size_t bytes = 0;
    for (size_t index = 0; index < composed_count; ++index)
        bytes += tok_nfc_utf8_put(normalized + bytes, buffer.values[index]);
    normalized[bytes] = '\0';
    free(buffer.values);
    *output = normalized;
    *output_length = bytes;
    return 1;
}

#endif
