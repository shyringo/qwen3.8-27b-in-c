#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "qwen38_gguf.h"

#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define Q38_GGUF_MAGIC 0x46554747u

typedef struct {
    const uint8_t *base;
    const uint8_t *at;
    const uint8_t *end;
    int failed;
} Q38Reader;

static int q38_span_equal(Q38GGUFString span, const char *text)
{
    const size_t length = strlen(text);
    return span.length == (uint64_t)length &&
           memcmp(span.data, text, length) == 0;
}

static int q38_take(Q38Reader *reader, uint64_t size, const uint8_t **data)
{
    if (reader->failed || size > (uint64_t)(reader->end - reader->at)) {
        reader->failed = 1;
        return 0;
    }
    *data = reader->at;
    reader->at += size;
    return 1;
}

static uint8_t q38_u8(Q38Reader *reader)
{
    const uint8_t *data;
    return q38_take(reader, 1, &data) ? data[0] : 0;
}

static uint16_t q38_u16(Q38Reader *reader)
{
    const uint8_t *data;
    if (!q38_take(reader, 2, &data)) return 0;
    return (uint16_t)data[0] | (uint16_t)data[1] << 8;
}

static uint32_t q38_u32(Q38Reader *reader)
{
    const uint8_t *data;
    if (!q38_take(reader, 4, &data)) return 0;
    return (uint32_t)data[0] | (uint32_t)data[1] << 8 |
           (uint32_t)data[2] << 16 | (uint32_t)data[3] << 24;
}

static uint64_t q38_u64(Q38Reader *reader)
{
    const uint8_t *data;
    uint64_t value = 0;
    if (!q38_take(reader, 8, &data)) return 0;
    for (int i = 7; i >= 0; --i) value = (value << 8) | data[i];
    return value;
}

static int q38_string(Q38Reader *reader, Q38GGUFString *string)
{
    const uint64_t length = q38_u64(reader);
    const uint8_t *data;
    if (!q38_take(reader, length, &data)) return 0;
    string->data = (const char *)data;
    string->length = length;
    return 1;
}

static uint64_t q38_meta_scalar_size(uint32_t type)
{
    switch (type) {
    case Q38_GGUF_META_UINT8:
    case Q38_GGUF_META_INT8:
    case Q38_GGUF_META_BOOL: return 1;
    case Q38_GGUF_META_UINT16:
    case Q38_GGUF_META_INT16: return 2;
    case Q38_GGUF_META_UINT32:
    case Q38_GGUF_META_INT32:
    case Q38_GGUF_META_FLOAT32: return 4;
    case Q38_GGUF_META_UINT64:
    case Q38_GGUF_META_INT64:
    case Q38_GGUF_META_FLOAT64: return 8;
    default: return 0;
    }
}

static int q38_skip_meta_value(Q38Reader *reader, uint32_t type,
                               uint64_t count)
{
    const uint64_t scalar_size = q38_meta_scalar_size(type);
    const uint8_t *unused;
    if (scalar_size) {
        if (count == 1) {
            switch (scalar_size) {
            case 1: (void)q38_u8(reader); break;
            case 2: (void)q38_u16(reader); break;
            case 4: (void)q38_u32(reader); break;
            case 8: (void)q38_u64(reader); break;
            default: return 0;
            }
            return !reader->failed;
        }
        if (count > UINT64_MAX / scalar_size) return 0;
        return q38_take(reader, count * scalar_size, &unused);
    }
    if (type == Q38_GGUF_META_STRING) {
        Q38GGUFString string;
        for (uint64_t i = 0; i < count; ++i) {
            if (!q38_string(reader, &string)) return 0;
        }
        return 1;
    }
    return 0;
}

uint32_t q38_ggml_block_elements(uint32_t type)
{
    switch (type) {
    case Q38_GGML_F32:
    case Q38_GGML_F16:
    case Q38_GGML_BF16: return 1;
    case Q38_GGML_Q4_0:
    case Q38_GGML_Q4_1:
    case Q38_GGML_Q5_0:
    case Q38_GGML_Q5_1:
    case Q38_GGML_Q8_0: return 32;
    case Q38_GGML_Q2_K:
    case Q38_GGML_Q3_K:
    case Q38_GGML_Q4_K:
    case Q38_GGML_Q5_K:
    case Q38_GGML_Q6_K: return 256;
    default: return 0;
    }
}

uint32_t q38_ggml_block_bytes(uint32_t type)
{
    switch (type) {
    case Q38_GGML_F32: return 4;
    case Q38_GGML_F16:
    case Q38_GGML_BF16: return 2;
    case Q38_GGML_Q4_0: return 18;
    case Q38_GGML_Q4_1: return 20;
    case Q38_GGML_Q5_0: return 22;
    case Q38_GGML_Q5_1: return 24;
    case Q38_GGML_Q8_0: return 34;
    case Q38_GGML_Q2_K: return 84;
    case Q38_GGML_Q3_K: return 110;
    case Q38_GGML_Q4_K: return 144;
    case Q38_GGML_Q5_K: return 176;
    case Q38_GGML_Q6_K: return 210;
    default: return 0;
    }
}

const Q38GGUFMeta *q38_gguf_find_meta(const Q38GGUF *gguf, const char *key)
{
    if (!gguf || !key) return NULL;
    for (uint64_t i = 0; i < gguf->metadata_count; ++i) {
        if (q38_span_equal(gguf->metadata[i].key, key)) return &gguf->metadata[i];
    }
    return NULL;
}

const Q38GGUFTensor *q38_gguf_find_tensor(const Q38GGUF *gguf,
                                          const char *name)
{
    if (!gguf || !name) return NULL;
    for (uint64_t i = 0; i < gguf->tensor_count; ++i) {
        if (q38_span_equal(gguf->tensors[i].name, name)) return &gguf->tensors[i];
    }
    return NULL;
}

int q38_gguf_meta_u32(const Q38GGUF *gguf, const char *key, uint32_t *value)
{
    const Q38GGUFMeta *meta = q38_gguf_find_meta(gguf, key);
    if (!meta || !value || meta->type != Q38_GGUF_META_UINT32 ||
        meta->data_size != 4) return 0;
    *value = (uint32_t)meta->data[0] | (uint32_t)meta->data[1] << 8 |
             (uint32_t)meta->data[2] << 16 | (uint32_t)meta->data[3] << 24;
    return 1;
}

int q38_gguf_meta_string(const Q38GGUF *gguf, const char *key,
                         Q38GGUFString *value)
{
    const Q38GGUFMeta *meta = q38_gguf_find_meta(gguf, key);
    if (!meta || !value || meta->type != Q38_GGUF_META_STRING) return 0;
    value->data = (const char *)meta->data;
    value->length = meta->data_size;
    return 1;
}

int q38_gguf_meta_array_string(const Q38GGUFMeta *meta, uint64_t index,
                               Q38GGUFString *value)
{
    if (!meta || !value || meta->type != Q38_GGUF_META_ARRAY ||
        meta->array_type != Q38_GGUF_META_STRING || index >= meta->count) return 0;
    Q38Reader reader = { meta->data, meta->data, meta->data + meta->data_size, 0 };
    for (uint64_t i = 0; i <= index; ++i) {
        if (!q38_string(&reader, value)) return 0;
    }
    return 1;
}

int q38_gguf_meta_array_strings(const Q38GGUFMeta *meta,
                                Q38GGUFString *values, uint64_t capacity)
{
    if (!meta || !values || meta->type != Q38_GGUF_META_ARRAY ||
        meta->array_type != Q38_GGUF_META_STRING || capacity < meta->count) return 0;
    Q38Reader reader = { meta->data, meta->data, meta->data + meta->data_size, 0 };
    for (uint64_t i = 0; i < meta->count; ++i) {
        if (!q38_string(&reader, &values[i])) return 0;
    }
    return !reader.failed && reader.at == reader.end;
}

int q38_gguf_meta_array_i32(const Q38GGUFMeta *meta, uint64_t index,
                            int32_t *value)
{
    if (!meta || !value || meta->type != Q38_GGUF_META_ARRAY ||
        meta->array_type != Q38_GGUF_META_INT32 || index >= meta->count ||
        index > UINT64_MAX / 4 || index * 4 + 4 > meta->data_size) return 0;
    const uint8_t *data = meta->data + index * 4;
    const uint32_t raw = (uint32_t)data[0] | (uint32_t)data[1] << 8 |
                         (uint32_t)data[2] << 16 | (uint32_t)data[3] << 24;
    memcpy(value, &raw, sizeof(raw));
    return 1;
}

static int q38_parse_metadata(Q38GGUF *gguf, Q38Reader *reader)
{
    for (uint64_t i = 0; i < gguf->metadata_count; ++i) {
        Q38GGUFMeta *meta = &gguf->metadata[i];
        if (!q38_string(reader, &meta->key)) return 0;
        meta->type = q38_u32(reader);
        if (reader->failed || meta->type > Q38_GGUF_META_FLOAT64) return 0;

        if (meta->type == Q38_GGUF_META_STRING) {
            Q38GGUFString string;
            if (!q38_string(reader, &string)) return 0;
            meta->data = (const uint8_t *)string.data;
            meta->data_size = string.length;
            meta->count = 1;
        } else if (meta->type == Q38_GGUF_META_ARRAY) {
            meta->array_type = q38_u32(reader);
            meta->count = q38_u64(reader);
            meta->data = reader->at;
            if (meta->array_type == Q38_GGUF_META_ARRAY ||
                !q38_skip_meta_value(reader, meta->array_type, meta->count)) return 0;
            meta->data_size = (uint64_t)(reader->at - meta->data);
        } else {
            const uint64_t size = q38_meta_scalar_size(meta->type);
            const uint8_t *data;
            if (!size || !q38_take(reader, size, &data)) return 0;
            meta->data = data;
            meta->data_size = size;
            meta->count = 1;
        }
    }
    return 1;
}

static int q38_parse_tensors(Q38GGUF *gguf, Q38Reader *reader)
{
    for (uint64_t i = 0; i < gguf->tensor_count; ++i) {
        Q38GGUFTensor *tensor = &gguf->tensors[i];
        if (!q38_string(reader, &tensor->name)) return 0;
        tensor->n_dims = q38_u32(reader);
        if (reader->failed || tensor->n_dims == 0 ||
            tensor->n_dims > Q38_GGUF_MAX_DIMS) return 0;

        uint64_t elements = 1;
        for (uint32_t dim = 0; dim < tensor->n_dims; ++dim) {
            tensor->shape[dim] = q38_u64(reader);
            if (!tensor->shape[dim] ||
                elements > UINT64_MAX / tensor->shape[dim]) return 0;
            elements *= tensor->shape[dim];
        }
        tensor->type = q38_u32(reader);
        tensor->offset = q38_u64(reader);

        const uint32_t block_elements = q38_ggml_block_elements(tensor->type);
        const uint32_t block_bytes = q38_ggml_block_bytes(tensor->type);
        if (!block_elements || !block_bytes ||
            tensor->shape[0] % block_elements != 0 ||
            elements % block_elements != 0 ||
            elements / block_elements > UINT64_MAX / block_bytes) return 0;
        tensor->nbytes = elements / block_elements * block_bytes;
    }
    return !reader->failed;
}

int q38_gguf_open(Q38GGUF *gguf, const char *path)
{
    if (!gguf || !path) return 0;
    memset(gguf, 0, sizeof(*gguf));
    gguf->fd = -1;

    gguf->fd = open(path, O_RDONLY);
    if (gguf->fd < 0) {
        perror(path);
        return 0;
    }
    struct stat stat_buffer;
    if (fstat(gguf->fd, &stat_buffer) != 0 || stat_buffer.st_size < 24) {
        fprintf(stderr, "qwen38: invalid GGUF file size for %s\n", path);
        q38_gguf_close(gguf);
        return 0;
    }
    gguf->mapping_size = (uint64_t)stat_buffer.st_size;
    gguf->mapping = (const uint8_t *)mmap(NULL, (size_t)gguf->mapping_size,
                                          PROT_READ, MAP_PRIVATE, gguf->fd, 0);
    if (gguf->mapping == MAP_FAILED) {
        gguf->mapping = NULL;
        perror("mmap");
        q38_gguf_close(gguf);
        return 0;
    }

    Q38Reader reader = { gguf->mapping, gguf->mapping,
                         gguf->mapping + gguf->mapping_size, 0 };
    const uint32_t magic = q38_u32(&reader);
    gguf->version = q38_u32(&reader);
    gguf->tensor_count = q38_u64(&reader);
    gguf->metadata_count = q38_u64(&reader);
    if (magic != Q38_GGUF_MAGIC || gguf->version < 2 || gguf->version > 3 ||
        gguf->tensor_count == 0 || gguf->tensor_count > SIZE_MAX / sizeof(*gguf->tensors) ||
        gguf->metadata_count > SIZE_MAX / sizeof(*gguf->metadata)) {
        fprintf(stderr, "qwen38: unsupported or corrupt GGUF header in %s\n", path);
        q38_gguf_close(gguf);
        return 0;
    }

    gguf->tensors = (Q38GGUFTensor *)calloc((size_t)gguf->tensor_count,
                                             sizeof(*gguf->tensors));
    gguf->metadata = (Q38GGUFMeta *)calloc((size_t)gguf->metadata_count,
                                           sizeof(*gguf->metadata));
    if (!gguf->tensors || (gguf->metadata_count && !gguf->metadata) ||
        !q38_parse_metadata(gguf, &reader) || !q38_parse_tensors(gguf, &reader)) {
        fprintf(stderr, "qwen38: malformed GGUF metadata in %s\n", path);
        q38_gguf_close(gguf);
        return 0;
    }

    gguf->alignment = 32;
    uint32_t declared_alignment;
    if (q38_gguf_meta_u32(gguf, "general.alignment", &declared_alignment)) {
        if (!declared_alignment || (declared_alignment & (declared_alignment - 1)) != 0) {
            fprintf(stderr, "qwen38: invalid GGUF alignment %u\n", declared_alignment);
            q38_gguf_close(gguf);
            return 0;
        }
        gguf->alignment = declared_alignment;
    }
    const uint64_t descriptor_end = (uint64_t)(reader.at - reader.base);
    if (descriptor_end > UINT64_MAX - (gguf->alignment - 1u)) {
        q38_gguf_close(gguf);
        return 0;
    }
    gguf->data_offset = (descriptor_end + gguf->alignment - 1u) &
                        ~((uint64_t)gguf->alignment - 1u);
    if (gguf->data_offset > gguf->mapping_size) {
        fprintf(stderr, "qwen38: GGUF tensor data begins past end of file\n");
        q38_gguf_close(gguf);
        return 0;
    }

    for (uint64_t i = 0; i < gguf->tensor_count; ++i) {
        Q38GGUFTensor *tensor = &gguf->tensors[i];
        if (tensor->offset > gguf->mapping_size - gguf->data_offset ||
            tensor->nbytes > gguf->mapping_size - gguf->data_offset - tensor->offset) {
            fprintf(stderr, "qwen38: tensor outside GGUF file: %.*s\n",
                    (int)tensor->name.length, tensor->name.data);
            q38_gguf_close(gguf);
            return 0;
        }
        tensor->data = gguf->mapping + gguf->data_offset + tensor->offset;
    }
    return 1;
}

void q38_gguf_close(Q38GGUF *gguf)
{
    if (!gguf) return;
    if (gguf->mapping) munmap((void *)gguf->mapping, (size_t)gguf->mapping_size);
    if (gguf->fd >= 0) close(gguf->fd);
    free(gguf->tensors);
    free(gguf->metadata);
    memset(gguf, 0, sizeof(*gguf));
    gguf->fd = -1;
}
