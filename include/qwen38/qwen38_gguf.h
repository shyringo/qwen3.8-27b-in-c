#ifndef QWEN38_GGUF_H
#define QWEN38_GGUF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q38_GGUF_MAX_DIMS 8

typedef enum {
    Q38_GGUF_META_UINT8   = 0,
    Q38_GGUF_META_INT8    = 1,
    Q38_GGUF_META_UINT16  = 2,
    Q38_GGUF_META_INT16   = 3,
    Q38_GGUF_META_UINT32  = 4,
    Q38_GGUF_META_INT32   = 5,
    Q38_GGUF_META_FLOAT32 = 6,
    Q38_GGUF_META_BOOL    = 7,
    Q38_GGUF_META_STRING  = 8,
    Q38_GGUF_META_ARRAY   = 9,
    Q38_GGUF_META_UINT64  = 10,
    Q38_GGUF_META_INT64   = 11,
    Q38_GGUF_META_FLOAT64 = 12
} Q38GGUFMetaType;

typedef enum {
    Q38_GGML_F32  = 0,
    Q38_GGML_F16  = 1,
    Q38_GGML_Q4_0 = 2,
    Q38_GGML_Q4_1 = 3,
    Q38_GGML_Q5_0 = 6,
    Q38_GGML_Q5_1 = 7,
    Q38_GGML_Q8_0 = 8,
    Q38_GGML_Q2_K = 10,
    Q38_GGML_Q3_K = 11,
    Q38_GGML_Q4_K = 12,
    Q38_GGML_Q5_K = 13,
    Q38_GGML_Q6_K = 14,
    Q38_GGML_BF16 = 30
} Q38GGMLType;

typedef struct {
    const char *data;
    uint64_t length;
} Q38GGUFString;

typedef struct {
    Q38GGUFString key;
    uint32_t type;
    uint32_t array_type;
    uint64_t count;
    const uint8_t *data;
    uint64_t data_size;
} Q38GGUFMeta;

typedef struct {
    Q38GGUFString name;
    uint32_t n_dims;
    uint64_t shape[Q38_GGUF_MAX_DIMS];
    uint32_t type;
    uint64_t offset;
    uint64_t nbytes;
    const uint8_t *data;
} Q38GGUFTensor;

typedef struct {
    int fd;
    const uint8_t *mapping;
    uint64_t mapping_size;
    uint32_t version;
    uint32_t alignment;
    uint64_t data_offset;
    uint64_t tensor_count;
    uint64_t metadata_count;
    Q38GGUFTensor *tensors;
    Q38GGUFMeta *metadata;
} Q38GGUF;

int q38_gguf_open(Q38GGUF *gguf, const char *path);
void q38_gguf_close(Q38GGUF *gguf);

const Q38GGUFTensor *q38_gguf_find_tensor(const Q38GGUF *gguf,
                                          const char *name);
const Q38GGUFMeta *q38_gguf_find_meta(const Q38GGUF *gguf,
                                      const char *key);

int q38_gguf_meta_u32(const Q38GGUF *gguf, const char *key, uint32_t *value);
int q38_gguf_meta_string(const Q38GGUF *gguf, const char *key,
                         Q38GGUFString *value);
int q38_gguf_meta_array_string(const Q38GGUFMeta *meta, uint64_t index,
                               Q38GGUFString *value);
int q38_gguf_meta_array_strings(const Q38GGUFMeta *meta,
                                Q38GGUFString *values, uint64_t capacity);
int q38_gguf_meta_array_i32(const Q38GGUFMeta *meta, uint64_t index,
                            int32_t *value);

uint32_t q38_ggml_block_elements(uint32_t type);
uint32_t q38_ggml_block_bytes(uint32_t type);

#ifdef __cplusplus
}
#endif

#endif
