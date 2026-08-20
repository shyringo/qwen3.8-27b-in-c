#include "qwen38_gguf.h"
#include "qwen38_tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

static void put_u32(FILE *file, uint32_t value)
{
    for (int i = 0; i < 4; ++i) fputc((int)(value >> (8 * i)) & 255, file);
}

static void put_u64(FILE *file, uint64_t value)
{
    for (int i = 0; i < 8; ++i) fputc((int)(value >> (8 * i)) & 255, file);
}

static void put_string(FILE *file, const char *string)
{
    const size_t length = strlen(string);
    put_u64(file, (uint64_t)length);
    fwrite(string, 1, length, file);
}

static int write_fixture(const char *path)
{
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    put_u32(file, 0x46554747u);
    put_u32(file, 3);
    put_u64(file, 1);
    put_u64(file, 8);

    put_string(file, "general.alignment");
    put_u32(file, Q38_GGUF_META_UINT32);
    put_u32(file, 32);

    put_string(file, "general.architecture");
    put_u32(file, Q38_GGUF_META_STRING);
    put_string(file, "qwen35");

    put_string(file, "test.values");
    put_u32(file, Q38_GGUF_META_ARRAY);
    put_u32(file, Q38_GGUF_META_INT32);
    put_u64(file, 6);
    const int32_t values[6] = { 4, 8, 15, 16, 23, 42 };
    for (int i = 0; i < 6; ++i) put_u32(file, (uint32_t)values[i]);

    put_string(file, "tokenizer.ggml.tokens");
    put_u32(file, Q38_GGUF_META_ARRAY);
    put_u32(file, Q38_GGUF_META_STRING);
    put_u64(file, 4);
    put_string(file, "a");
    put_string(file, "b");
    put_string(file, "<control>");
    put_string(file, "<think>");

    put_string(file, "tokenizer.ggml.token_type");
    put_u32(file, Q38_GGUF_META_ARRAY);
    put_u32(file, Q38_GGUF_META_INT32);
    put_u64(file, 4);
    put_u32(file, 1);
    put_u32(file, 1);
    put_u32(file, 3);
    put_u32(file, 4);

    put_string(file, "tokenizer.ggml.merges");
    put_u32(file, Q38_GGUF_META_ARRAY);
    put_u32(file, Q38_GGUF_META_STRING);
    put_u64(file, 1);
    put_string(file, "a b");

    put_string(file, "tokenizer.ggml.pre");
    put_u32(file, Q38_GGUF_META_STRING);
    put_string(file, "qwen2");

    put_string(file, "tokenizer.ggml.eos_token_id");
    put_u32(file, Q38_GGUF_META_UINT32);
    put_u32(file, 2);

    put_string(file, "test.weight");
    put_u32(file, 2);
    put_u64(file, 4);
    put_u64(file, 2);
    put_u32(file, Q38_GGML_F32);
    put_u64(file, 0);

    long offset = ftell(file);
    if (offset < 0) { fclose(file); return 0; }
    while ((offset++ & 31) != 0) fputc(0, file);
    const float tensor[8] = { 0.5f, -1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f };
    fwrite(tensor, sizeof(float), 8, file);
    return fclose(file) == 0;
}

static int test_quantized_block_layouts(void)
{
    static const struct {
        uint32_t type;
        uint32_t elements;
        uint32_t bytes;
    } cases[] = {
        { Q38_GGML_Q2_K, 256, 84 },
        { Q38_GGML_Q3_K, 256, 110 },
        { Q38_GGML_Q4_K, 256, 144 },
        { Q38_GGML_Q5_K, 256, 176 },
        { Q38_GGML_Q6_K, 256, 210 },
        { Q38_GGML_IQ2_XXS, 256, 66 },
        { Q38_GGML_IQ2_XS, 256, 74 },
        { Q38_GGML_IQ3_XXS, 256, 98 },
        { Q38_GGML_IQ1_S, 256, 50 },
        { Q38_GGML_IQ4_NL, 32, 18 },
        { Q38_GGML_IQ3_S, 256, 110 },
        { Q38_GGML_IQ2_S, 256, 82 },
        { Q38_GGML_IQ4_XS, 256, 136 },
        { Q38_GGML_IQ1_M, 256, 56 },
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        CHECK(q38_ggml_block_elements(cases[i].type) == cases[i].elements,
              "quantized block element count");
        CHECK(q38_ggml_block_bytes(cases[i].type) == cases[i].bytes,
              "quantized block byte count");
    }
    CHECK(q38_ggml_block_elements(UINT32_MAX) == 0,
          "reject unknown block element count");
    CHECK(q38_ggml_block_bytes(UINT32_MAX) == 0,
          "reject unknown block byte count");
    return 0;
}

int main(void)
{
    CHECK(test_quantized_block_layouts() == 0,
          "quantized GGUF block layouts");

    const char *path = "build/test_qwen38_gguf.gguf";
    CHECK(write_fixture(path), "write synthetic GGUF");

    Q38GGUF gguf;
    CHECK(q38_gguf_open(&gguf, path), "open synthetic GGUF");
    CHECK(gguf.version == 3 && gguf.tensor_count == 1 &&
          gguf.metadata_count == 8 && gguf.alignment == 32,
          "header metadata");

    Q38GGUFString architecture;
    CHECK(q38_gguf_meta_string(&gguf, "general.architecture", &architecture),
          "architecture string");
    CHECK(architecture.length == 6 && memcmp(architecture.data, "qwen35", 6) == 0,
          "architecture value");

    const Q38GGUFMeta *array = q38_gguf_find_meta(&gguf, "test.values");
    int32_t last = 0;
    CHECK(array && array->count == 6 && q38_gguf_meta_array_i32(array, 5, &last) &&
          last == 42, "integer metadata array");

    const Q38GGUFTensor *weight = q38_gguf_find_tensor(&gguf, "test.weight");
    CHECK(weight && weight->type == Q38_GGML_F32 && weight->n_dims == 2 &&
          weight->shape[0] == 4 && weight->shape[1] == 2 && weight->nbytes == 32,
          "tensor descriptor");
    float first = 0.0f, last_weight = 0.0f;
    memcpy(&first, weight->data, sizeof(first));
    memcpy(&last_weight, weight->data + 7 * sizeof(float), sizeof(last_weight));
    CHECK(first == 0.5f && last_weight == 64.0f, "mapped tensor data");

    q38_gguf_close(&gguf);

    Q38Tokenizer *tokenizer = q38_tokenizer_open_gguf(path);
    CHECK(tokenizer, "open tokenizer from synthetic GGUF");
    const char input[] = "<control><think>";
    uint32_t ids[4] = {0};
    CHECK(q38_tokenizer_encode(tokenizer, input, strlen(input), ids, 4) == 2,
          "encode control and user-defined tokens atomically");
    CHECK(q38_tokenizer_encode(tokenizer, input, strlen(input), ids, 1) < 0,
          "reject truncated tokenizer output");
    CHECK(ids[0] == 2 && ids[1] == 3, "added token ids");
    CHECK(q38_tokenizer_is_special(tokenizer, ids[0]),
          "control token is hidden output");
    CHECK(!q38_tokenizer_is_special(tokenizer, ids[1]),
          "user-defined token remains visible output");
    char decoded[32];
    CHECK(q38_tokenizer_decode_token(tokenizer, ids[1], decoded,
                                     sizeof(decoded)) == 7 &&
          strcmp(decoded, "<think>") == 0,
          "decode user-defined token literally");
    CHECK(q38_tokenizer_decode_token(tokenizer, ids[1], decoded, 4) < 0,
          "reject truncated decoded token");
    q38_tokenizer_close(tokenizer);
    puts("qwen38 gguf: ok");
    return 0;
}
