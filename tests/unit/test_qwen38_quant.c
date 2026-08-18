#include "qwen38_quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

static void set_half(uint8_t *data, uint16_t value)
{
    data[0] = (uint8_t)value;
    data[1] = (uint8_t)(value >> 8);
}

static int close_to(float actual, float expected)
{
    return fabsf(actual - expected) <= 1e-5f * fmaxf(1.0f, fabsf(expected));
}

static uint32_t next_random(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static int test_integer_kernel_against_decoded(uint32_t type, uint64_t bytes)
{
    uint8_t block[210] = {0};
    float input[256];
    float reconstructed[256];
    float decoded = 0.0f;
    float integer = 0.0f;
    uint32_t random = 0x38c99a4du + type;
    for (uint64_t i = 0; i < bytes; ++i)
        block[i] = (uint8_t)(next_random(&random) >> 24);

    if (type == Q38_GGML_Q4_K || type == Q38_GGML_Q5_K) {
        set_half(block, 0x2e66u);
        set_half(block + 2, 0x28cdu);
    } else {
        set_half(block + 208, 0x2e66u);
    }
    for (int i = 0; i < 256; ++i)
        input[i] = (float)((int)(next_random(&random) >> 20) - 2048) / 257.0f;

    Q38Q8KBlock q8k;
    CHECK(q38_quantize_q8_k(&q8k, input, 256),
          "nonuniform Q8_K activation quantization");
    for (int i = 0; i < 256; ++i)
        reconstructed[i] = q8k.scale * q8k.quants[i];

    Q38GGUFTensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.n_dims = 2;
    tensor.shape[0] = 256;
    tensor.shape[1] = 1;
    tensor.type = type;
    tensor.nbytes = bytes;
    tensor.data = block;
    CHECK(q38_tensor_gemv_f32(&decoded, reconstructed, &tensor),
          "decoded nonuniform GEMV");
    CHECK(q38_tensor_gemv_q8_k(&integer, &q8k, 256, &tensor),
          "integer nonuniform GEMV");
    CHECK(fabsf(decoded - integer) <= 2e-4f * fmaxf(1.0f, fabsf(decoded)),
          "integer kernel agrees with decoded nonuniform weights");
    return 0;
}

static int test_integer_batch_exact(uint32_t type, uint64_t block_bytes)
{
    enum { WIDTH = 512, ROWS = 3, TOKENS = 4 };
    uint8_t weights[3 * 2 * 210] = {0};
    float activations[TOKENS * WIDTH];
    Q38Q8KBlock quantized[TOKENS * 2];
    float batch[TOKENS * ROWS];
    float single[ROWS];
    uint32_t random = 0x91e10da5u + type;
    const uint64_t bytes = ROWS * 2u * block_bytes;
    for (uint64_t i = 0; i < bytes; ++i)
        weights[i] = (uint8_t)(next_random(&random) >> 24);
    for (int row = 0; row < ROWS; ++row) {
        for (int block = 0; block < 2; ++block) {
            uint8_t *data = weights + (row * 2 + block) * block_bytes;
            if (type == Q38_GGML_Q4_K || type == Q38_GGML_Q5_K) {
                set_half(data, (uint16_t)(0x2800u + row * 0x80u + block));
                set_half(data + 2, (uint16_t)(0x2400u + row * 0x40u + block));
            } else {
                set_half(data + 208,
                         (uint16_t)(0x2800u + row * 0x80u + block));
            }
        }
    }
    for (int i = 0; i < TOKENS * WIDTH; ++i)
        activations[i] = (float)((int)(next_random(&random) >> 19) - 4096) /
                         (float)(101 + (i % 7));
    CHECK(q38_quantize_q8_k(quantized, activations, TOKENS * WIDTH),
          "batched exact Q8_K activation quantization");

    Q38GGUFTensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.n_dims = 2;
    tensor.shape[0] = WIDTH;
    tensor.shape[1] = ROWS;
    tensor.type = type;
    tensor.nbytes = bytes;
    tensor.data = weights;
    CHECK(q38_tensor_gemm_q8_k(batch, quantized, TOKENS, WIDTH, &tensor),
          "integer tiled GEMM dispatch");
    for (int token = 0; token < TOKENS; ++token) {
        CHECK(q38_tensor_gemv_q8_k(single, quantized + token * 2,
                                    WIDTH, &tensor),
              "integer reference GEMV dispatch");
        CHECK(memcmp(batch + token * ROWS, single, sizeof(single)) == 0,
              "integer tiled GEMM is bit-exact with GEMV");
    }
    return 0;
}

static int test_block(uint8_t *block, uint32_t type, uint64_t bytes,
                      float expected)
{
    float input[256], output = 0.0f;
    for (int i = 0; i < 256; ++i) input[i] = 1.0f;
    Q38GGUFTensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.n_dims = 2;
    tensor.shape[0] = type == Q38_GGML_Q8_0 ? 32 : 256;
    tensor.shape[1] = 1;
    tensor.type = type;
    tensor.nbytes = bytes;
    tensor.data = block;
    CHECK(q38_tensor_gemv_f32(&output, input, &tensor), "quantized GEMV dispatch");
    CHECK(close_to(output, expected), "quantized GEMV value");
    if (type == Q38_GGML_Q4_K || type == Q38_GGML_Q5_K ||
        type == Q38_GGML_Q6_K) {
        Q38Q8KBlock quantized;
        float integer_output = 0.0f;
        CHECK(q38_quantize_q8_k(&quantized, input, 256),
              "Q8_K activation quantization");
        CHECK(q38_tensor_gemv_q8_k(&integer_output, &quantized, 256, &tensor),
              "integer GEMV dispatch");
        CHECK(close_to(integer_output, expected), "integer GEMV value");
    }
    return 0;
}

int main(void)
{
    CHECK(q38_f16_to_f32(0x3c00u) == 1.0f, "f16 one");
    CHECK(q38_f16_to_f32(0xc000u) == -2.0f, "f16 negative two");
    CHECK(q38_f16_to_f32(0x0001u) == 0x1p-24f, "f16 subnormal");

    float activation[256];
    for (int i = 0; i < 256; ++i) activation[i] = (float)(i - 128) / 128.0f;
    Q38Q8KBlock q8k;
    CHECK(q38_quantize_q8_k(&q8k, activation, 256), "Q8_K block");
    CHECK(q8k.quants[0] == -127 && q8k.quants[128] == 0,
          "Q8_K signed range");

    uint8_t q4[144] = {0};
    set_half(q4, 0x3c00u);
    set_half(q4 + 2, 0x3800u);
    for (int i = 0; i < 4; ++i) q4[4 + i] = 1;
    for (int i = 0; i < 4; ++i) q4[8 + i] = 2;
    memset(q4 + 16, 0x21, 128);
    CHECK(test_block(q4, Q38_GGML_Q4_K, sizeof(q4), 64.0f) == 0, "Q4_K block");

    uint8_t q4_matrix[288];
    memcpy(q4_matrix, q4, sizeof(q4));
    memcpy(q4_matrix + sizeof(q4), q4, sizeof(q4));
    Q38GGUFTensor q4_tensor;
    memset(&q4_tensor, 0, sizeof(q4_tensor));
    q4_tensor.n_dims = 2;
    q4_tensor.shape[0] = 256;
    q4_tensor.shape[1] = 2;
    q4_tensor.type = Q38_GGML_Q4_K;
    q4_tensor.nbytes = sizeof(q4_matrix);
    q4_tensor.data = q4_matrix;
    float q8_inputs[512];
    for (int i = 0; i < 256; ++i) {
        q8_inputs[i] = 1.0f;
        q8_inputs[256 + i] = 2.0f;
    }
    Q38Q8KBlock q8_batches[2];
    float q4_batch_output[4] = {0};
    CHECK(q38_quantize_q8_k(q8_batches, q8_inputs, 512),
          "batched Q8_K activation quantization");
    CHECK(q38_tensor_gemm_q8_k(q4_batch_output, q8_batches, 2, 256,
                                &q4_tensor),
          "Q4_K integer GEMM dispatch");
    CHECK(close_to(q4_batch_output[0], 64.0f) &&
          close_to(q4_batch_output[1], 64.0f) &&
          close_to(q4_batch_output[2], 128.0f) &&
          close_to(q4_batch_output[3], 128.0f),
          "Q4_K integer GEMM batch-major output");

    uint8_t q5[176] = {0};
    set_half(q5, 0x3c00u);
    set_half(q5 + 2, 0x3800u);
    for (int i = 0; i < 4; ++i) q5[4 + i] = 1;
    for (int i = 0; i < 4; ++i) q5[8 + i] = 2;
    memset(q5 + 48, 0x21, 128);
    CHECK(test_block(q5, Q38_GGML_Q5_K, sizeof(q5), 64.0f) == 0, "Q5_K block");

    uint8_t q6[210] = {0};
    memset(q6 + 192, 1, 16);
    set_half(q6 + 208, 0x3c00u);
    CHECK(test_block(q6, Q38_GGML_Q6_K, sizeof(q6), -8192.0f) == 0, "Q6_K block");

    uint8_t q8[34] = {0};
    set_half(q8, 0x3800u);
    memset(q8 + 2, 2, 32);
    CHECK(test_block(q8, Q38_GGML_Q8_0, sizeof(q8), 32.0f) == 0, "Q8_0 block");

    CHECK(test_integer_kernel_against_decoded(Q38_GGML_Q4_K, 144) == 0,
          "Q4_K nonuniform block");
    CHECK(test_integer_kernel_against_decoded(Q38_GGML_Q5_K, 176) == 0,
          "Q5_K nonuniform block");
    CHECK(test_integer_kernel_against_decoded(Q38_GGML_Q6_K, 210) == 0,
          "Q6_K nonuniform block");
    CHECK(test_integer_batch_exact(Q38_GGML_Q4_K, 144) == 0,
          "Q4_K tiled integer batch");
    CHECK(test_integer_batch_exact(Q38_GGML_Q5_K, 176) == 0,
          "Q5_K tiled integer batch");
    CHECK(test_integer_batch_exact(Q38_GGML_Q6_K, 210) == 0,
          "Q6_K tiled integer batch");

    float matrix[8] = {1, 2, 3, 4, -1, -2, -3, -4};
    float input[4] = {1, 1, 1, 1};
    float output[2] = {0};
    Q38GGUFTensor f32;
    memset(&f32, 0, sizeof(f32));
    f32.n_dims = 2;
    f32.shape[0] = 4;
    f32.shape[1] = 2;
    f32.type = Q38_GGML_F32;
    f32.data = (const uint8_t *)matrix;
    CHECK(q38_tensor_gemv_f32(output, input, &f32), "F32 GEMV dispatch");
    CHECK(output[0] == 10.0f && output[1] == -10.0f, "F32 GEMV value");

    float batch_input[8] = {1, 1, 1, 1, 2, 2, 2, 2};
    float batch_output[4] = {0};
    CHECK(q38_tensor_gemm_f32(batch_output, batch_input, 2, 4, &f32),
          "F32 GEMM dispatch");
    CHECK(batch_output[0] == 10.0f && batch_output[1] == -10.0f &&
          batch_output[2] == 20.0f && batch_output[3] == -20.0f,
          "F32 GEMM batch-major output");

    puts("qwen38 quant: ok");
    return 0;
}
