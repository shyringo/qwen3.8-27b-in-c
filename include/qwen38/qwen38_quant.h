#ifndef QWEN38_QUANT_H
#define QWEN38_QUANT_H

#include <stdint.h>

#include "qwen38_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q38_Q8_K_BLOCK_SIZE 256

typedef struct {
    float scale;
    int8_t quants[Q38_Q8_K_BLOCK_SIZE];
    int16_t sums[Q38_Q8_K_BLOCK_SIZE / 16];
} Q38Q8KBlock;

float q38_f16_to_f32(uint16_t value);

/* Quantize an activation row once, then reuse it across projections that
 * consume the same input. length must be divisible by 256. */
int q38_quantize_q8_k(Q38Q8KBlock *output, const float *input,
                      uint64_t length);

/* Decode one logical row without materializing the rest of the tensor. */
int q38_tensor_row_f32(float *output, const Q38GGUFTensor *tensor,
                       uint64_t row);

int q38_tensor_dot_row_f32(float *output, const float *input,
                            const Q38GGUFTensor *tensor, uint64_t row);

/* Matrix layout follows GGUF: shape[0] is input width, shape[1] is output rows. */
int q38_tensor_gemv_f32(float *output, const float *input,
                        const Q38GGUFTensor *tensor);

/* Integer dot-product path for Q4_K, Q5_K, and Q6_K matrices. */
int q38_tensor_gemv_q8_k(float *output, const Q38Q8KBlock *input,
                         uint64_t input_length,
                         const Q38GGUFTensor *tensor);

/* Batch-major output. Weight rows are the outer loop so a resident row serves
 * every input before the kernel advances through the model file. */
int q38_tensor_gemm_f32(float *output, const float *input,
                        uint32_t batch_size, uint64_t input_length,
                        const Q38GGUFTensor *tensor);
int q38_tensor_gemm_q8_k(float *output, const Q38Q8KBlock *input,
                         uint32_t batch_size, uint64_t input_length,
                         const Q38GGUFTensor *tensor);

#ifdef __cplusplus
}
#endif

#endif
