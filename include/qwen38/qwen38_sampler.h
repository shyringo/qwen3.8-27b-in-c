#ifndef QWEN38_SAMPLER_H
#define QWEN38_SAMPLER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t state;
    float temperature;
    float top_p;
    uint32_t top_k;
} Q38Sampler;

void q38_sampler_init(Q38Sampler *sampler, uint64_t seed);
uint32_t q38_sample(Q38Sampler *sampler, const float *logits,
                    uint32_t vocabulary_size);

#ifdef __cplusplus
}
#endif

#endif
