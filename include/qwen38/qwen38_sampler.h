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
    float presence_penalty;
    uint32_t top_k;
    uint8_t *presence;
    uint32_t presence_size;
} Q38Sampler;

void q38_sampler_init(Q38Sampler *sampler, uint64_t seed);
void q38_sampler_observe(Q38Sampler *sampler, uint32_t token);
int q38_sample(Q38Sampler *sampler, const float *logits,
               uint32_t vocabulary_size, uint32_t *token);

#ifdef __cplusplus
}
#endif

#endif
