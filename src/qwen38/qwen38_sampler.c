#include "qwen38_sampler.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

typedef struct {
    uint32_t id;
    float logit;
    float probability;
} Q38Candidate;

static uint64_t q38_random(Q38Sampler *sampler)
{
    uint64_t value = sampler->state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    sampler->state = value;
    return value * UINT64_C(2685821657736338717);
}

void q38_sampler_init(Q38Sampler *sampler, uint64_t seed)
{
    if (!sampler) return;
    sampler->state = seed ? seed : UINT64_C(0x9e3779b97f4a7c15);
    sampler->temperature = 1.0f;
    sampler->top_p = 0.95f;
    sampler->top_k = 20;
}

uint32_t q38_sample(Q38Sampler *sampler, const float *logits,
                    uint32_t vocabulary_size)
{
    if (!sampler || !logits || vocabulary_size == 0) return 0;
    if (sampler->temperature <= 0.0f || sampler->top_k <= 1) {
        uint32_t best = 0;
        for (uint32_t id = 1; id < vocabulary_size; ++id) {
            if (logits[id] > logits[best]) best = id;
        }
        return best;
    }
    uint32_t count = sampler->top_k;
    if (count > vocabulary_size) count = vocabulary_size;
    Q38Candidate *candidates = (Q38Candidate *)malloc(
        (size_t)count * sizeof(*candidates));
    if (!candidates) return 0;
    for (uint32_t i = 0; i < count; ++i) {
        candidates[i].id = 0;
        candidates[i].logit = -FLT_MAX;
        candidates[i].probability = 0.0f;
    }
    for (uint32_t id = 0; id < vocabulary_size; ++id) {
        if (logits[id] <= candidates[count - 1].logit) continue;
        uint32_t slot = count - 1;
        while (slot > 0 && logits[id] > candidates[slot - 1].logit) {
            candidates[slot] = candidates[slot - 1];
            --slot;
        }
        candidates[slot].id = id;
        candidates[slot].logit = logits[id];
    }
    float total = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
        candidates[i].probability = expf(
            (candidates[i].logit - candidates[0].logit) / sampler->temperature);
        total += candidates[i].probability;
    }
    float cumulative = 0.0f;
    uint32_t retained = count;
    const float top_p = sampler->top_p > 0.0f && sampler->top_p < 1.0f
        ? sampler->top_p : 1.0f;
    for (uint32_t i = 0; i < count; ++i) {
        cumulative += candidates[i].probability / total;
        if (cumulative >= top_p) {
            retained = i + 1;
            break;
        }
    }
    float retained_total = 0.0f;
    for (uint32_t i = 0; i < retained; ++i)
        retained_total += candidates[i].probability;
    const double unit = (double)(q38_random(sampler) >> 11) * 0x1.0p-53;
    const float target = (float)unit * retained_total;
    float running = 0.0f;
    uint32_t selected = candidates[retained - 1].id;
    for (uint32_t i = 0; i < retained; ++i) {
        running += candidates[i].probability;
        if (target < running) {
            selected = candidates[i].id;
            break;
        }
    }
    free(candidates);
    return selected;
}
