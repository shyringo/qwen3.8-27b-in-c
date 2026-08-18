#define _POSIX_C_SOURCE 200809L

#include "qwen38_model.h"

#include <float.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define BENCH_TOKENS 16u

static const uint32_t prompt[] = {
    248045, 846, 198, 116348, 111764, 109994, 10992, 248046,
    198, 248045, 74455, 198, 248068, 271, 248069, 271
};

static double now_seconds(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

static uint32_t argmax(const float *logits, uint32_t count)
{
    uint32_t best = 0;
    float maximum = -FLT_MAX;
    for (uint32_t id = 0; id < count; ++id) {
        if (logits[id] > maximum) {
            maximum = logits[id];
            best = id;
        }
    }
    return best;
}

static int prepare(Q38Model *model, const float **logits)
{
    q38_model_reset(model);
    return q38_model_prefill(
        model, prompt, (uint32_t)(sizeof(prompt) / sizeof(prompt[0])), logits);
}

static int run_baseline(Q38Model *model, uint32_t *tokens, double *elapsed)
{
    const float *logits = NULL;
    if (!prepare(model, &logits)) return 0;
    tokens[0] = argmax(logits, q38_model_vocab_size(model));
    const double started = now_seconds();
    for (uint32_t i = 1; i < BENCH_TOKENS; ++i) {
        if (!q38_model_forward_token(model, tokens[i - 1u], &logits)) return 0;
        tokens[i] = argmax(logits, q38_model_vocab_size(model));
    }
    *elapsed = now_seconds() - started;
    return 1;
}

static int run_speculative(Q38Model *model, uint32_t depth,
                           const uint32_t *expected, double *elapsed,
                           uint32_t *transactions)
{
    const float *logits = NULL;
    if (!prepare(model, &logits)) return 0;
    uint32_t actual[BENCH_TOKENS];
    actual[0] = argmax(logits, q38_model_vocab_size(model));
    uint32_t produced = 1;
    *transactions = 0;
    const double started = now_seconds();
    while (produced < BENCH_TOKENS) {
        uint32_t batch[4];
        uint32_t count = 0;
        uint32_t requested = BENCH_TOKENS - produced;
        if (requested > depth) requested = depth;
        if (!q38_model_speculative_greedy(
                model, actual[produced - 1u], 248046, requested,
                batch, &count, &logits)) return 0;
        ++*transactions;
        for (uint32_t i = 0; i < count && produced < BENCH_TOKENS; ++i)
            actual[produced++] = batch[i];
    }
    *elapsed = now_seconds() - started;
    return memcmp(actual, expected, sizeof(actual)) == 0;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }
    Q38Model *model = q38_model_open_gguf(argv[1], 64);
    if (!model || !q38_model_enable_mtp(model)) {
        q38_model_close(model);
        return 1;
    }
    uint32_t expected[BENCH_TOKENS];
    double elapsed = 0.0;
    if (!run_baseline(model, expected, &elapsed)) {
        q38_model_close(model);
        return 1;
    }
    printf("mode=baseline elapsed=%.6f tpot=%.6f\n",
           elapsed, elapsed / (BENCH_TOKENS - 1u));
    for (uint32_t depth = 1; depth <= 4; ++depth) {
        uint32_t transactions = 0;
        if (!run_speculative(model, depth, expected, &elapsed, &transactions)) {
            fprintf(stderr, "depth %u changed the greedy token sequence\n", depth);
            q38_model_close(model);
            return 1;
        }
        printf("mode=mtp depth=%u transactions=%u elapsed=%.6f tpot=%.6f\n",
               depth, transactions, elapsed,
               elapsed / (BENCH_TOKENS - 1u));
        fflush(stdout);
    }
    q38_model_close(model);
    return 0;
}
