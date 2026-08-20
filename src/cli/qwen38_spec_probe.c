#define _POSIX_C_SOURCE 200809L

#include "qwen38_model.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static const uint32_t prompt[] = {
    248045, 846, 198, 116348, 111764, 109994, 10992, 248046,
    198, 248045, 74455, 198, 248068, 271, 248069, 271
};

static const char *mtp_path;

static int baseline(const char *path, uint32_t *tokens, uint32_t count)
{
    Q38Model *model = q38_model_open_gguf(path, 64);
    const float *logits = NULL;
    int ok = model && q38_model_prefill(
        model, prompt, (uint32_t)(sizeof(prompt) / sizeof(prompt[0])), &logits);
    for (uint32_t i = 0; ok && i < count; ++i) {
        tokens[i] = argmax(logits, q38_model_vocab_size(model));
        if (i + 1u < count)
            ok = q38_model_forward_token(model, tokens[i], &logits);
    }
    q38_model_close(model);
    return ok;
}

static int speculative(const char *path, uint32_t *tokens, uint32_t count)
{
    Q38Model *model = q38_model_open_gguf(path, 64);
    if (model && mtp_path && !q38_model_attach_mtp_gguf(model, mtp_path)) {
        q38_model_close(model);
        model = NULL;
    }
    const float *logits = NULL;
    int ok = model && q38_model_enable_mtp(model) && q38_model_prefill(
        model, prompt, (uint32_t)(sizeof(prompt) / sizeof(prompt[0])), &logits);
    uint32_t produced = 0;
    uint32_t current = ok ? argmax(logits, q38_model_vocab_size(model)) : 0;
    if (ok) tokens[produced++] = current;
    while (ok && produced < count) {
        uint32_t batch[4];
        uint32_t batch_count = 0;
        uint32_t depth = count - produced;
        if (depth > 3u) depth = 3u;
        ok = q38_model_speculative_greedy(model, current, 248046, depth,
                                          batch, &batch_count, &logits);
        for (uint32_t i = 0; ok && i < batch_count && produced < count; ++i)
            tokens[produced++] = batch[i];
        if (ok) current = batch[batch_count - 1u];
    }
    q38_model_close(model);
    return ok;
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s MODEL.gguf [--spec-only]\n", argv[0]);
        return 2;
    }
    if (argc == 3 && strcmp(argv[2], "--spec-only") != 0)
        mtp_path = argv[2];
    if (argc == 3 && strcmp(argv[2], "--spec-only") == 0) {
        uint32_t tokens[3];
        if (!speculative(argv[1], tokens, 3)) return 1;
        printf("spec-only: %u %u %u\n", tokens[0], tokens[1], tokens[2]);
        return 0;
    }
    uint32_t expected[12];
    uint32_t actual[12];
    if (!baseline(argv[1], expected, 12) ||
        !speculative(argv[1], actual, 12)) return 1;
    for (uint32_t i = 0; i < 12; ++i) {
        printf("%u expected=%u actual=%u%s\n", i, expected[i], actual[i],
               expected[i] == actual[i] ? "" : " MISMATCH");
        if (expected[i] != actual[i]) return 1;
    }
    puts("speculative greedy: exact");
    return 0;
}
