#define _POSIX_C_SOURCE 200809L

#include "qwen38_model.h"

#include <float.h>
#include <stdio.h>
#include <time.h>

static double now_seconds(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

static uint32_t argmax(const float *logits, uint32_t count, float *value)
{
    uint32_t best = 0;
    float maximum = -FLT_MAX;
    for (uint32_t id = 0; id < count; ++id) {
        if (logits[id] > maximum) {
            maximum = logits[id];
            best = id;
        }
    }
    if (value) *value = maximum;
    return best;
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s MODEL.gguf [MTP.gguf]\n", argv[0]);
        return 2;
    }
    static const uint32_t prompt[] = {
        248045, 846, 198, 116348, 111764, 109994, 10992, 248046,
        198, 248045, 74455, 198, 248068, 271, 248069, 271
    };
    Q38Model *model = q38_model_open_gguf(argv[1], 64);
    if (model && argc == 3 && !q38_model_attach_mtp_gguf(model, argv[2])) {
        q38_model_close(model);
        model = NULL;
    }
    if (!model || !q38_model_enable_mtp(model)) {
        q38_model_close(model);
        return 1;
    }
    const float *target_logits = NULL;
    const double prefill_started = now_seconds();
    if (!q38_model_prefill(model, prompt,
                           (uint32_t)(sizeof(prompt) / sizeof(prompt[0])),
                           &target_logits)) {
        q38_model_close(model);
        return 1;
    }
    printf("prefill=%.6f\n", now_seconds() - prefill_started);
    uint32_t current = argmax(target_logits, q38_model_vocab_size(model), NULL);
    int accepted = 0;
    for (int step = 0; step < 8; ++step) {
        const float *draft_logits = NULL;
        const double draft_started = now_seconds();
        if (!q38_model_mtp_forward(model, current, &draft_logits)) {
            q38_model_close(model);
            return 1;
        }
        float draft_value = 0.0f;
        const uint32_t draft = argmax(draft_logits,
                                      q38_model_vocab_size(model),
                                      &draft_value);
        const double draft_elapsed = now_seconds() - draft_started;

        const double target_started = now_seconds();
        if (!q38_model_forward_token(model, current, &target_logits)) {
            q38_model_close(model);
            return 1;
        }
        float target_value = 0.0f;
        const uint32_t target = argmax(target_logits,
                                       q38_model_vocab_size(model),
                                       &target_value);
        const double target_elapsed = now_seconds() - target_started;
        const int match = draft == target;
        accepted += match;
        printf("step=%d input=%u draft=%u:%.6f target=%u:%.6f "
               "accepted=%d draft_s=%.6f target_s=%.6f\n",
               step, current, draft, draft_value, target, target_value,
               match, draft_elapsed, target_elapsed);
        fflush(stdout);
        current = target;
    }
    printf("acceptance=%d/8\n", accepted);
    q38_model_close(model);
    return 0;
}
