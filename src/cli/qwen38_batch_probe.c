#define _POSIX_C_SOURCE 200809L

#include "qwen38_model.h"

#include <stdio.h>
#include <time.h>

static double now_seconds(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }
    static const uint32_t prompt[16] = {
        248045, 846, 198, 116348, 111764, 109994, 10992, 248046,
        198, 248045, 74455, 198, 248068, 271, 248069, 271
    };
    static const uint32_t sizes[] = {1, 2, 4, 8, 16};
    Q38Model *model = q38_model_open_gguf(argv[1], 16);
    if (!model) return 1;

    const float *logits = NULL;
    if (!q38_model_forward_token(model, prompt[0], &logits)) {
        q38_model_close(model);
        return 1;
    }
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        q38_model_reset(model);
        const double started = now_seconds();
        const int ok = q38_model_prefill(model, prompt, sizes[i], &logits);
        const double elapsed = now_seconds() - started;
        if (!ok) {
            q38_model_close(model);
            return 1;
        }
        printf("batch=%u elapsed=%.6f per_token=%.6f\n",
               sizes[i], elapsed, elapsed / sizes[i]);
        fflush(stdout);
    }
    q38_model_close(model);
    return 0;
}
