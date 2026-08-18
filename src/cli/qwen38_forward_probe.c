#define _POSIX_C_SOURCE 200809L

#include "qwen38_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_seconds(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL.gguf TOKEN_ID LAYER_COUNT\n", argv[0]);
        return 2;
    }
    const unsigned long token = strtoul(argv[2], NULL, 10);
    const unsigned long layers = strtoul(argv[3], NULL, 10);
    if (token >= 248320 || layers > 64) return 2;

    Q38Model *model = q38_model_open_gguf(argv[1], 16);
    if (!model) return 1;
    float hidden[5120];
    const double start = now_seconds();
    const int ok = q38_model_forward_token_layers(model, (uint32_t)token,
                                                   (uint32_t)layers, hidden);
    const double elapsed = now_seconds() - start;
    if (ok) {
        double sum = 0.0, sum_abs = 0.0;
        for (int i = 0; i < 5120; ++i) {
            sum += hidden[i];
            sum_abs += hidden[i] < 0.0f ? -hidden[i] : hidden[i];
        }
        printf("token=%lu layers=%lu time=%.6f sum=%.9g abs=%.9g first=",
               token, layers, elapsed, sum, sum_abs);
        for (int i = 0; i < 8; ++i) printf("%s%.9g", i ? "," : "", hidden[i]);
        putchar('\n');
    }
    q38_model_close(model);
    return ok ? 0 : 1;
}
