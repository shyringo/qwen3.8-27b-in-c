#define _POSIX_C_SOURCE 200809L

#include "qwen38_model.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

static void compare_reference(const float *logits, uint32_t vocabulary_size)
{
    const char *path = getenv("QWEN38_REFERENCE_LOGITS");
    if (!path || !*path) return;
    FILE *file = fopen(path, "rb");
    float *reference = malloc((size_t)vocabulary_size * sizeof(*reference));
    if (!file || !reference ||
        fread(reference, sizeof(*reference), vocabulary_size, file) !=
            vocabulary_size) {
        fprintf(stderr, "unable to read reference logits: %s\n", path);
        if (file) fclose(file);
        free(reference);
        return;
    }
    fclose(file);
    float maximum = 0.0f;
    uint32_t maximum_id = 0;
    double total = 0.0;
    double squared = 0.0;
    double dot = 0.0;
    double native_squared = 0.0;
    double reference_squared = 0.0;
    uint32_t top_ids[5] = {0};
    float top_values[5] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (uint32_t id = 0; id < vocabulary_size; ++id) {
        const float difference = fabsf(logits[id] - reference[id]);
        total += difference;
        squared += (double)difference * difference;
        dot += (double)logits[id] * reference[id];
        native_squared += (double)logits[id] * logits[id];
        reference_squared += (double)reference[id] * reference[id];
        if (difference > maximum) {
            maximum = difference;
            maximum_id = id;
        }
        int slot = 4;
        if (reference[id] <= top_values[slot]) continue;
        while (slot > 0 && reference[id] > top_values[slot - 1]) {
            top_values[slot] = top_values[slot - 1];
            top_ids[slot] = top_ids[slot - 1];
            --slot;
        }
        top_values[slot] = reference[id];
        top_ids[slot] = id;
    }
    printf("reference maxdiff=%.9g at=%u meanabs=%.9g rmse=%.9g cosine=%.12g\n",
           maximum, maximum_id, total / vocabulary_size,
           sqrt(squared / vocabulary_size),
           dot / sqrt(native_squared * reference_squared));
    printf("reference top5=");
    for (int i = 0; i < 5; ++i)
        printf("%s%u:%.9g", i ? "," : "", top_ids[i], top_values[i]);
    putchar('\n');
    free(reference);
}

static int dump_logits(const float *logits, uint32_t vocabulary_size)
{
    const char *path = getenv("QWEN38_DUMP_LOGITS");
    if (!path || !*path) return 1;
    FILE *file = fopen(path, "wb");
    if (!file) {
        perror(path);
        return 0;
    }
    const size_t written = fwrite(logits, sizeof(*logits), vocabulary_size,
                                  file);
    const int closed = fclose(file) == 0;
    const int ok = written == vocabulary_size && closed;
    if (!ok) fprintf(stderr, "unable to write logits: %s\n", path);
    return ok;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s MODEL.gguf TOKEN_ID [TOKEN_ID ...] [--prefill]\n",
                argv[0]);
        return 2;
    }

    const int prefill = strcmp(argv[argc - 1], "--prefill") == 0;
    const int token_count = argc - 2 - prefill;
    if (token_count < 1) return 2;
    uint32_t *tokens = malloc((size_t)token_count * sizeof(*tokens));
    if (!tokens) return 1;
    for (int i = 0; i < token_count; ++i) {
        char *end = NULL;
        const unsigned long token = strtoul(argv[i + 2], &end, 10);
        if (!end || *end != '\0' || token >= 248320) {
            free(tokens);
            return 2;
        }
        tokens[i] = (uint32_t)token;
    }

    const uint32_t context = token_count > 16 ? (uint32_t)token_count : 16u;
    Q38Model *model = q38_model_open_gguf(argv[1], context);
    if (!model) {
        free(tokens);
        return 1;
    }
    const float *logits = NULL;
    const double start = now_seconds();
    int ok = 1;
    if (prefill) {
        ok = q38_model_prefill(model, tokens, (uint32_t)token_count, &logits);
    } else {
        for (int i = 0; i < token_count && ok; ++i)
            ok = q38_model_forward_token(model, tokens[i], &logits);
    }
    const double elapsed = now_seconds() - start;
    if (ok) {
        uint32_t top_ids[5] = {0};
        float top_values[5] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (uint32_t id = 0; id < q38_model_vocab_size(model); ++id) {
            int slot = 4;
            if (logits[id] <= top_values[slot]) continue;
            while (slot > 0 && logits[id] > top_values[slot - 1]) {
                top_values[slot] = top_values[slot - 1];
                top_ids[slot] = top_ids[slot - 1];
                --slot;
            }
            top_values[slot] = logits[id];
            top_ids[slot] = id;
        }
        printf("tokens=%d mode=%s time=%.6f top5=",
               token_count, prefill ? "prefill" : "sequential", elapsed);
        for (int i = 0; i < 5; ++i)
            printf("%s%u:%.9g", i ? "," : "", top_ids[i], top_values[i]);
        putchar('\n');
        compare_reference(logits, q38_model_vocab_size(model));
        ok = dump_logits(logits, q38_model_vocab_size(model));
    }
    q38_model_close(model);
    free(tokens);
    return ok ? 0 : 1;
}
