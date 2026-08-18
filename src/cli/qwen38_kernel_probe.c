#include "qwen38_gguf.h"
#include "qwen38_quant.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL.gguf TENSOR ROW\n", argv[0]);
        return 2;
    }
    char *end = NULL;
    const unsigned long long requested_row = strtoull(argv[3], &end, 10);
    if (!end || *end) {
        fprintf(stderr, "invalid row: %s\n", argv[3]);
        return 2;
    }

    Q38GGUF gguf;
    if (!q38_gguf_open(&gguf, argv[1])) return 1;
    const Q38GGUFTensor *tensor = q38_gguf_find_tensor(&gguf, argv[2]);
    if (!tensor || tensor->n_dims != 2 || requested_row >= tensor->shape[1]) {
        fprintf(stderr, "missing matrix or row outside matrix: %s[%llu]\n",
                argv[2], requested_row);
        q38_gguf_close(&gguf);
        return 1;
    }
    float *input = (float *)malloc((size_t)tensor->shape[0] * sizeof(float));
    if (!input) {
        q38_gguf_close(&gguf);
        return 1;
    }
    for (uint64_t i = 0; i < tensor->shape[0]; ++i) {
        input[i] = (float)((int)(i % 17u) - 8) * 0.125f;
    }
    float output = 0.0f;
    const int ok = q38_tensor_dot_row_f32(&output, input, tensor,
                                           (uint64_t)requested_row);
    if (ok) printf("%.9g\n", output);
    free(input);
    q38_gguf_close(&gguf);
    return ok ? 0 : 1;
}
