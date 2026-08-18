#include "qwen38_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL.gguf TEXT\n", argv[0]);
        return 2;
    }
    Q38Tokenizer *tokenizer = q38_tokenizer_open_gguf(argv[1]);
    if (!tokenizer) return 1;
    const size_t length = strlen(argv[2]);
    uint32_t *tokens = (uint32_t *)calloc(length + 16u, sizeof(uint32_t));
    if (!tokens) {
        q38_tokenizer_close(tokenizer);
        return 1;
    }
    const int count = q38_tokenizer_encode(tokenizer, argv[2], length,
                                            tokens, length + 16u);
    if (count >= 0) {
        printf("tokens=%d ids=", count);
        for (int i = 0; i < count; ++i) printf("%s%u", i ? "," : "", tokens[i]);
        putchar('\n');
    }
    free(tokens);
    q38_tokenizer_close(tokenizer);
    return count >= 0 ? 0 : 1;
}
