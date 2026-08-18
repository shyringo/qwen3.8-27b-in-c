#ifndef QWEN38_TOKENIZER_H
#define QWEN38_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Q38Tokenizer Q38Tokenizer;

Q38Tokenizer *q38_tokenizer_open_gguf(const char *path);
void q38_tokenizer_close(Q38Tokenizer *tokenizer);

int q38_tokenizer_encode(Q38Tokenizer *tokenizer, const char *text,
                         size_t text_length, uint32_t *tokens,
                         size_t capacity);
int q38_tokenizer_decode_token(Q38Tokenizer *tokenizer, uint32_t token,
                               char *text, size_t capacity);
int q38_tokenizer_find(Q38Tokenizer *tokenizer, const char *text);
int q38_tokenizer_is_special(const Q38Tokenizer *tokenizer, uint32_t token);
uint32_t q38_tokenizer_eos(const Q38Tokenizer *tokenizer);
uint32_t q38_tokenizer_vocab_size(const Q38Tokenizer *tokenizer);

#ifdef __cplusplus
}
#endif

#endif
