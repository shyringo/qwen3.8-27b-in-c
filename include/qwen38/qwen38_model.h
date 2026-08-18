#ifndef QWEN38_MODEL_H
#define QWEN38_MODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Q38Model Q38Model;

Q38Model *q38_model_open_gguf(const char *path, uint32_t context_length);
void q38_model_close(Q38Model *model);
void q38_model_reset(Q38Model *model);

/* Evaluate one token through the complete model. The returned logits remain owned by
 * the model and are replaced by the next call. Tokens must be submitted in order. */
int q38_model_forward_token(Q38Model *model, uint32_t token_id,
                            const float **logits);

/* Layer-major prompt evaluation. The output is the logits after the final
 * token, and recurrent/KV state is advanced by token_count positions. */
int q38_model_prefill(Q38Model *model, const uint32_t *tokens,
                      uint32_t token_count, const float **logits);

/* Enable and probe the checkpoint's native one-layer MTP head. MTP must be
 * enabled before the first target token is evaluated. */
int q38_model_enable_mtp(Q38Model *model);
int q38_model_mtp_forward(Q38Model *model, uint32_t token_id,
                          const float **logits);

/* Consume the pending target token and verify up to draft_count MTP proposals.
 * accepted_tokens contains newly available tokens after token_id; its final
 * token remains pending for the next call. This path is greedy and preserves
 * the target model's exact greedy token sequence. */
int q38_model_speculative_greedy(Q38Model *model, uint32_t token_id,
                                 uint32_t stop_token, uint32_t draft_count,
                                 uint32_t *accepted_tokens,
                                 uint32_t *accepted_count,
                                 const float **next_logits);

/* Correctness probe: evaluate through the first layer_count blocks and return the
 * hidden state. A fresh/reset model should be used when comparing layer boundaries. */
int q38_model_forward_token_layers(Q38Model *model, uint32_t token_id,
                                   uint32_t layer_count, float *hidden_out);

uint32_t q38_model_vocab_size(const Q38Model *model);
uint32_t q38_model_position(const Q38Model *model);
uint32_t q38_model_context_length(const Q38Model *model);

#ifdef __cplusplus
}
#endif

#endif
