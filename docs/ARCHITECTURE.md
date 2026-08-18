# Qwen3.8-27B inference architecture

The native runtime consumes the single-file Q4_K_M GGUF directly. It does not
convert or unpack the complete model and does not call another inference
engine.

## Target model

| component | checkpoint value |
|---|---:|
| decoder layers | 64 |
| hidden width | 5,120 |
| dense FFN width | 17,408 |
| vocabulary | 248,320 |
| Gated DeltaNet layers | 48 |
| full-attention layers | 16 |
| full-attention interval | every fourth layer |
| full-attention query / KV heads | 24 / 4 |
| attention head width | 256 |
| MTP heads | one additional decoder layer |

Each target layer applies pre-normalization, either Gated DeltaNet or causal
full attention, a residual update, a second normalization, and a dense SwiGLU
FFN. The output normalization and shared vocabulary projection produce logits.

The recurrent layers keep a convolution window and a per-head DeltaNet state.
The full-attention layers keep an ordinary key/value cache. Both are retained
between generated tokens and chat turns.

## Prompt evaluation

Prompt tokens are evaluated layer by layer. For a layer, all prompt positions
share each weight-row traversal before the runtime advances to the next layer.
Recurrent state updates and causal attention remain ordered by position, so the
batched path has the same model semantics as token-at-a-time evaluation.

## Decode

Activations are quantized to Q8_K once for a projection input. Q4_K, Q5_K and
Q6_K matrices stay packed and are multiplied directly by integer SIMD kernels;
the runtime never materializes a floating-point copy of the 27B weights.
OpenMP divides output rows among CPU workers. During multi-token evaluation,
tiles of up to four activations share each packed-weight decode while retaining
an independent accumulator for every token. SiLU uses the exact scalar `expf`
definition by default.

The GGUF is mapped read-only. This lets the operating system keep hot pages in
RAM and evict them when another application needs memory. The model file is
17,106,775,008 bytes; runtime state is separate from the file mapping.

At context 4,096, target and MTP KV storage is about 544 MiB. DeltaNet and
convolution state use about 150 MiB. Enabling experimental MTP verification
adds a roughly 150 MiB rollback checkpoint.

On the reference machine, a complete 16-token prompt followed by eight output
tokens at context 4,096 reached 15,980,928 KiB (15.24 GiB) peak RSS with no swap.
The model mapping accounts for most of that value. A machine with at least
20 GiB available to the inference process is recommended so the operating
system and other applications retain headroom.

## Chat turns

The model mapping, recurrent state, KV cache and tokenizer remain resident for
the complete interactive session. The final sampled token has not yet entered
the model state when it becomes visible. At a turn boundary, the runtime keeps
that pending token together with the official thinking closure and EOS, then
prepends them to the next user message's layer-major prefill. This preserves the
chat template and avoids separate boundary-token forward passes. `/reset`
clears both model state and the pending turn tail.

## MTP verification

The checkpoint includes one trained next-token-prediction layer. The optional
`--mtp` path runs that layer autoregressively to propose a short block, then
evaluates the target model once over the block.

Before verification, the runtime checkpoints every recurrent target state. On
a rejection it restores the checkpoint and replays only the confirmed prefix.
It then rebuilds the MTP continuation from the target model's normalized hidden
states. Rejected draft tokens are never displayed or retained. This path is
currently limited to greedy generation; ordinary sampling remains the default.
