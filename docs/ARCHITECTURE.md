# Qwen3.8-27B inference architecture

The native runtime consumes Unsloth Dynamic V3 GGUF weights directly. It does
not convert or expand the complete model and does not call another inference
engine. The 64-layer target and optional MTP sidecar are mapped independently;
older integrated 65-layer GGUF files remain compatible.

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
| optional MTP layers | one |

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

## Decode and quantized kernels

Activations are quantized to Q8_K once for a projection input. Q2_K through
Q6_K and the IQ1/IQ2/IQ3/IQ4 matrices used by the pinned files stay packed and
are multiplied directly by integer SIMD kernels; the runtime never
materializes a floating-point copy of the 27B weights. OpenMP divides output
rows among CPU workers. During multi-token evaluation, tiles of up to four
activations share each packed-weight decode while retaining an independent
accumulator for every token. x86 builds use F16C for exact FP16 scale
conversion when available, with a portable C fallback. SiLU uses the exact
scalar `expf` definition by default.

On machines with at least 12 GiB available, the IQ1 path may build a
SIMD-friendly view of IQ1_S block metadata at model load. The unpacked values
and accumulation order do not change, and full logits are byte-identical to
the original packed path. Lower-memory machines keep only the GGUF mapping.

Gated DeltaNet updates each recurrent-state row and computes the corresponding
output dot product in one traversal. Independent reduction lanes preserve the
original result bit for bit while avoiding a second state-matrix read.

The GGUF is mapped read-only. This lets the operating system keep hot pages in
RAM and evict them when another application needs memory. Runtime state and an
optional IQ1 metadata layout are separate from the file mapping.

At context 4,096, target and MTP KV storage is about 544 MiB. DeltaNet and
convolution state use about 150 MiB. Enabling experimental MTP verification
adds a roughly 150 MiB rollback checkpoint.

Release validation at context 4,096 produced these `/usr/bin/time -v`
measurements with no swap. The constrained IQ1 row is a normal first-token
request under an 8 GiB process-address limit:

| weights | peak RSS | recommended available memory |
|---|---:|---:|
| Dynamic V3 IQ1_M, constrained path | 6.01 GiB | 8 GiB minimum |
| Dynamic V3 IQ1_M, runtime repack | 9.30 GiB | about 12 GiB |
| Dynamic V3 Q4_K_M | 14.49 GiB | about 20 GiB |

The target files are 6,729,166,848 bytes for IQ1_M and 16,464,440,224 bytes
for Q4_K_M. The launcher selects IQ1_M below 20 GiB of currently available
memory and Q4_K_M otherwise; `QWEN38_QUANT` can override that choice. The IQ1
runtime repack disables itself when available memory or the process address
limit is below 12 GiB.

## Laptop scheduling

The runtime uses at most 12 workers by default. On WSL2 systems with more than
12 visible vCPUs, the launcher keeps those workers on a stable vCPU set unless
the user supplied an OpenMP configuration. Other POSIX systems retain their
native scheduler policy.

## Chat turns

The model mapping, recurrent state, KV cache and tokenizer remain resident for
the complete interactive session. The final sampled token has not yet entered
the model state when it becomes visible. At a turn boundary, the runtime keeps
that pending token together with the official thinking closure and EOS, then
prepends them to the next user message's layer-major prefill. This preserves the
chat template and avoids separate boundary-token forward passes. `/reset`
clears both model state and the pending turn tail.

## Split-checkpoint MTP verification

The optional `--mtp-model` file contains the trained layer 64, shared
embedding, normalization, and output tensors. Main and sidecar GGUF contracts
are checked independently and mapped separately. The `--mtp` path uses that
layer autoregressively to propose a short block, then evaluates the target
model once over the block.

Before verification, the runtime checkpoints every recurrent target state. On
a rejection it restores the checkpoint and replays only the confirmed prefix.
It then rebuilds the MTP continuation from the target model's normalized hidden
states. Rejected draft tokens are never displayed or retained. This path is
currently greedy-only and experimental because acceptance is prompt-dependent
and did not improve TPOT on the reference workload.
