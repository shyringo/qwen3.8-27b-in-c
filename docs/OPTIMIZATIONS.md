# Optimizations and provenance

The implementation has three clear sources: a small tokenizer foundation
adapted from `kimi-k3-in-c`, Qwen3.8-specific graph work derived from the
published checkpoint contract, and inference optimizations implemented in this
project.

## Adapted foundations

The byte-level BPE and Unicode-table foundation originated in
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c). It was reduced
to the Qwen GGUF tokenizer path and given explicit allocation-failure handling.
The retained license notice is recorded in [NOTICE](../NOTICE).

## Qwen3.8 model support

- The 64-layer hybrid graph: three Gated DeltaNet layers followed by one causal
  full-attention layer.
- Qwen3.8 normalization, convolution, recurrent-state update, gated attention,
  RoPE and dense SwiGLU equations.
- The 248,320-token GGUF tokenizer, including atomic control and user-defined
  tokens used by the official chat template.
- The checkpoint's additional MTP layer, shared embedding and shared output
  projection.

## Inference engineering implemented here

- **Layer-major prompt batching.** All prompt positions traverse a weight row
  before the runtime advances to the next row or layer, reducing repeated reads
  of the dense 27B checkpoint.
- **Multi-token packed-weight decode reuse.** Q4_K/Q5_K/Q6_K batch kernels
  unpack each weight block once for a tile of up to four tokens while keeping
  independent accumulators. The tiled result is bit-identical to separate
  GEMV calls.
- **Activation-quantization reuse.** Quantize a normalized activation once and
  share it across related projections such as QKV/gate/decay and gate/up.
- **Packed low-bit CPU kernels.** Compute directly from GGUF K-quant blocks with
  AVX2 integer dot products and OpenMP output-row parallelism; no expanded
  floating-point weight copy is allocated.
- **Vectorized recurrent kernels.** SIMD implementations cover DeltaNet dot and
  state updates and gated output. SiLU uses exact `expf` by default so the
  optimized build does not silently change the model's nonlinear function.
- **Layer-ordered recurrent batching.** Prompt batching shares matrix reads but
  preserves the sequential convolution, DeltaNet and causal-attention state
  transitions within every layer.
- **Memory-mapped model residency.** The operating system manages the 17 GB
  read-only weight mapping, allowing hot-cache decode without a second weight
  allocation.
- **Resident multi-turn inference.** Model mapping, recurrent state, KV cache
  and tokenizer remain alive across chat turns.
- **Cross-turn tail fusion.** A pending final token, thinking closure and EOS
  are folded into the next prompt prefill. This preserves official chat-template
  boundaries without paying separate single-token passes between turns.
- **Transactional MTP verification.** A recurrent-state checkpoint, rejected
  suffix rollback, confirmed-prefix replay and target-hidden MTP realignment
  make multi-token greedy verification output-equivalent to the target model.
- **Generated-token timing and practical defaults.** TTFT starts before prompt
  construction; TPOT spans actual token-ready timestamps and includes sampling.
  The launcher caps automatic CPU use at 12 threads, exposes one context
  control, and resumes ModelScope downloads with a Hugging Face fallback.

MTP remains an explicit experimental option because acceptance and rollback
cost depend on the prompt. On the reference 32-token workload, ordinary greedy
decode is faster, so speculation is not enabled by default.
