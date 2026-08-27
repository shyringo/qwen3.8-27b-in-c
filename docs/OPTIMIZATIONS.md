# Optimizations and provenance

This implementation falls into three groups: reused code and data, adaptations
of published formats and public projects, and original engineering designed for
this Qwen3.8 runtime. The Qwen model graph, bounded GGUF reader, recurrent state,
sampling, and chat runtime were implemented for this project. Laptop-CPU
execution architecture was carried forward from
[deepseek-v4-flash-0731-in-c](https://github.com/shyringo/deepseek-v4-flash-0731-in-c)
and redesigned around Qwen3.8's dense hybrid graph.

## 1. Reused code and data

- The byte-level BPE and Unicode-category foundation was adapted from
  [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c). The retained
  path was narrowed to Qwen's GGUF tokenizer and extended with Unicode 9.0 NFC,
  strict UTF-8 validation, capacity checks and allocation-failure handling.
- The generated normalization tables come from the Unicode 9.0 Character
  Database under the Unicode data license.
- Fixed IQ codebook tables come from ggml under the MIT License. Packed
  low-bit decoding and kernel techniques are also adapted from ggml's work.
  The retained license is included in `third_party/GGML-LICENSE.txt`.

No external inference runtime is included, downloaded or invoked. The retained
license notices are recorded in [NOTICE](../NOTICE).

## 2. Published formats and ideas adapted here

- **Qwen3.8 model contract.** The 64-layer graph, Gated DeltaNet equations,
  causal full attention, dense SwiGLU, MTP layer, chat template and tensor
  shapes follow the official
  [Qwen3.8-27B checkpoint](https://www.modelscope.cn/models/Qwen/Qwen3.8-27B).
  Their C execution is implemented in this repository.
- **GGUF low-bit formats.** GGUF v3, Q2_K through Q6_K, Q8_0/Q8_K, and the
  IQ1/IQ2/IQ3/IQ4 encodings used by the pinned model files are compatibility
  targets published by the
  [ggml](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and
  [llama.cpp](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes)
  ecosystem. This project has its own bounded GGUF parser and CPU kernels; it
  does not link to either runtime.
- **Packed IQ kernel techniques.** IQ1/IQ2 codebook lookup, packed sign and
  index extraction, block-scale reconstruction, and architecture-specific
  integer-dot techniques are adapted from ggml's MIT-licensed CPU work and
  integrated into this runtime's row-partitioned and multi-token APIs.
- **CPU instruction-set adaptation.** AVX2, AVX-VNNI, BMI2, and F16C are used
  when the compiler target provides them. Other 64-bit POSIX CPUs retain the
  tested scalar C path.
- **Techniques carried into a dense hybrid model.** Direct computation from
  packed low-bit weights, one-time activation quantization, layer-major
  multi-token execution, memory-mapped weights, resident multi-turn state and
  laptop-aware thread caps build on techniques already used in
  [deepseek-v4-flash-0731-in-c](https://github.com/shyringo/deepseek-v4-flash-0731-in-c).
  They were redesigned around Qwen3.8's dense FFN and recurrent/attention
  schedule rather than copied as a model wrapper.

## 3. Project-specific original engineering

- **Recurrence-safe layer-major batching.** Each layer shares packed-weight
  reads across all prompt positions while convolution, DeltaNet and causal KV
  state still advance in exact token order. This removes repeated checkpoint
  scans without treating the recurrent layers as independent tokens.
- **Fused multi-token low-bit kernels.** K-quant and IQ kernels decode a packed
  block once for a four-token tile, keep independent accumulators, and produce
  output that is bit-identical to separate GEMV calls. Q8_K activations are
  quantized once and reused across related projections.
- **Memory-aware bit-exact IQ1 repacking.** When at least 12 GiB is available,
  IQ1_S block metadata is rearranged once into a SIMD-friendly runtime view.
  Constrained systems keep the original mapping-only path. Both routes produce
  byte-identical full logits.
- **Vectorized hybrid-state kernels with exact nonlinearities.** AVX2/VNNI
  covers packed integer dot products, DeltaNet state prediction/update and
  gated output. Exact SiLU remains the default so the optimized binary does not
  silently substitute a different model function.
- **Fused DeltaNet state traversal.** Recurrent-state update and the following
  output dot product share one traversal. Independent reduction lanes preserve
  full native logits bit for bit while reducing warm one-token latency by about
  2.7% on the reference laptop.
- **Cross-turn tail fusion.** A pending final token, thinking closure and EOS
  are folded into the next prompt prefill. Official chat-template boundaries
  are preserved without separate single-token model passes between turns.
- **Native resident chat API.** A bounded C HTTP/JSON path holds the model,
  tokenizer, runtime layouts and OpenMP worker pool across chat-completion
  requests. Each request receives clean recurrent and sampling state; SSE
  output combines split UTF-8 token bytes before emitting JSON deltas. The
  loopback-only server requires neither Python nor an external inference
  runtime.
- **Transactional MTP for recurrent state.** Target and MTP state checkpoints,
  rejected-suffix rollback, confirmed-prefix replay and hidden-state
  realignment make speculative greedy verification output-equivalent to
  ordinary target-model decoding.
- **Laptop-aware automatic scheduling.** The runtime caps automatic CPU use at
  12 workers. WSL2 hybrid systems receive a stable vCPU affinity only when the
  user has not supplied OpenMP settings.
- **Model-matched tokenizer validation.** Unicode 9.0 normalization, atomic
  control/user-defined tokens and ByteLevel BPE behavior are checked across the
  official normalization suite and the complete Unicode scalar range.
- **User-facing model bootstrap.** ModelScope-first resumable download, pinned
  revisions, exact size/SHA checks, hash-specific verification markers, disk
  capacity checks, memory-based model selection, and a native-filesystem cache
  turn a multi-gigabyte checkpoint into one command.

## Runtime consequences

The operating system manages the selected 6.27 or 15.33 GiB read-only model
mapping; the engine does not allocate an expanded floating-point model. TTFT
starts when a resident runtime accepts the request, and TPOT spans actual
token-ready timestamps including sampling and all model work.

Layer-major prompt batching reaches 0.253 s/token on a four-token target batch
in the reference environment. The optional IQ1 runtime layout reduced the
isolated warm one-token median from about 0.396 s to 0.362 s in its focused
A/B test; it is enabled only when the extra memory is available.

MTP remains an explicit experimental option because acceptance and rollback
cost depend on the prompt. On the reference 32-token workload, ordinary greedy
decode is faster, so speculation is not enabled by default.
