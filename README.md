<div align="center">

# Run Qwen3.8-27B on a Laptop CPU

**A native C inference engine for local Qwen3.8-27B chat and work.**<br>
No GPU, CUDA, Python, PyTorch, model conversion, or external inference runtime.

**27B parameters · single CPU · 15.24 GiB peak RSS · best measured TPOT 0.568 s/token (1.76 token/s)**

[简体中文](README.zh-CN.md) · [Architecture](docs/ARCHITECTURE.md) · [Optimizations](docs/OPTIMIZATIONS.md) · [Correctness](docs/CORRECTNESS.md)

</div>

## Quick start

On Ubuntu, Debian, or Windows with WSL2:

```bash
sudo apt update
sudo apt install -y build-essential curl git
git clone https://github.com/shyringo/qwen3.8-27b-in-c.git
cd qwen3.8-27b-in-c
./qwen38.sh
```

On macOS, install Apple's command-line tools and OpenMP first:

```bash
xcode-select --install
brew install libomp
git clone https://github.com/shyringo/qwen3.8-27b-in-c.git
cd qwen3.8-27b-in-c
./qwen38.sh
```

The launcher builds the engine, resumes the 17.1 GB Q4_K_M download from
ModelScope when needed, and opens an interactive conversation. Enter `/reset`
for a new conversation and `/exit` to quit.

Run one request:

```bash
./qwen38.sh --prompt "科技的边界在哪里？" --max-tokens 256
```

Other useful forms:

```bash
./qwen38.sh --prompt-file request.txt
./qwen38.sh --no-thinking
./qwen38.sh --reasoning-effort low
./qwen38.sh --system "You are a careful C code reviewer."
```

Context length and CPU threads can be changed without rebuilding:

```bash
QWEN38_CONTEXT=8192 ./qwen38.sh
QWEN38_THREADS=6 ./qwen38.sh
```

Run the fixed 32-token greedy benchmark:

```bash
./scripts/benchmark.sh
```

The defaults are a 4,096-token context, a 1,024-token output limit per turn,
and at most 12 CPU threads. The model is stored under
`~/.cache/qwen3.8-27b-in-c/model`; set `QWEN38_MODEL_DIR` to use another
location.

## Environment requirements

- A 64-bit POSIX system, a C99 compiler, `make`, and `curl`. Windows users
  should use WSL2.
- At least 18 GB of free disk space for the pinned Q4_K_M GGUF.
- About 20 GiB of memory available to the inference process is recommended.
  Peak RSS measured at context 4,096 is 15.24 GiB; extra headroom keeps the
  operating system responsive.
- An x86-64 CPU with AVX2 is recommended for the optimized kernels. A tested
  portable scalar build is available for other 64-bit POSIX CPUs.

Keeping the model on a native Linux or macOS filesystem gives the operating
system the best chance to cache its 17 GB mapping. Under WSL2, the default
`~/.cache` location is on the Linux filesystem.

## Measured performance

The following are best wall-clock results from the fixed benchmark, not an
estimate:

| mode | TTFT | TPOT | throughput |
|---|---:|---:|---:|
| ordinary greedy decode (default) | 4.001 s | **0.568 s/token** | **1.76 token/s** |
| MTP verification, depth 3 | 5.349 s | 0.806 s/token | 1.24 token/s |
| MTP verification, depth 4 | 5.233 s | 0.928 s/token | 1.08 token/s |

Reference environment: Intel Core i5-1340P laptop, 32 GB host memory, Windows
11 with WSL2 Ubuntu 22.04.5 (24 GiB guest limit), GCC 11.4, 12 OpenMP threads,
4,096-token context, pinned Q4_K_M weights on the WSL2 ext4 filesystem, warm
model pages, a fixed 16-token prompt, and 32 greedy output tokens. The default
exact-SiLU path was used. `/usr/bin/time -v` measured 15,980,928 KiB
(15.24 GiB) peak RSS with no swap.

TTFT is especially sensitive to model-page cache state, storage, power mode,
temperature, and other running programs. TPOT is measured between generated
tokens and includes sampling and all model work. MTP remains optional because
its verification cost exceeded its acceptance benefit on this workload.

## Native inference engine

This repository implements the Qwen3.8-27B graph directly in C and reads the
single-file GGUF in place. It is not a wrapper around llama.cpp or another
inference runtime.

- All 64 target layers: 48 Gated DeltaNet and 16 causal full-attention layers.
- Recurrent convolution and DeltaNet state, KV cache, RoPE, RMSNorm, dense
  SwiGLU, model-matched NFC and ByteLevel BPE tokenization, sampling,
  reasoning control, and multi-turn chat.
- Packed Q4_K/Q5_K/Q6_K kernels, Q8_K activation quantization, AVX2/VNNI
  integer dot products, and OpenMP row parallelism.
- Layer-major prompt batching, packed-weight decode reuse across tokens,
  activation reuse across projections, memory-mapped weights, and resident
  cross-turn state.
- Exact SiLU by default and output-equivalent transactional greedy MTP as an
  explicit experimental option.

See [Optimizations](docs/OPTIMIZATIONS.md) for implementation details and code
provenance. [Architecture](docs/ARCHITECTURE.md) describes the graph, state,
batching, and memory layout.

## Inference correctness

Correctness is checked with the same pinned Q4_K_M weights. Native layer-major
prefill and native token-at-a-time evaluation produce byte-identical 248,320
dimensional logits for the fixed oracle prompt. An independent no-repack
evaluator returns the same top-five token order. Native transactional MTP also
reproduces ordinary native greedy decoding token for token.

```bash
make strict
make portable
```

The model hash, oracle SHA, logits comparison, token IDs, and test scope are in
[Correctness](docs/CORRECTNESS.md).

## Model, license, and attribution

The launcher downloads
[Unsloth's Qwen3.8-27B Q4_K_M GGUF](https://www.modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF)
from ModelScope first, with a Hugging Face fallback. The official
[Qwen3.8-27B model](https://www.modelscope.cn/models/Qwen/Qwen3.8-27B), the
[Qwen3.8 project](https://github.com/QwenLM/Qwen3.8), and this repository use
the Apache License 2.0. Model weights are downloaded to the user's cache and
are not included here.

The byte-level BPE foundation was adapted from
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c). See
[NOTICE](NOTICE) for attribution and third-party details.
