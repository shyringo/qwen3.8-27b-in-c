# Inference accuracy and execution fidelity

The optimized runtime adds zero accuracy loss beyond the chosen weight
quantization: optimized and native baseline paths are verified with 100%
bit-identical full logits for the same GGUF. Comparisons always use the exact
same checkpoint and quantization.
An IQ1_M runtime is not expected to reproduce BF16, FP8, or Q4_K_M logits. The
tests below isolate runtime execution from weight-quantization loss.

## Pinned model files

```text
Dynamic V3 IQ1_M
file:      Qwen3.8-27B-UD-IQ1_M.gguf
size:      6,729,166,848 bytes
revision:  1fa4a98544ac96043a10649853051f1d5e72a008
SHA-256:   1b5165a7149ea51e683c8eaf23372188ad9fc9d1a795386f7a1b558acf847dc6

Dynamic V3 Q4_K_M
file:      Qwen3.8-27B-UD-Q4_K_M.gguf
size:      16,464,440,224 bytes
revision:  ba7608d4e5e1f3ea3d016cebd1c972c42686e9da
SHA-256:   322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482

MTP sidecar (optional)
file:      mtp-Qwen3.8-27B-Q4_0.gguf
size:      1,369,590,656 bytes
SHA-256:   50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e
```

The first successful download is verified against the selected SHA-256 value
and gets a local verification marker. Later launches check the exact file size
and the GGUF model contract without hashing the complete file again. Set
`QWEN38_VERIFY=1` when running `scripts/get-model.sh` to force another full
hash pass.

## Unit and build checks

```bash
make strict
make portable
```

The tests cover bounded GGUF parsing, split and integrated checkpoint
contracts, BPE control/user-defined tokens, Unicode NFC, activation
quantization, sampling, and every low-bit format used by the pinned files:
Q2_K through Q6_K, Q8_0/Q8_K, plus IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
IQ3_XXS, IQ3_S, IQ4_NL, and IQ4_XS. Multi-token integer kernels are checked byte for
byte against separate GEMV calls for batches of one through five tokens.

The optional IQ1 runtime repack is checked against the original packed path.
The fused Gated DeltaNet traversal is checked against separate state update
and output-dot passes. Native one-token, 16-token sequential, and 16-token
layer-major runs produce bit-identical full logits; the portable scalar path
is checked independently.

`strict` enables native CPU/OpenMP paths and treats warnings as errors.
`portable` disables architecture-specific SIMD and OpenMP. The complete suite
also passes AddressSanitizer and UndefinedBehaviorSanitizer.

The C NFC implementation passes all 18,722 cases in the Unicode 9.0 official
normalization suite. Independent comparisons against Hugging Face `tokenizers`
0.22 match across the complete Unicode scalar range in 8,688 chunks and a
separate 200-case set covering multilingual text, emoji, combining marks and
chat-template tokens.

## Q4_K_M full-checkpoint oracle

The fixed direct-answer prompt is:

```text
<|im_start|>user
科技的边界在哪里？<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Its 16 token IDs are:

```text
248045 846 198 116348 111764 109994 10992 248046
198 248045 74455 198 248068 271 248069 271
```

For the pinned Dynamic V3 Q4_K_M file, the native prefill top five are:

```text
109455:25.0723038
2005:24.3620987
104362:18.2299805
116348:17.9985847
97785:17.0889854
```

The complete 248,320-float output has SHA-256:

```text
c41064cc8fa9b5bd8150b182fed48a9ca53fb1b59b2582c316e8e9ce545ba31b
```

An independent no-repack, float-KV evaluation of the same GGUF produces the
same five token IDs in the same order. The reference evaluator is
`llama-completion` built from llama.cpp commit
`7acdbb1f191d869bad8c5da9d4a2121defa340af`; it is an oracle only and is not a
runtime dependency of this project. Across all 248,320 logits, the observed
maximum absolute difference is 0.431469679 (token 33617), mean absolute
difference is 0.059229222, RMSE is 0.0747201506, and cosine similarity is
0.999270634387. The difference reflects different legal accumulation grouping
and execution order; it does not compare against a different precision
checkpoint.

The layer-major native prefill and native token-at-a-time evaluation produce
byte-identical full logits for this prompt. Greedy decoding begins with token
109455.

The independent comparison uses greedy sampling, a 64-token context and the
reference engine's `--no-repack` option. The prompt bytes are kept in
`tests/fixtures/qwen38_oracle_prompt.txt` so the chat-template boundary is part
of the test rather than an unstated assumption.

## Dynamic V3 IQ1_M full-checkpoint oracle

Native layer-major prefill and native token-at-a-time execution produce
byte-identical 248,320-float logits. The result is unchanged when the optional
SIMD-friendly runtime repack is enabled. Its SHA-256 is:

```text
bd6a05d14b66d2bde0a35494f570df0677391a4f83c7c76c8a8be25804a4adb8
```

Native top five:

```text
116348:19.4790115
109455:18.9361401
104362:17.174427
2005:16.9269104
96848:16.3994484
```

The same fixed independent evaluator returns the same first three tokens and
the same set of five tokens; positions four and five are reversed. Across the
full vector, maximum absolute difference is 2.58586025, mean absolute
difference is 0.351448409, RMSE is 0.436007106, and cosine similarity is
0.98118915326. The residual difference reflects different legal low-bit
dot-product and floating-point execution paths, not different weights.

For a separate one-token full-model check, native and portable C builds return
the same top-five order. Their maximum absolute difference is 0.284544945,
mean absolute difference is 0.0461115617, RMSE is 0.0582796411, and cosine
similarity is 0.999835987704.

## Transactional MTP check

The integration probe maps the 64-layer IQ1_M target and separate MTP sidecar,
then generates the same sequence with ordinary target decode and with draft,
verification, rejection rollback, confirmed-prefix replay, and MTP
realignment. All 12 IDs match:

```text
116348 111764 101935 100049 114709 101671
131021 3709 143764 96336 332 101671
```

The test covers accepted and rejected drafts. Rejected IDs never enter visible
output or retained target state. The feature remains opt-in because ordinary
decode was faster on the fixed workload.
