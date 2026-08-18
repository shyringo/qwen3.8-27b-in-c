# Inference correctness

Correctness comparisons must use the same checkpoint and quantization. A Q4_K_M
runtime is not expected to reproduce logits from the original BF16 or FP8
weights.

## Pinned model file

```text
file:     Qwen3.8-27B-Q4_K_M.gguf
size:     17,106,775,008 bytes
revision: d7e4524557ccb88b39f5ab3a925f62f7fec49502
SHA-256:  7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
```

Set `QWEN38_VERIFY=1` when running `scripts/get-model.sh` to verify the full
file. Normal startup checks the exact file size and validates the GGUF model
contract without adding a complete 17 GB hash pass to every launch.

## Unit and build checks

```bash
make strict
make portable
```

The tests cover GGUF bounds and metadata, BPE control/user-defined tokens,
Unicode NFC composition and canonical ordering, Q4_K/Q5_K/Q6_K/Q8_0 decoding,
Q8_K activation quantization, bit-exact batched matrix output, and
greedy/top-k/top-p sampling. `strict` enables native
CPU/OpenMP code and treats warnings as errors; `portable` builds without
architecture-specific SIMD or OpenMP. The same tests also run under
AddressSanitizer and UndefinedBehaviorSanitizer during release validation.

The C NFC implementation passes all 18,722 cases in the Unicode 9.0 official
normalization suite. Independent comparisons against Hugging Face `tokenizers`
0.22 match across the complete Unicode scalar range in 8,688 chunks and a
separate 200-case set covering multilingual text, emoji, combining marks and
chat-template tokens.

## Full-checkpoint oracle

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

For the pinned Unsloth Q4_K_M file, the native prefill top five are:

```text
109455:25.2019787
2005:24.6387787
116348:18.9979134
104362:18.4371262
97785:17.5203133
```

The complete 248,320-float output has SHA-256:

```text
5dbc1f1da59beb67bd963419f4ab2a06afaff33adb25b9fb37d054edcac8863d
```

An independent no-repack, float-KV evaluation of the same GGUF produces the
same five token IDs in the same order. The reference evaluator is
`llama-completion` built from llama.cpp commit
`7acdbb1f191d869bad8c5da9d4a2121defa340af`; it is an oracle only and is not a
runtime dependency of this project. Across all 248,320 logits, the observed
maximum absolute difference is 1.85296762 (token 271), mean absolute difference
is 0.145507682, RMSE is 0.182951039, and cosine similarity is
0.995841032507. The difference reflects different legal accumulation grouping
and execution order; it does not compare against a different precision
checkpoint.

The layer-major native prefill and native token-at-a-time evaluation produce
byte-identical full logits for this prompt. Greedy decoding begins with token
109455. The first 12 native greedy token IDs are:

```text
109455 96762 98698 97447 98865 98625
51846 116348 111764 828 103764 96125
```

The independent comparison uses greedy sampling, a 64-token context and the
reference engine's `--no-repack` option. The prompt bytes are kept in
`tests/fixtures/qwen38_oracle_prompt.txt` so the chat-template boundary is part
of the test rather than an unstated assumption.

## Transactional MTP check

The integration probe generates the same 12 greedy tokens twice: first with
ordinary target-model decode, then with MTP draft blocks, target verification,
rejection rollback and MTP realignment. The IDs must match at every position:

```text
109455 96762 98698 97447 98865 98625
51846 116348 111764 828 103764 96125
```

The test sequence covers an immediate rejection, a completely accepted block
and a later rollback. Rejected tokens never enter the visible output or retained
target state.
