#include "qwen38_gguf.h"

#include <stdio.h>
#include <string.h>

static const char *type_name(uint32_t type)
{
    switch (type) {
    case Q38_GGML_F32: return "F32";
    case Q38_GGML_F16: return "F16";
    case Q38_GGML_Q4_0: return "Q4_0";
    case Q38_GGML_Q4_1: return "Q4_1";
    case Q38_GGML_Q5_0: return "Q5_0";
    case Q38_GGML_Q5_1: return "Q5_1";
    case Q38_GGML_Q8_0: return "Q8_0";
    case Q38_GGML_Q2_K: return "Q2_K";
    case Q38_GGML_Q3_K: return "Q3_K";
    case Q38_GGML_Q4_K: return "Q4_K";
    case Q38_GGML_Q5_K: return "Q5_K";
    case Q38_GGML_Q6_K: return "Q6_K";
    case Q38_GGML_IQ2_XXS: return "IQ2_XXS";
    case Q38_GGML_IQ2_XS: return "IQ2_XS";
    case Q38_GGML_IQ3_XXS: return "IQ3_XXS";
    case Q38_GGML_IQ1_S: return "IQ1_S";
    case Q38_GGML_IQ4_NL: return "IQ4_NL";
    case Q38_GGML_IQ3_S: return "IQ3_S";
    case Q38_GGML_IQ2_S: return "IQ2_S";
    case Q38_GGML_IQ4_XS: return "IQ4_XS";
    case Q38_GGML_IQ1_M: return "IQ1_M";
    case Q38_GGML_BF16: return "BF16";
    default: return "UNKNOWN";
    }
}

static int string_is(Q38GGUFString string, const char *expected)
{
    const size_t length = strlen(expected);
    return string.length == (uint64_t)length &&
           memcmp(string.data, expected, length) == 0;
}

static int require_u32(const Q38GGUF *gguf, const char *key, uint32_t expected)
{
    uint32_t actual = 0;
    if (q38_gguf_meta_u32(gguf, key, &actual) && actual == expected) return 1;
    fprintf(stderr, "checkpoint contract failed: %s must be %u (got %u)\n",
            key, expected, actual);
    return 0;
}

static int require_shape(const Q38GGUF *gguf, const char *name,
                         uint64_t d0, uint64_t d1)
{
    const Q38GGUFTensor *tensor = q38_gguf_find_tensor(gguf, name);
    const uint32_t n_dims = d1 ? 2u : 1u;
    if (tensor && tensor->n_dims == n_dims && tensor->shape[0] == d0 &&
        (!d1 || tensor->shape[1] == d1)) return 1;
    fprintf(stderr, "checkpoint contract failed: %s expected [%llu",
            name, (unsigned long long)d0);
    if (d1) fprintf(stderr, ", %llu", (unsigned long long)d1);
    fprintf(stderr, "]\n");
    if (tensor) {
        fprintf(stderr, "  got %s [", type_name(tensor->type));
        for (uint32_t i = 0; i < tensor->n_dims; ++i) {
            fprintf(stderr, "%s%llu", i ? ", " : "",
                    (unsigned long long)tensor->shape[i]);
        }
        fprintf(stderr, "]\n");
    } else {
        fprintf(stderr, "  tensor is missing\n");
    }
    return 0;
}

static int require_layer(const Q38GGUF *gguf, int layer)
{
    char name[128];
    int ok = 1;
#define REQUIRE(suffix, d0, d1) do { \
    snprintf(name, sizeof(name), "blk.%d.%s", layer, suffix); \
    ok &= require_shape(gguf, name, d0, d1); \
} while (0)

    REQUIRE("attn_norm.weight", 5120, 0);
    REQUIRE("post_attention_norm.weight", 5120, 0);
    REQUIRE("ffn_gate.weight", 5120, 17408);
    REQUIRE("ffn_up.weight", 5120, 17408);
    REQUIRE("ffn_down.weight", 17408, 5120);
    if ((layer + 1) % 4 == 0) {
        REQUIRE("attn_q.weight", 5120, 12288);
        REQUIRE("attn_k.weight", 5120, 1024);
        REQUIRE("attn_v.weight", 5120, 1024);
        REQUIRE("attn_output.weight", 6144, 5120);
        REQUIRE("attn_q_norm.weight", 256, 0);
        REQUIRE("attn_k_norm.weight", 256, 0);
    } else {
        REQUIRE("attn_qkv.weight", 5120, 10240);
        REQUIRE("attn_gate.weight", 5120, 6144);
        REQUIRE("ssm_alpha.weight", 5120, 48);
        REQUIRE("ssm_beta.weight", 5120, 48);
        REQUIRE("ssm_conv1d.weight", 4, 10240);
        REQUIRE("ssm_dt.bias", 48, 0);
        REQUIRE("ssm_a", 48, 0);
        REQUIRE("ssm_norm.weight", 128, 0);
        REQUIRE("ssm_out.weight", 6144, 5120);
    }
#undef REQUIRE
    return ok;
}

static int require_mtp_layer(const Q38GGUF *gguf)
{
    int ok = 1;
    ok &= require_shape(gguf, "blk.64.attn_norm.weight", 5120, 0);
    ok &= require_shape(gguf, "blk.64.post_attention_norm.weight", 5120, 0);
    ok &= require_shape(gguf, "blk.64.ffn_gate.weight", 5120, 17408);
    ok &= require_shape(gguf, "blk.64.ffn_up.weight", 5120, 17408);
    ok &= require_shape(gguf, "blk.64.ffn_down.weight", 17408, 5120);
    ok &= require_shape(gguf, "blk.64.attn_q.weight", 5120, 12288);
    ok &= require_shape(gguf, "blk.64.attn_k.weight", 5120, 1024);
    ok &= require_shape(gguf, "blk.64.attn_v.weight", 5120, 1024);
    ok &= require_shape(gguf, "blk.64.attn_output.weight", 6144, 5120);
    ok &= require_shape(gguf, "blk.64.attn_q_norm.weight", 256, 0);
    ok &= require_shape(gguf, "blk.64.attn_k_norm.weight", 256, 0);
    ok &= require_shape(gguf, "blk.64.nextn.eh_proj.weight", 10240, 5120);
    ok &= require_shape(gguf, "blk.64.nextn.enorm.weight", 5120, 0);
    ok &= require_shape(gguf, "blk.64.nextn.hnorm.weight", 5120, 0);
    ok &= require_shape(gguf, "blk.64.nextn.shared_head_norm.weight", 5120, 0);
    return ok;
}

static int require_tokenizer(const Q38GGUF *gguf)
{
    int ok = 1;
    const Q38GGUFMeta *tokens = q38_gguf_find_meta(gguf,
                                                   "tokenizer.ggml.tokens");
    const Q38GGUFMeta *merges = q38_gguf_find_meta(gguf,
                                                   "tokenizer.ggml.merges");
    if (!tokens || tokens->type != Q38_GGUF_META_ARRAY ||
        tokens->array_type != Q38_GGUF_META_STRING ||
        tokens->count != 248320) {
        fprintf(stderr, "checkpoint contract failed: tokenizer vocabulary\n");
        ok = 0;
    }
    if (!merges || merges->type != Q38_GGUF_META_ARRAY ||
        merges->array_type != Q38_GGUF_META_STRING || merges->count == 0) {
        fprintf(stderr, "checkpoint contract failed: tokenizer merges\n");
        ok = 0;
    }
    return ok;
}

static int inspect_contract(const Q38GGUF *gguf)
{
    Q38GGUFString architecture;
    int ok = q38_gguf_meta_string(gguf, "general.architecture", &architecture) &&
             string_is(architecture, "qwen35");
    if (!ok) fprintf(stderr, "checkpoint contract failed: architecture is not qwen35\n");
    uint32_t block_count = 0;
    uint32_t nextn_layers = 0;
    const int has_base_layer =
        q38_gguf_find_tensor(gguf, "blk.0.attn_norm.weight") != NULL;
    (void)q38_gguf_meta_u32(gguf, "qwen35.block_count", &block_count);
    (void)q38_gguf_meta_u32(gguf, "qwen35.nextn_predict_layers",
                            &nextn_layers);

    if (block_count == 64 && nextn_layers == 0 && has_base_layer) {
        ok &= require_u32(gguf, "qwen35.embedding_length", 5120);
        ok &= require_u32(gguf, "qwen35.feed_forward_length", 17408);
        ok &= require_u32(gguf, "qwen35.attention.head_count", 24);
        ok &= require_u32(gguf, "qwen35.attention.head_count_kv", 4);
        ok &= require_u32(gguf, "qwen35.full_attention_interval", 4);
        ok &= require_u32(gguf, "qwen35.ssm.state_size", 128);
        ok &= require_u32(gguf, "qwen35.ssm.group_count", 16);
        ok &= require_u32(gguf, "qwen35.ssm.time_step_rank", 48);
        ok &= require_u32(gguf, "qwen35.ssm.inner_size", 6144);
        ok &= require_shape(gguf, "token_embd.weight", 5120, 248320);
        ok &= require_shape(gguf, "output_norm.weight", 5120, 0);
        ok &= require_shape(gguf, "output.weight", 5120, 248320);
        for (int layer = 0; layer < 64; ++layer)
            ok &= require_layer(gguf, layer);
        ok &= require_tokenizer(gguf);
        return ok;
    }

    if (block_count == 65 && nextn_layers == 1 && !has_base_layer) {
        ok &= require_shape(gguf, "token_embd.weight", 5120, 248320);
        ok &= require_shape(gguf, "output_norm.weight", 5120, 0);
        ok &= require_shape(gguf, "output.weight", 5120, 248320);
        ok &= require_mtp_layer(gguf);
        return ok;
    }

    ok &= require_u32(gguf, "qwen35.block_count", 65);
    ok &= require_u32(gguf, "qwen35.nextn_predict_layers", 1);
    ok &= require_u32(gguf, "qwen35.embedding_length", 5120);
    ok &= require_u32(gguf, "qwen35.feed_forward_length", 17408);
    ok &= require_u32(gguf, "qwen35.attention.head_count", 24);
    ok &= require_u32(gguf, "qwen35.attention.head_count_kv", 4);
    ok &= require_u32(gguf, "qwen35.full_attention_interval", 4);
    ok &= require_u32(gguf, "qwen35.ssm.state_size", 128);
    ok &= require_u32(gguf, "qwen35.ssm.group_count", 16);
    ok &= require_u32(gguf, "qwen35.ssm.time_step_rank", 48);
    ok &= require_u32(gguf, "qwen35.ssm.inner_size", 6144);
    ok &= require_shape(gguf, "token_embd.weight", 5120, 248320);
    ok &= require_shape(gguf, "output_norm.weight", 5120, 0);
    ok &= require_shape(gguf, "output.weight", 5120, 248320);
    for (int layer = 0; layer < 64; ++layer) ok &= require_layer(gguf, layer);
    ok &= require_mtp_layer(gguf);

    const Q38GGUFMeta *tokens = q38_gguf_find_meta(gguf, "tokenizer.ggml.tokens");
    const Q38GGUFMeta *merges = q38_gguf_find_meta(gguf, "tokenizer.ggml.merges");
    if (!tokens || tokens->type != Q38_GGUF_META_ARRAY ||
        tokens->array_type != Q38_GGUF_META_STRING || tokens->count != 248320) {
        fprintf(stderr, "checkpoint contract failed: tokenizer vocabulary\n");
        ok = 0;
    }
    if (!merges || merges->type != Q38_GGUF_META_ARRAY ||
        merges->array_type != Q38_GGUF_META_STRING || merges->count == 0) {
        fprintf(stderr, "checkpoint contract failed: tokenizer merges\n");
        ok = 0;
    }
    return ok;
}

int main(int argc, char **argv)
{
    if (argc != 2 && argc != 3) {
        fprintf(stderr, "usage: %s MODEL.gguf [METADATA_KEY]\n", argv[0]);
        return 2;
    }
    Q38GGUF gguf;
    if (!q38_gguf_open(&gguf, argv[1])) return 1;

    if (argc == 3) {
        const Q38GGUFMeta *meta = q38_gguf_find_meta(&gguf, argv[2]);
        if (!meta) {
            fprintf(stderr, "metadata key not found: %s\n", argv[2]);
            q38_gguf_close(&gguf);
            return 1;
        }
        if (meta->type == Q38_GGUF_META_STRING) {
            fwrite(meta->data, 1, (size_t)meta->data_size, stdout);
            putchar('\n');
        } else if (meta->type == Q38_GGUF_META_UINT32 && meta->data_size == 4) {
            uint32_t value = 0;
            (void)q38_gguf_meta_u32(&gguf, argv[2], &value);
            printf("%u\n", value);
        } else {
            fprintf(stderr, "metadata key has unsupported display type %u\n", meta->type);
            q38_gguf_close(&gguf);
            return 1;
        }
        q38_gguf_close(&gguf);
        return 0;
    }

    uint64_t bytes_by_type[Q38_GGML_BF16 + 1] = {0};
    uint32_t count_by_type[Q38_GGML_BF16 + 1] = {0};
    for (uint64_t i = 0; i < gguf.tensor_count; ++i) {
        const Q38GGUFTensor *tensor = &gguf.tensors[i];
        if (tensor->type <= Q38_GGML_BF16) {
            bytes_by_type[tensor->type] += tensor->nbytes;
            ++count_by_type[tensor->type];
        }
    }
    printf("GGUF v%u: %llu tensors, %llu metadata entries, data at %llu\n",
           gguf.version, (unsigned long long)gguf.tensor_count,
           (unsigned long long)gguf.metadata_count,
           (unsigned long long)gguf.data_offset);
    for (uint32_t type = 0; type <= Q38_GGML_BF16; ++type) {
        if (count_by_type[type]) {
            printf("  %-7s %4u tensors  %7.3f GiB\n", type_name(type),
                   count_by_type[type],
                   (double)bytes_by_type[type] / (1024.0 * 1024.0 * 1024.0));
        }
    }

    const int ok = inspect_contract(&gguf);
    q38_gguf_close(&gguf);
    if (!ok) return 1;
    puts("Qwen3.8-27B GGUF contract: OK");
    return 0;
}
