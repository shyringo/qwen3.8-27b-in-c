#define _POSIX_C_SOURCE 200809L

#include "qwen38_model.h"

#include "qwen38_gguf.h"
#include "qwen38_quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define Q38_HIDDEN 5120
#define Q38_FFN 17408
#define Q38_LAYERS 64
#define Q38_TOTAL_LAYERS 65
#define Q38_RECURRENT_LAYERS 48
#define Q38_FULL_LAYERS 16
#define Q38_TOTAL_FULL_LAYERS 17
#define Q38_LINEAR_QK_HEADS 16
#define Q38_LINEAR_V_HEADS 48
#define Q38_LINEAR_HEAD_DIM 128
#define Q38_LINEAR_QK_DIM 2048
#define Q38_LINEAR_V_DIM 6144
#define Q38_LINEAR_QKV_DIM 10240
#define Q38_ATTN_HEADS 24
#define Q38_ATTN_KV_HEADS 4
#define Q38_ATTN_HEAD_DIM 256
#define Q38_ATTN_Q_DIM 6144
#define Q38_ATTN_QG_DIM 12288
#define Q38_ATTN_KV_DIM 1024
#define Q38_VOCAB 248320
#define Q38_RMS_EPS 1e-6f

typedef struct {
    const Q38GGUFTensor *attn_norm;
    const Q38GGUFTensor *post_norm;
    const Q38GGUFTensor *ffn_gate;
    const Q38GGUFTensor *ffn_up;
    const Q38GGUFTensor *ffn_down;
    union {
        struct {
            const Q38GGUFTensor *qkv;
            const Q38GGUFTensor *z;
            const Q38GGUFTensor *alpha;
            const Q38GGUFTensor *beta;
            const Q38GGUFTensor *conv;
            const Q38GGUFTensor *dt;
            const Q38GGUFTensor *a;
            const Q38GGUFTensor *norm;
            const Q38GGUFTensor *out;
        } linear;
        struct {
            const Q38GGUFTensor *q;
            const Q38GGUFTensor *k;
            const Q38GGUFTensor *v;
            const Q38GGUFTensor *out;
            const Q38GGUFTensor *q_norm;
            const Q38GGUFTensor *k_norm;
        } full;
    } attention;
} Q38Layer;

typedef struct {
    const Q38GGUFTensor *eh_proj;
    const Q38GGUFTensor *enorm;
    const Q38GGUFTensor *hnorm;
    const Q38GGUFTensor *shared_head_norm;
} Q38MTPWeights;

typedef struct {
    float *hidden;
    float *norm;
    float *branch;
    float *wide0;
    float *wide1;
    float *q;
    float *gate;
    float *k;
    float *v;
    float *attn;
    float *scores;
    float *beta;
    float *alpha;
    float *logits;
    Q38Q8KBlock *quantized;
} Q38Scratch;

typedef struct {
    float *hidden;
    float *norm;
    float *branch;
    float *wide0;
    float *wide1;
    float *key;
    float *value;
    float *attention;
    float *beta;
    float *alpha;
    float *target_norm;
    Q38Q8KBlock *quantized;
} Q38BatchScratch;

struct Q38Model {
    Q38GGUF gguf;
    Q38GGUF mtp_gguf;
    const Q38GGUFTensor *embedding;
    const Q38GGUFTensor *output_norm;
    const Q38GGUFTensor *output;
    const Q38GGUFTensor *mtp_embedding;
    const Q38GGUFTensor *mtp_output;
    Q38Layer layers[Q38_TOTAL_LAYERS];
    Q38MTPWeights mtp;
    uint32_t context_length;
    uint32_t position;
    uint32_t mtp_position;
    int mtp_weights_loaded;
    int mtp_enabled;
    float *mtp_input_hidden;
    float *checkpoint_conv_state;
    float *checkpoint_delta_state;
    float *checkpoint_mtp_hidden;
    uint32_t checkpoint_position;
    uint32_t checkpoint_mtp_position;
    float *conv_state;
    float *delta_state;
    float *key_cache;
    float *value_cache;
    Q38Scratch scratch;
};

static const Q38GGUFTensor *q38_weight_from(const Q38GGUF *gguf,
                                            const char *name)
{
    const Q38GGUFTensor *tensor = q38_gguf_find_tensor(gguf, name);
    if (!tensor) fprintf(stderr, "qwen38: missing tensor %s\n", name);
    return tensor;
}

static const Q38GGUFTensor *q38_layer_weight_from(const Q38GGUF *gguf,
                                                  int layer,
                                                  const char *suffix)
{
    char name[128];
    snprintf(name, sizeof(name), "blk.%d.%s", layer, suffix);
    return q38_weight_from(gguf, name);
}

static int q38_meta_string_is(const Q38GGUF *gguf, const char *key,
                              const char *expected)
{
    Q38GGUFString value;
    const size_t length = strlen(expected);
    return q38_gguf_meta_string(gguf, key, &value) &&
           value.length == (uint64_t)length &&
           memcmp(value.data, expected, length) == 0;
}

static int q38_meta_u32_is(const Q38GGUF *gguf, const char *key,
                           uint32_t expected)
{
    uint32_t value = 0;
    return q38_gguf_meta_u32(gguf, key, &value) && value == expected;
}

static int q38_matrix_type_q8_k(uint32_t type)
{
    return type == Q38_GGML_Q2_K || type == Q38_GGML_Q3_K ||
           type == Q38_GGML_Q4_K || type == Q38_GGML_Q5_K ||
           type == Q38_GGML_Q6_K || type == Q38_GGML_IQ2_XXS ||
           type == Q38_GGML_IQ2_XS || type == Q38_GGML_IQ3_XXS ||
           type == Q38_GGML_IQ4_NL || type == Q38_GGML_IQ3_S ||
           type == Q38_GGML_IQ2_S ||
           type == Q38_GGML_IQ4_XS || type == Q38_GGML_IQ1_S ||
           type == Q38_GGML_IQ1_M;
}

static int q38_matrix_type_supported(uint32_t type)
{
    return type == Q38_GGML_F32 || type == Q38_GGML_F16 ||
           type == Q38_GGML_BF16 || type == Q38_GGML_Q8_0 ||
           q38_matrix_type_q8_k(type);
}

static int q38_expect_tensor(const Q38GGUFTensor *tensor, const char *name,
                             uint64_t width, uint64_t rows, int f32_only)
{
    const uint32_t dimensions = rows ? 2u : 1u;
    const int shape_ok = tensor && tensor->n_dims == dimensions &&
                         tensor->shape[0] == width &&
                         (!rows || tensor->shape[1] == rows);
    const int type_ok = tensor && (f32_only
        ? tensor->type == Q38_GGML_F32
        : q38_matrix_type_supported(tensor->type));
    if (shape_ok && type_ok) return 1;
    fprintf(stderr, "qwen38: incompatible tensor %s; expected %s[%llu",
            name, f32_only ? "F32 " : "", (unsigned long long)width);
    if (rows) fprintf(stderr, ",%llu", (unsigned long long)rows);
    fprintf(stderr, "]\n");
    return 0;
}

static int q38_validate_layer(const Q38Layer *weights, int layer)
{
    char name[128];
    int ok = 1;
#define EXPECT(field, suffix, width, rows, f32_only) do { \
    snprintf(name, sizeof(name), "blk.%d.%s", layer, suffix); \
    ok &= q38_expect_tensor((field), name, width, rows, f32_only); \
} while (0)
    EXPECT(weights->attn_norm, "attn_norm.weight", Q38_HIDDEN, 0, 1);
    EXPECT(weights->post_norm, "post_attention_norm.weight", Q38_HIDDEN, 0, 1);
    EXPECT(weights->ffn_gate, "ffn_gate.weight", Q38_HIDDEN, Q38_FFN, 0);
    EXPECT(weights->ffn_up, "ffn_up.weight", Q38_HIDDEN, Q38_FFN, 0);
    EXPECT(weights->ffn_down, "ffn_down.weight", Q38_FFN, Q38_HIDDEN, 0);
    if ((layer + 1) % 4 == 0 || layer == Q38_LAYERS) {
        EXPECT(weights->attention.full.q, "attn_q.weight", Q38_HIDDEN,
               Q38_ATTN_QG_DIM, 0);
        EXPECT(weights->attention.full.k, "attn_k.weight", Q38_HIDDEN,
               Q38_ATTN_KV_DIM, 0);
        EXPECT(weights->attention.full.v, "attn_v.weight", Q38_HIDDEN,
               Q38_ATTN_KV_DIM, 0);
        EXPECT(weights->attention.full.out, "attn_output.weight", Q38_ATTN_Q_DIM,
               Q38_HIDDEN, 0);
        EXPECT(weights->attention.full.q_norm, "attn_q_norm.weight",
               Q38_ATTN_HEAD_DIM, 0, 1);
        EXPECT(weights->attention.full.k_norm, "attn_k_norm.weight",
               Q38_ATTN_HEAD_DIM, 0, 1);
    } else {
        EXPECT(weights->attention.linear.qkv, "attn_qkv.weight", Q38_HIDDEN,
               Q38_LINEAR_QKV_DIM, 0);
        EXPECT(weights->attention.linear.z, "attn_gate.weight", Q38_HIDDEN,
               Q38_LINEAR_V_DIM, 0);
        EXPECT(weights->attention.linear.alpha, "ssm_alpha.weight", Q38_HIDDEN,
               Q38_LINEAR_V_HEADS, 0);
        EXPECT(weights->attention.linear.beta, "ssm_beta.weight", Q38_HIDDEN,
               Q38_LINEAR_V_HEADS, 0);
        EXPECT(weights->attention.linear.conv, "ssm_conv1d.weight", 4,
               Q38_LINEAR_QKV_DIM, 1);
        EXPECT(weights->attention.linear.dt, "ssm_dt.bias",
               Q38_LINEAR_V_HEADS, 0, 1);
        EXPECT(weights->attention.linear.a, "ssm_a", Q38_LINEAR_V_HEADS, 0, 1);
        EXPECT(weights->attention.linear.norm, "ssm_norm.weight",
               Q38_LINEAR_HEAD_DIM, 0, 1);
        EXPECT(weights->attention.linear.out, "ssm_out.weight", Q38_LINEAR_V_DIM,
               Q38_HIDDEN, 0);
    }
#undef EXPECT
    return ok;
}

static int q38_validate_base_contract(const Q38Model *model)
{
    const Q38GGUF *gguf = &model->gguf;
    uint32_t block_count = 0;
    uint32_t nextn_layers = 0;
    int ok = q38_gguf_meta_u32(gguf, "qwen35.block_count", &block_count) &&
             q38_gguf_meta_u32(gguf, "qwen35.nextn_predict_layers",
                               &nextn_layers) &&
             ((block_count == Q38_LAYERS && nextn_layers == 0) ||
              (block_count == Q38_TOTAL_LAYERS && nextn_layers == 1)) &&
             q38_meta_string_is(gguf, "general.architecture", "qwen35") &&
             q38_meta_u32_is(gguf, "qwen35.embedding_length", Q38_HIDDEN) &&
             q38_meta_u32_is(gguf, "qwen35.feed_forward_length", Q38_FFN) &&
             q38_meta_u32_is(gguf, "qwen35.attention.head_count", Q38_ATTN_HEADS) &&
             q38_meta_u32_is(gguf, "qwen35.attention.head_count_kv", Q38_ATTN_KV_HEADS) &&
             q38_meta_u32_is(gguf, "qwen35.full_attention_interval", 4) &&
             q38_meta_u32_is(gguf, "qwen35.ssm.state_size", Q38_LINEAR_HEAD_DIM) &&
             q38_meta_u32_is(gguf, "qwen35.ssm.group_count", Q38_LINEAR_QK_HEADS) &&
             q38_meta_u32_is(gguf, "qwen35.ssm.time_step_rank", Q38_LINEAR_V_HEADS) &&
             q38_meta_u32_is(gguf, "qwen35.ssm.inner_size", Q38_LINEAR_V_DIM);
    if (!ok) {
        fprintf(stderr, "qwen38: checkpoint metadata is not Qwen3.8-27B\n");
        return 0;
    }
    ok &= q38_expect_tensor(model->embedding, "token_embd.weight", Q38_HIDDEN,
                            Q38_VOCAB, 0);
    ok &= q38_expect_tensor(model->output_norm, "output_norm.weight", Q38_HIDDEN,
                            0, 1);
    ok &= q38_expect_tensor(model->output, "output.weight", Q38_HIDDEN,
                            Q38_VOCAB, 0);
    for (int layer = 0; layer < Q38_LAYERS; ++layer)
        ok &= q38_validate_layer(&model->layers[layer], layer);
    return ok;
}

static int q38_validate_mtp_contract(const Q38Model *model,
                                     const Q38GGUF *gguf)
{
    int ok = q38_meta_string_is(gguf, "general.architecture", "qwen35") &&
             q38_meta_u32_is(gguf, "qwen35.block_count", Q38_TOTAL_LAYERS) &&
             q38_meta_u32_is(gguf, "qwen35.nextn_predict_layers", 1) &&
             q38_meta_u32_is(gguf, "qwen35.embedding_length", Q38_HIDDEN) &&
             q38_meta_u32_is(gguf, "qwen35.feed_forward_length", Q38_FFN);
    if (!ok) {
        fprintf(stderr, "qwen38: MTP checkpoint metadata is not Qwen3.8-27B\n");
        return 0;
    }
    ok &= q38_expect_tensor(model->mtp_embedding, "token_embd.weight",
                            Q38_HIDDEN, Q38_VOCAB, 0);
    ok &= q38_expect_tensor(model->mtp_output, "output.weight", Q38_HIDDEN,
                            Q38_VOCAB, 0);
    ok &= q38_validate_layer(&model->layers[Q38_LAYERS], Q38_LAYERS);
    ok &= q38_expect_tensor(model->mtp.eh_proj, "blk.64.nextn.eh_proj.weight",
                            2u * Q38_HIDDEN, Q38_HIDDEN, 0);
    ok &= q38_expect_tensor(model->mtp.enorm, "blk.64.nextn.enorm.weight",
                            Q38_HIDDEN, 0, 1);
    ok &= q38_expect_tensor(model->mtp.hnorm, "blk.64.nextn.hnorm.weight",
                            Q38_HIDDEN, 0, 1);
    ok &= q38_expect_tensor(model->mtp.shared_head_norm,
                            "blk.64.nextn.shared_head_norm.weight",
                            Q38_HIDDEN, 0, 1);
    return ok;
}

static int q38_bind_layer(Q38Layer *weights, const Q38GGUF *gguf, int layer)
{
    memset(weights, 0, sizeof(*weights));
    weights->attn_norm = q38_layer_weight_from(gguf, layer, "attn_norm.weight");
    weights->post_norm = q38_layer_weight_from(
        gguf, layer, "post_attention_norm.weight");
    weights->ffn_gate = q38_layer_weight_from(gguf, layer, "ffn_gate.weight");
    weights->ffn_up = q38_layer_weight_from(gguf, layer, "ffn_up.weight");
    weights->ffn_down = q38_layer_weight_from(gguf, layer, "ffn_down.weight");
    if ((layer + 1) % 4 == 0 || layer == Q38_LAYERS) {
        weights->attention.full.q = q38_layer_weight_from(
            gguf, layer, "attn_q.weight");
        weights->attention.full.k = q38_layer_weight_from(
            gguf, layer, "attn_k.weight");
        weights->attention.full.v = q38_layer_weight_from(
            gguf, layer, "attn_v.weight");
        weights->attention.full.out = q38_layer_weight_from(
            gguf, layer, "attn_output.weight");
        weights->attention.full.q_norm = q38_layer_weight_from(
            gguf, layer, "attn_q_norm.weight");
        weights->attention.full.k_norm = q38_layer_weight_from(
            gguf, layer, "attn_k_norm.weight");
        return weights->attn_norm && weights->post_norm && weights->ffn_gate &&
               weights->ffn_up && weights->ffn_down &&
               weights->attention.full.q && weights->attention.full.k &&
               weights->attention.full.v && weights->attention.full.out &&
               weights->attention.full.q_norm && weights->attention.full.k_norm;
    }
    weights->attention.linear.qkv = q38_layer_weight_from(
        gguf, layer, "attn_qkv.weight");
    weights->attention.linear.z = q38_layer_weight_from(
        gguf, layer, "attn_gate.weight");
    weights->attention.linear.alpha = q38_layer_weight_from(
        gguf, layer, "ssm_alpha.weight");
    weights->attention.linear.beta = q38_layer_weight_from(
        gguf, layer, "ssm_beta.weight");
    weights->attention.linear.conv = q38_layer_weight_from(
        gguf, layer, "ssm_conv1d.weight");
    weights->attention.linear.dt = q38_layer_weight_from(
        gguf, layer, "ssm_dt.bias");
    weights->attention.linear.a = q38_layer_weight_from(gguf, layer, "ssm_a");
    weights->attention.linear.norm = q38_layer_weight_from(
        gguf, layer, "ssm_norm.weight");
    weights->attention.linear.out = q38_layer_weight_from(
        gguf, layer, "ssm_out.weight");
    return weights->attn_norm && weights->post_norm && weights->ffn_gate &&
           weights->ffn_up && weights->ffn_down &&
           weights->attention.linear.qkv && weights->attention.linear.z &&
           weights->attention.linear.alpha && weights->attention.linear.beta &&
           weights->attention.linear.conv && weights->attention.linear.dt &&
           weights->attention.linear.a && weights->attention.linear.norm &&
           weights->attention.linear.out;
}

static int q38_bind_mtp_weights(Q38Model *model, const Q38GGUF *gguf)
{
    model->mtp_embedding = q38_weight_from(gguf, "token_embd.weight");
    model->mtp_output = q38_weight_from(gguf, "output.weight");
    if (!q38_bind_layer(&model->layers[Q38_LAYERS], gguf, Q38_LAYERS))
        return 0;
    model->mtp.eh_proj = q38_layer_weight_from(
        gguf, Q38_LAYERS, "nextn.eh_proj.weight");
    model->mtp.enorm = q38_layer_weight_from(
        gguf, Q38_LAYERS, "nextn.enorm.weight");
    model->mtp.hnorm = q38_layer_weight_from(
        gguf, Q38_LAYERS, "nextn.hnorm.weight");
    model->mtp.shared_head_norm = q38_layer_weight_from(
        gguf, Q38_LAYERS, "nextn.shared_head_norm.weight");
    if (!model->mtp_embedding || !model->mtp_output || !model->mtp.eh_proj ||
        !model->mtp.enorm || !model->mtp.hnorm ||
        !model->mtp.shared_head_norm || !q38_validate_mtp_contract(model, gguf))
        return 0;
    model->mtp_weights_loaded = 1;
    return 1;
}

static int q38_bind_weights(Q38Model *model)
{
    model->embedding = q38_weight_from(&model->gguf, "token_embd.weight");
    model->output_norm = q38_weight_from(&model->gguf, "output_norm.weight");
    model->output = q38_weight_from(&model->gguf, "output.weight");
    if (!model->embedding || !model->output_norm || !model->output) return 0;
    for (int layer = 0; layer < Q38_LAYERS; ++layer)
        if (!q38_bind_layer(&model->layers[layer], &model->gguf, layer)) return 0;
    if (!q38_validate_base_contract(model)) return 0;

    uint32_t block_count = 0;
    uint32_t nextn_layers = 0;
    if (!q38_gguf_meta_u32(&model->gguf, "qwen35.block_count", &block_count) ||
        !q38_gguf_meta_u32(&model->gguf, "qwen35.nextn_predict_layers",
                            &nextn_layers)) return 0;
    if (block_count == Q38_TOTAL_LAYERS && nextn_layers == 1)
        return q38_bind_mtp_weights(model, &model->gguf);
    return 1;
}

static float q38_scalar(const Q38GGUFTensor *tensor, uint64_t index)
{
    if (!tensor || tensor->type != Q38_GGML_F32 || index >= tensor->shape[0]) return NAN;
    float value;
    memcpy(&value, tensor->data + index * 4, sizeof(value));
    return value;
}

static void q38_rmsnorm(float *output, const float *input,
                        const Q38GGUFTensor *weight, int length)
{
    double sum = 0.0;
    for (int i = 0; i < length; ++i)
        sum += (double)(input[i] * input[i]);
    const float mean = (float)(sum / (double)length);
    const float scale = 1.0f / sqrtf(mean + Q38_RMS_EPS);
    for (int i = 0; i < length; ++i) output[i] = input[i] * scale * q38_scalar(weight, (uint64_t)i);
}

static float q38_sigmoid(float value)
{
    return 1.0f / (1.0f + expf(-value));
}

static float q38_silu(float value)
{
    return value / (1.0f + expf(-value));
}

#if defined(__AVX2__) && defined(__FMA__) && defined(Q38_FAST_EXP)
static __m256 q38_exp_f32x8(__m256 values)
{
    const __m256 bias = _mm256_set1_ps(0x1.8p23f);
    const __m256 rounded = _mm256_fmadd_ps(
        values, _mm256_set1_ps(0x1.715476p+0f), bias);
    const __m256 exponent = _mm256_sub_ps(rounded, bias);
    const __m256 remainder = _mm256_fnmadd_ps(
        exponent, _mm256_set1_ps(0x1.7f7d1cp-20f),
        _mm256_fnmadd_ps(exponent, _mm256_set1_ps(0x1.62e4p-1f), values));
    const __m256i exponent_bits = _mm256_slli_epi32(
        _mm256_castps_si256(rounded), 23);
    const __m256 power = _mm256_castsi256_ps(_mm256_add_epi32(
        exponent_bits, _mm256_castps_si256(_mm256_set1_ps(1.0f))));
    const __m256i exceptional = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0f), exponent),
        _mm256_set1_ps(126.0f), _CMP_GT_OQ));
    const __m256 squared = _mm256_mul_ps(remainder, remainder);
    const __m256 polynomial = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), remainder,
                            _mm256_set1_ps(0x1.573e2ep-5f)),
            squared,
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), remainder,
                            _mm256_set1_ps(0x1.fffdb6p-2f))),
        squared, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), remainder));
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(exceptional)))
        return _mm256_fmadd_ps(polynomial, power, power);

    const __m256i adjustment = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(
            exponent, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32((int)0x82000000u));
    const __m256 scale1 = _mm256_castsi256_ps(_mm256_add_epi32(
        adjustment, _mm256_set1_epi32(0x7f000000)));
    const __m256 scale2 = _mm256_castsi256_ps(
        _mm256_sub_epi32(exponent_bits, adjustment));
    const __m256i overflow = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0f), exponent),
        _mm256_set1_ps(192.0f), _CMP_GT_OQ));
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(overflow),
                      _mm256_mul_ps(scale1, scale1)),
        _mm256_andnot_ps(
            _mm256_castsi256_ps(overflow),
            _mm256_or_ps(
                _mm256_and_ps(_mm256_castsi256_ps(exceptional),
                    _mm256_mul_ps(
                        _mm256_fmadd_ps(scale2, polynomial, scale2), scale1)),
                _mm256_andnot_ps(_mm256_castsi256_ps(exceptional),
                    _mm256_fmadd_ps(power, polynomial, power)))));
}

static __m256 q38_silu_f32x8(__m256 values)
{
    const __m256 negative = _mm256_sub_ps(_mm256_setzero_ps(), values);
    return _mm256_div_ps(values, _mm256_add_ps(
        _mm256_set1_ps(1.0f), q38_exp_f32x8(negative)));
}
#endif

static void q38_silu_inplace(float *values, int length)
{
    int i = 0;
#if defined(__AVX2__) && defined(__FMA__) && defined(Q38_FAST_EXP)
    for (; i + 7 < length; i += 8)
        _mm256_storeu_ps(values + i,
            q38_silu_f32x8(_mm256_loadu_ps(values + i)));
#endif
    for (; i < length; ++i) values[i] = q38_silu(values[i]);
}

static void q38_silu_multiply(float *values, const float *gate, int length)
{
    int i = 0;
#if defined(__AVX2__) && defined(__FMA__) && defined(Q38_FAST_EXP)
    for (; i + 7 < length; i += 8) {
        const __m256 activated = q38_silu_f32x8(_mm256_loadu_ps(gate + i));
        _mm256_storeu_ps(values + i,
            _mm256_mul_ps(_mm256_loadu_ps(values + i), activated));
    }
#endif
    for (; i < length; ++i) values[i] *= q38_silu(gate[i]);
}

static void q38_swiglu(float *gate, const float *up, uint64_t length)
{
    uint64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__) && defined(Q38_FAST_EXP)
    const uint64_t vector_end = length & ~(uint64_t)7;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(length >= (uint64_t)Q38_FFN * 2u)
#endif
    for (uint64_t offset = 0; offset < vector_end; offset += 8) {
        const __m256 activated = q38_silu_f32x8(
            _mm256_loadu_ps(gate + offset));
        _mm256_storeu_ps(gate + offset,
            _mm256_mul_ps(activated, _mm256_loadu_ps(up + offset)));
    }
    i = vector_end;
#endif
    for (; i < length; ++i) gate[i] = q38_silu(gate[i]) * up[i];
}

static float q38_softplus(float value)
{
    if (value > 20.0f) return value;
    return logf(1.0f + expf(value));
}

static int q38_trace_layers_enabled(void)
{
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized) {
        const char *value = getenv("QWEN38_TRACE_LAYERS");
        enabled = value && value[0] && strcmp(value, "0") != 0;
        initialized = 1;
    }
    return enabled;
}

static void q38_trace_vector(const char *name, int layer,
                             const float *values, uint64_t length)
{
    if (!q38_trace_layers_enabled()) return;
    float sum = 0.0f;
    double sum_abs = 0.0;
    for (uint64_t i = 0; i < length; ++i) {
        sum += values[i];
        sum_abs += fabsf(values[i]);
    }
    fprintf(stderr,
            "trace %s%d sum=%.9g abs=%.9g first=%.9g,%.9g,%.9g "
            "last=%.9g,%.9g,%.9g\n",
            name, layer, sum, sum_abs,
            values[0], values[1], values[2],
            values[length - 3], values[length - 2], values[length - 1]);
}

static void q38_trace_hidden(const char *name, int layer,
                             const float *hidden)
{
    q38_trace_vector(name, layer, hidden, Q38_HIDDEN);
}

static int q38_quantize_input(Q38Scratch *scratch, const float *input,
                              uint64_t length)
{
    return q38_quantize_q8_k(scratch->quantized, input, length);
}

static int q38_project(float *output, const float *input,
                       const Q38Q8KBlock *quantized, uint64_t length,
                       const Q38GGUFTensor *weight)
{
    if (q38_matrix_type_q8_k(weight->type))
        return q38_tensor_gemv_q8_k(output, quantized, length, weight);
    return q38_tensor_gemv_f32(output, input, weight);
}

static void q38_l2norm(float *values, int length)
{
    double sum = 0.0;
    for (int i = 0; i < length; ++i) sum += (double)(values[i] * values[i]);
    const float scale = 1.0f / fmaxf(sqrtf((float)sum), Q38_RMS_EPS);
    for (int i = 0; i < length; ++i) values[i] *= scale;
}

static float q38_dot_f32_128(const float *left, const float *right)
{
#if defined(__AVX2__) && defined(__FMA__)
    __m256 sums[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps()
    };
    for (int i = 0; i < Q38_LINEAR_HEAD_DIM; i += 32) {
        for (int lane = 0; lane < 4; ++lane) {
            const int offset = i + lane * 8;
            sums[lane] = _mm256_fmadd_ps(
                _mm256_loadu_ps(left + offset),
                _mm256_loadu_ps(right + offset), sums[lane]);
        }
    }
    sums[0] = _mm256_add_ps(sums[0], sums[2]);
    sums[1] = _mm256_add_ps(sums[1], sums[3]);
    sums[0] = _mm256_add_ps(sums[0], sums[1]);
    const __m128 halves = _mm_add_ps(_mm256_castps256_ps128(sums[0]),
                                     _mm256_extractf128_ps(sums[0], 1));
    const __m128 pairs = _mm_hadd_ps(halves, halves);
    return _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
#else
    double sum = 0.0;
    for (int i = 0; i < Q38_LINEAR_HEAD_DIM; ++i)
        sum += (double)(left[i] * right[i]);
    return (float)sum;
#endif
}

#if !defined(__AVX2__) || !defined(__FMA__)
static void q38_mad_f32_128(float *output, const float *input, float scale)
{
#if defined(__AVX2__) && defined(__FMA__)
    const __m256 factor = _mm256_set1_ps(scale);
    for (int i = 0; i < Q38_LINEAR_HEAD_DIM; i += 8) {
        const __m256 value = _mm256_fmadd_ps(
            _mm256_loadu_ps(input + i), factor,
            _mm256_loadu_ps(output + i));
        _mm256_storeu_ps(output + i, value);
    }
#else
    for (int i = 0; i < Q38_LINEAR_HEAD_DIM; ++i)
        output[i] += input[i] * scale;
#endif
}
#endif

static float q38_mad_dot_f32_128(float *output, const float *input,
                                 float scale, const float *right)
{
#if defined(__AVX2__) && defined(__FMA__)
    __m256 sums[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps()
    };
    const __m256 factor = _mm256_set1_ps(scale);
    for (int i = 0; i < Q38_LINEAR_HEAD_DIM; i += 32) {
        for (int lane = 0; lane < 4; ++lane) {
            const int offset = i + lane * 8;
            const __m256 updated = _mm256_fmadd_ps(
                _mm256_loadu_ps(input + offset), factor,
                _mm256_loadu_ps(output + offset));
            _mm256_storeu_ps(output + offset, updated);
            sums[lane] = _mm256_fmadd_ps(
                updated, _mm256_loadu_ps(right + offset), sums[lane]);
        }
    }
    sums[0] = _mm256_add_ps(sums[0], sums[2]);
    sums[1] = _mm256_add_ps(sums[1], sums[3]);
    sums[0] = _mm256_add_ps(sums[0], sums[1]);
    const __m128 halves = _mm_add_ps(_mm256_castps256_ps128(sums[0]),
                                     _mm256_extractf128_ps(sums[0], 1));
    const __m128 pairs = _mm_hadd_ps(halves, halves);
    return _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
#else
    q38_mad_f32_128(output, input, scale);
    return q38_dot_f32_128(output, right);
#endif
}

static void q38_rope(float *values, int heads, uint32_t position)
{
    const float theta_scale = powf(10000000.0f, -2.0f / 64.0f);
    for (int head = 0; head < heads; ++head) {
        float *vector = values + (size_t)head * Q38_ATTN_HEAD_DIM;
        float theta = (float)position;
        for (int pair = 0; pair < 32; ++pair) {
            const float cosine = cosf(theta);
            const float sine = sinf(theta);
            const float x0 = vector[pair];
            const float x1 = vector[pair + 32];
            vector[pair] = x0 * cosine - x1 * sine;
            vector[pair + 32] = x0 * sine + x1 * cosine;
            theta *= theta_scale;
        }
    }
}

static int q38_linear_core(Q38Model *model, int layer, float *qkv,
                           const float *z, float *beta, float *alpha,
                           float *attention)
{
    const Q38Layer *weights = &model->layers[layer];
    const int recurrent_index = layer - (layer + 1) / 4;
    float *conv_state = model->conv_state +
        (size_t)recurrent_index * Q38_LINEAR_QKV_DIM * 3u;
    const Q38GGUFTensor *conv = weights->attention.linear.conv;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int channel = 0; channel < Q38_LINEAR_QKV_DIM; ++channel) {
        float *state = conv_state + (size_t)channel * 3u;
        const float *kernel = (const float *)(conv->data + (size_t)channel * 4u * sizeof(float));
        float value = 0.0f;
        value += state[0] * kernel[0];
        value += state[1] * kernel[1];
        value += state[2] * kernel[2];
        value += qkv[channel] * kernel[3];
        state[0] = state[1];
        state[1] = state[2];
        state[2] = qkv[channel];
        qkv[channel] = value;
    }
    q38_silu_inplace(qkv, Q38_LINEAR_QKV_DIM);

    float *query = qkv;
    float *key = qkv + Q38_LINEAR_QK_DIM;
    float *value = qkv + 2 * Q38_LINEAR_QK_DIM;
    for (int head = 0; head < Q38_LINEAR_QK_HEADS; ++head) {
        q38_l2norm(query + head * Q38_LINEAR_HEAD_DIM, Q38_LINEAR_HEAD_DIM);
        q38_l2norm(key + head * Q38_LINEAR_HEAD_DIM, Q38_LINEAR_HEAD_DIM);
    }
    for (int head = 0; head < Q38_LINEAR_V_HEADS; ++head) {
        beta[head] = q38_sigmoid(beta[head]);
        alpha[head] = q38_scalar(weights->attention.linear.a, (uint64_t)head) *
            q38_softplus(alpha[head] +
                         q38_scalar(weights->attention.linear.dt, (uint64_t)head));
    }

    float *delta_state = model->delta_state +
        (size_t)recurrent_index * Q38_LINEAR_V_HEADS *
        Q38_LINEAR_HEAD_DIM * Q38_LINEAR_HEAD_DIM;
    const float query_scale = 1.0f / sqrtf((float)Q38_LINEAR_HEAD_DIM);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int value_head = 0; value_head < Q38_LINEAR_V_HEADS; ++value_head) {
        const int key_head = value_head % Q38_LINEAR_QK_HEADS;
        const float *q = query + key_head * Q38_LINEAR_HEAD_DIM;
        const float *k = key + key_head * Q38_LINEAR_HEAD_DIM;
        const float *v = value + value_head * Q38_LINEAR_HEAD_DIM;
        float *state = delta_state + (size_t)value_head *
            Q38_LINEAR_HEAD_DIM * Q38_LINEAR_HEAD_DIM;
        float *head_output = attention + value_head * Q38_LINEAR_HEAD_DIM;
        const float decay = expf(alpha[value_head]);
        for (int i = 0; i < Q38_LINEAR_HEAD_DIM * Q38_LINEAR_HEAD_DIM; ++i)
            state[i] *= decay;
        for (int column = 0; column < Q38_LINEAR_HEAD_DIM; ++column) {
            float *state_column = state + column * Q38_LINEAR_HEAD_DIM;
            const float prediction = q38_dot_f32_128(state_column, k);
            const float delta = (v[column] - prediction) * beta[value_head];
            const float result = q38_mad_dot_f32_128(
                state_column, k, delta, q);
            head_output[column] = result * query_scale;
        }
    }

    for (int head = 0; head < Q38_LINEAR_V_HEADS; ++head) {
        float *head_output = attention + head * Q38_LINEAR_HEAD_DIM;
        q38_rmsnorm(head_output, head_output, weights->attention.linear.norm,
                    Q38_LINEAR_HEAD_DIM);
        const float *head_gate = z + head * Q38_LINEAR_HEAD_DIM;
        q38_silu_multiply(head_output, head_gate, Q38_LINEAR_HEAD_DIM);
    }
    return 1;
}

static int q38_linear_attention(Q38Model *model, int layer,
                                const float *input, float *output)
{
    Q38Scratch *scratch = &model->scratch;
    const Q38Layer *weights = &model->layers[layer];
    if (!q38_quantize_input(scratch, input, Q38_HIDDEN) ||
        !q38_project(scratch->wide0, input, scratch->quantized, Q38_HIDDEN,
                     weights->attention.linear.qkv) ||
        !q38_project(scratch->wide1, input, scratch->quantized, Q38_HIDDEN,
                     weights->attention.linear.z) ||
        !q38_project(scratch->beta, input, scratch->quantized, Q38_HIDDEN,
                     weights->attention.linear.beta) ||
        !q38_project(scratch->alpha, input, scratch->quantized, Q38_HIDDEN,
                     weights->attention.linear.alpha) ||
        !q38_linear_core(model, layer, scratch->wide0, scratch->wide1,
                         scratch->beta, scratch->alpha, scratch->attn)) return 0;
    if (!q38_quantize_input(scratch, scratch->attn, Q38_LINEAR_V_DIM)) return 0;
    return q38_project(output, scratch->attn, scratch->quantized,
                       Q38_LINEAR_V_DIM,
                       weights->attention.linear.out);
}

static int q38_full_core(Q38Model *model, int layer, const float *q_and_gate,
                         float *key, const float *value, float *attention,
                         uint32_t current_position)
{
    Q38Scratch *scratch = &model->scratch;
    const Q38Layer *weights = &model->layers[layer];
    for (int head = 0; head < Q38_ATTN_HEADS; ++head) {
        memcpy(scratch->q + head * Q38_ATTN_HEAD_DIM,
               q_and_gate + head * 2 * Q38_ATTN_HEAD_DIM,
               Q38_ATTN_HEAD_DIM * sizeof(float));
        memcpy(scratch->gate + head * Q38_ATTN_HEAD_DIM,
               q_and_gate + head * 2 * Q38_ATTN_HEAD_DIM + Q38_ATTN_HEAD_DIM,
               Q38_ATTN_HEAD_DIM * sizeof(float));
        q38_rmsnorm(scratch->q + head * Q38_ATTN_HEAD_DIM,
                    scratch->q + head * Q38_ATTN_HEAD_DIM,
                    weights->attention.full.q_norm, Q38_ATTN_HEAD_DIM);
    }
    for (int head = 0; head < Q38_ATTN_KV_HEADS; ++head) {
        q38_rmsnorm(key + head * Q38_ATTN_HEAD_DIM,
                    key + head * Q38_ATTN_HEAD_DIM,
                    weights->attention.full.k_norm, Q38_ATTN_HEAD_DIM);
    }
    q38_rope(scratch->q, Q38_ATTN_HEADS, current_position);
    q38_rope(key, Q38_ATTN_KV_HEADS, current_position);

    const int full_index = layer / 4;
    const size_t cache_row = ((size_t)full_index * model->context_length + current_position) *
                             Q38_ATTN_KV_DIM;
    memcpy(model->key_cache + cache_row, key, Q38_ATTN_KV_DIM * sizeof(float));
    memcpy(model->value_cache + cache_row, value, Q38_ATTN_KV_DIM * sizeof(float));

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int head = 0; head < Q38_ATTN_HEADS; ++head) {
        const int kv_head = head / (Q38_ATTN_HEADS / Q38_ATTN_KV_HEADS);
        const float *q = scratch->q + head * Q38_ATTN_HEAD_DIM;
        float *scores = scratch->scores + (size_t)head * model->context_length;
        float maximum = -INFINITY;
        for (uint32_t position = 0; position <= current_position; ++position) {
            const size_t row = ((size_t)full_index * model->context_length + position) *
                               Q38_ATTN_KV_DIM + kv_head * Q38_ATTN_HEAD_DIM;
            const float *k = model->key_cache + row;
            float score = 0.0f;
            for (int i = 0; i < Q38_ATTN_HEAD_DIM; ++i) score = fmaf(q[i], k[i], score);
            score *= 1.0f / sqrtf((float)Q38_ATTN_HEAD_DIM);
            scores[position] = score;
            if (score > maximum) maximum = score;
        }
        float denominator = 0.0f;
        for (uint32_t position = 0; position <= current_position; ++position) {
            scores[position] = expf(scores[position] - maximum);
            denominator += scores[position];
        }
        float *head_output = attention + head * Q38_ATTN_HEAD_DIM;
        memset(head_output, 0, Q38_ATTN_HEAD_DIM * sizeof(float));
        for (uint32_t position = 0; position <= current_position; ++position) {
            const size_t row = ((size_t)full_index * model->context_length + position) *
                               Q38_ATTN_KV_DIM + kv_head * Q38_ATTN_HEAD_DIM;
            const float *v = model->value_cache + row;
            const float probability = scores[position] / denominator;
            for (int i = 0; i < Q38_ATTN_HEAD_DIM; ++i)
                head_output[i] = fmaf(probability, v[i], head_output[i]);
        }
        const float *gate = scratch->gate + head * Q38_ATTN_HEAD_DIM;
        for (int i = 0; i < Q38_ATTN_HEAD_DIM; ++i)
            head_output[i] *= q38_sigmoid(gate[i]);
    }
    return 1;
}

static int q38_full_attention(Q38Model *model, int layer,
                              const float *input, float *output)
{
    Q38Scratch *scratch = &model->scratch;
    const Q38Layer *weights = &model->layers[layer];
    if (!q38_quantize_input(scratch, input, Q38_HIDDEN) ||
        !q38_project(scratch->wide0, input, scratch->quantized, Q38_HIDDEN,
                     weights->attention.full.q) ||
        !q38_project(scratch->k, input, scratch->quantized, Q38_HIDDEN,
                     weights->attention.full.k) ||
        !q38_project(scratch->v, input, scratch->quantized, Q38_HIDDEN,
                     weights->attention.full.v) ||
        !q38_full_core(model, layer, scratch->wide0, scratch->k, scratch->v,
                       scratch->attn, model->position)) return 0;
    q38_trace_vector("Qcur_full-", layer, scratch->wide0, Q38_ATTN_QG_DIM);
    q38_trace_vector("Kcur-", layer, scratch->k, Q38_ATTN_KV_DIM);
    q38_trace_vector("Vcur-", layer, scratch->v, Q38_ATTN_KV_DIM);
    q38_trace_vector("attn_gated-", layer, scratch->attn, Q38_ATTN_Q_DIM);
    if (!q38_quantize_input(scratch, scratch->attn, Q38_ATTN_Q_DIM)) return 0;
    if (!q38_project(output, scratch->attn, scratch->quantized,
                     Q38_ATTN_Q_DIM, weights->attention.full.out)) return 0;
    q38_trace_vector("attn_output-", layer, output, Q38_HIDDEN);
    return 1;
}

static int q38_full_attention_at(Q38Model *model, int layer,
                                 const float *input, float *output,
                                 uint32_t position)
{
    Q38Scratch *scratch = &model->scratch;
    const Q38Layer *weights = &model->layers[layer];
    if (!q38_quantize_input(scratch, input, Q38_HIDDEN) ||
        !q38_project(scratch->wide0, input, scratch->quantized, Q38_HIDDEN,
                     weights->attention.full.q) ||
        !q38_project(scratch->k, input, scratch->quantized, Q38_HIDDEN,
                     weights->attention.full.k) ||
        !q38_project(scratch->v, input, scratch->quantized, Q38_HIDDEN,
                     weights->attention.full.v) ||
        !q38_full_core(model, layer, scratch->wide0, scratch->k, scratch->v,
                       scratch->attn, position) ||
        !q38_quantize_input(scratch, scratch->attn, Q38_ATTN_Q_DIM)) return 0;
    return q38_project(output, scratch->attn, scratch->quantized,
                       Q38_ATTN_Q_DIM, weights->attention.full.out);
}

static int q38_mtp_layer_forward(Q38Model *model, uint32_t position)
{
    Q38Scratch *scratch = &model->scratch;
    const Q38Layer *weights = &model->layers[Q38_LAYERS];
    q38_rmsnorm(scratch->norm, scratch->hidden, weights->attn_norm, Q38_HIDDEN);
    if (!q38_full_attention_at(model, Q38_LAYERS, scratch->norm,
                               scratch->branch, position)) return 0;
    for (int i = 0; i < Q38_HIDDEN; ++i) scratch->hidden[i] += scratch->branch[i];

    q38_rmsnorm(scratch->norm, scratch->hidden, weights->post_norm, Q38_HIDDEN);
    if (!q38_quantize_input(scratch, scratch->norm, Q38_HIDDEN) ||
        !q38_project(scratch->wide0, scratch->norm, scratch->quantized,
                     Q38_HIDDEN, weights->ffn_gate) ||
        !q38_project(scratch->wide1, scratch->norm, scratch->quantized,
                     Q38_HIDDEN, weights->ffn_up)) return 0;
    q38_swiglu(scratch->wide0, scratch->wide1, Q38_FFN);
    if (!q38_quantize_input(scratch, scratch->wide0, Q38_FFN) ||
        !q38_project(scratch->branch, scratch->wide0, scratch->quantized,
                     Q38_FFN, weights->ffn_down)) return 0;
    for (int i = 0; i < Q38_HIDDEN; ++i) scratch->hidden[i] += scratch->branch[i];
    return 1;
}

static int q38_layer_forward(Q38Model *model, int layer)
{
    Q38Scratch *scratch = &model->scratch;
    const Q38Layer *weights = &model->layers[layer];
    q38_rmsnorm(scratch->norm, scratch->hidden, weights->attn_norm, Q38_HIDDEN);
    const int attention_ok = (layer + 1) % 4 == 0
        ? q38_full_attention(model, layer, scratch->norm, scratch->branch)
        : q38_linear_attention(model, layer, scratch->norm, scratch->branch);
    if (!attention_ok) return 0;
    q38_trace_vector("attention_branch-", layer, scratch->branch, Q38_HIDDEN);
    for (int i = 0; i < Q38_HIDDEN; ++i) scratch->hidden[i] += scratch->branch[i];
    q38_trace_vector("attn_residual-", layer, scratch->hidden, Q38_HIDDEN);

    q38_rmsnorm(scratch->norm, scratch->hidden, weights->post_norm, Q38_HIDDEN);
    if (!q38_quantize_input(scratch, scratch->norm, Q38_HIDDEN) ||
        !q38_project(scratch->wide0, scratch->norm, scratch->quantized,
                     Q38_HIDDEN,
                     weights->ffn_gate) ||
        !q38_project(scratch->wide1, scratch->norm, scratch->quantized,
                     Q38_HIDDEN,
                     weights->ffn_up)) return 0;
    q38_swiglu(scratch->wide0, scratch->wide1, Q38_FFN);
    if (!q38_quantize_input(scratch, scratch->wide0, Q38_FFN) ||
        !q38_project(scratch->branch, scratch->wide0, scratch->quantized,
                     Q38_FFN,
                     weights->ffn_down)) return 0;
    q38_trace_vector("ffn_out-", layer, scratch->branch, Q38_HIDDEN);
    for (int i = 0; i < Q38_HIDDEN; ++i) scratch->hidden[i] += scratch->branch[i];
    return 1;
}

static void q38_batch_free(Q38BatchScratch *scratch)
{
    if (!scratch) return;
    free(scratch->hidden);
    free(scratch->norm);
    free(scratch->branch);
    free(scratch->wide0);
    free(scratch->wide1);
    free(scratch->key);
    free(scratch->value);
    free(scratch->attention);
    free(scratch->beta);
    free(scratch->alpha);
    free(scratch->target_norm);
    free(scratch->quantized);
    memset(scratch, 0, sizeof(*scratch));
}

static int q38_batch_alloc(Q38BatchScratch *scratch, uint32_t count)
{
    memset(scratch, 0, sizeof(*scratch));
#define BATCH_ALLOC(field, elements) do { \
    scratch->field = (float *)calloc((size_t)count * (elements), sizeof(float)); \
    if (!scratch->field) { q38_batch_free(scratch); return 0; } \
} while (0)
    BATCH_ALLOC(hidden, Q38_HIDDEN);
    BATCH_ALLOC(norm, Q38_HIDDEN);
    BATCH_ALLOC(branch, Q38_HIDDEN);
    BATCH_ALLOC(wide0, Q38_FFN);
    BATCH_ALLOC(wide1, Q38_FFN);
    BATCH_ALLOC(key, Q38_ATTN_KV_DIM);
    BATCH_ALLOC(value, Q38_ATTN_KV_DIM);
    BATCH_ALLOC(attention, Q38_LINEAR_V_DIM);
    BATCH_ALLOC(beta, Q38_LINEAR_V_HEADS);
    BATCH_ALLOC(alpha, Q38_LINEAR_V_HEADS);
    BATCH_ALLOC(target_norm, Q38_HIDDEN);
#undef BATCH_ALLOC
    scratch->quantized = (Q38Q8KBlock *)calloc(
        (size_t)count * (Q38_FFN / Q38_Q8_K_BLOCK_SIZE),
        sizeof(Q38Q8KBlock));
    if (!scratch->quantized) {
        q38_batch_free(scratch);
        return 0;
    }
    return 1;
}

static int q38_batch_quantize(Q38BatchScratch *scratch, const float *input,
                              uint32_t count, uint64_t length)
{
    const uint64_t blocks = length / Q38_Q8_K_BLOCK_SIZE;
    for (uint32_t token = 0; token < count; ++token) {
        if (!q38_quantize_q8_k(scratch->quantized + (uint64_t)token * blocks,
                               input + (uint64_t)token * length, length)) return 0;
    }
    return 1;
}

static int q38_batch_project(float *output, const float *input,
                             const Q38Q8KBlock *quantized,
                             uint32_t count, uint64_t length,
                             const Q38GGUFTensor *weight)
{
    if (q38_matrix_type_q8_k(weight->type))
        return q38_tensor_gemm_q8_k(output, quantized, count, length, weight);
    return q38_tensor_gemm_f32(output, input, count, length, weight);
}

static void q38_batch_norm(float *output, const float *input,
                           const Q38GGUFTensor *weight, uint32_t count)
{
    for (uint32_t token = 0; token < count; ++token) {
        q38_rmsnorm(output + (uint64_t)token * Q38_HIDDEN,
                    input + (uint64_t)token * Q38_HIDDEN,
                    weight, Q38_HIDDEN);
    }
}

static int q38_batch_linear_attention(Q38Model *model,
                                      Q38BatchScratch *scratch,
                                      int layer, uint32_t count)
{
    const Q38Layer *weights = &model->layers[layer];
    if (!q38_batch_quantize(scratch, scratch->norm, count, Q38_HIDDEN) ||
        !q38_batch_project(scratch->wide0, scratch->norm, scratch->quantized,
                           count, Q38_HIDDEN, weights->attention.linear.qkv) ||
        !q38_batch_project(scratch->wide1, scratch->norm, scratch->quantized,
                           count, Q38_HIDDEN, weights->attention.linear.z) ||
        !q38_batch_project(scratch->beta, scratch->norm, scratch->quantized,
                           count, Q38_HIDDEN, weights->attention.linear.beta) ||
        !q38_batch_project(scratch->alpha, scratch->norm, scratch->quantized,
                           count, Q38_HIDDEN, weights->attention.linear.alpha)) return 0;
    for (uint32_t token = 0; token < count; ++token) {
        if (!q38_linear_core(model, layer,
                scratch->wide0 + (uint64_t)token * Q38_LINEAR_QKV_DIM,
                scratch->wide1 + (uint64_t)token * Q38_LINEAR_V_DIM,
                scratch->beta + (uint64_t)token * Q38_LINEAR_V_HEADS,
                scratch->alpha + (uint64_t)token * Q38_LINEAR_V_HEADS,
                scratch->attention + (uint64_t)token * Q38_LINEAR_V_DIM)) return 0;
    }
    if (!q38_batch_quantize(scratch, scratch->attention, count,
                            Q38_LINEAR_V_DIM)) return 0;
    return q38_batch_project(scratch->branch, scratch->attention,
                             scratch->quantized, count, Q38_LINEAR_V_DIM,
                             weights->attention.linear.out);
}

static int q38_batch_full_attention(Q38Model *model,
                                    Q38BatchScratch *scratch,
                                    int layer, uint32_t count,
                                    uint32_t base_position)
{
    const Q38Layer *weights = &model->layers[layer];
    if (!q38_batch_quantize(scratch, scratch->norm, count, Q38_HIDDEN) ||
        !q38_batch_project(scratch->wide0, scratch->norm, scratch->quantized,
                           count, Q38_HIDDEN, weights->attention.full.q) ||
        !q38_batch_project(scratch->key, scratch->norm, scratch->quantized,
                           count, Q38_HIDDEN, weights->attention.full.k) ||
        !q38_batch_project(scratch->value, scratch->norm, scratch->quantized,
                           count, Q38_HIDDEN, weights->attention.full.v)) return 0;
    for (uint32_t token = 0; token < count; ++token) {
        if (!q38_full_core(model, layer,
                scratch->wide0 + (uint64_t)token * Q38_ATTN_QG_DIM,
                scratch->key + (uint64_t)token * Q38_ATTN_KV_DIM,
                scratch->value + (uint64_t)token * Q38_ATTN_KV_DIM,
                scratch->attention + (uint64_t)token * Q38_ATTN_Q_DIM,
                base_position + token)) return 0;
    }
    if (!q38_batch_quantize(scratch, scratch->attention, count,
                            Q38_ATTN_Q_DIM)) return 0;
    return q38_batch_project(scratch->branch, scratch->attention,
                             scratch->quantized, count, Q38_ATTN_Q_DIM,
                             weights->attention.full.out);
}

static int q38_batch_layer(Q38Model *model, Q38BatchScratch *scratch,
                           int layer, uint32_t count,
                           uint32_t base_position)
{
    const Q38Layer *weights = &model->layers[layer];
    q38_batch_norm(scratch->norm, scratch->hidden, weights->attn_norm, count);
    const int attention_ok = (layer + 1) % 4 == 0
        ? q38_batch_full_attention(model, scratch, layer, count, base_position)
        : q38_batch_linear_attention(model, scratch, layer, count);
    if (!attention_ok) return 0;
    const uint64_t hidden_count = (uint64_t)count * Q38_HIDDEN;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (uint64_t i = 0; i < hidden_count; ++i)
        scratch->hidden[i] += scratch->branch[i];

    q38_batch_norm(scratch->norm, scratch->hidden, weights->post_norm, count);
    if (!q38_batch_quantize(scratch, scratch->norm, count, Q38_HIDDEN) ||
        !q38_batch_project(scratch->wide0, scratch->norm, scratch->quantized,
                           count, Q38_HIDDEN, weights->ffn_gate) ||
        !q38_batch_project(scratch->wide1, scratch->norm, scratch->quantized,
                           count, Q38_HIDDEN, weights->ffn_up)) return 0;
    const uint64_t wide_count = (uint64_t)count * Q38_FFN;
    q38_swiglu(scratch->wide0, scratch->wide1, wide_count);
    if (!q38_batch_quantize(scratch, scratch->wide0, count, Q38_FFN) ||
        !q38_batch_project(scratch->branch, scratch->wide0, scratch->quantized,
                           count, Q38_FFN, weights->ffn_down)) return 0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (uint64_t i = 0; i < hidden_count; ++i)
        scratch->hidden[i] += scratch->branch[i];
    return 1;
}

static int q38_batch_mtp_layer(Q38Model *model, Q38BatchScratch *scratch,
                               uint32_t count, uint32_t base_position)
{
    const Q38Layer *weights = &model->layers[Q38_LAYERS];
    q38_batch_norm(scratch->norm, scratch->hidden, weights->attn_norm, count);
    if (!q38_batch_full_attention(model, scratch, Q38_LAYERS, count,
                                  base_position)) return 0;
    const uint64_t hidden_count = (uint64_t)count * Q38_HIDDEN;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (uint64_t i = 0; i < hidden_count; ++i)
        scratch->hidden[i] += scratch->branch[i];

    q38_batch_norm(scratch->norm, scratch->hidden, weights->post_norm, count);
    if (!q38_batch_quantize(scratch, scratch->norm, count, Q38_HIDDEN) ||
        !q38_batch_project(scratch->wide0, scratch->norm, scratch->quantized,
                           count, Q38_HIDDEN, weights->ffn_gate) ||
        !q38_batch_project(scratch->wide1, scratch->norm, scratch->quantized,
                           count, Q38_HIDDEN, weights->ffn_up)) return 0;
    q38_swiglu(scratch->wide0, scratch->wide1, (uint64_t)count * Q38_FFN);
    if (!q38_batch_quantize(scratch, scratch->wide0, count, Q38_FFN) ||
        !q38_batch_project(scratch->branch, scratch->wide0, scratch->quantized,
                           count, Q38_FFN, weights->ffn_down)) return 0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (uint64_t i = 0; i < hidden_count; ++i)
        scratch->hidden[i] += scratch->branch[i];
    return 1;
}

static size_t q38_conv_state_count(void)
{
    return (size_t)Q38_RECURRENT_LAYERS * Q38_LINEAR_QKV_DIM * 3u;
}

static size_t q38_delta_state_count(void)
{
    return (size_t)Q38_RECURRENT_LAYERS * Q38_LINEAR_V_HEADS *
           Q38_LINEAR_HEAD_DIM * Q38_LINEAR_HEAD_DIM;
}

static void q38_checkpoint_state(Q38Model *model)
{
    memcpy(model->checkpoint_conv_state, model->conv_state,
           q38_conv_state_count() * sizeof(float));
    memcpy(model->checkpoint_delta_state, model->delta_state,
           q38_delta_state_count() * sizeof(float));
    memcpy(model->checkpoint_mtp_hidden, model->mtp_input_hidden,
           Q38_HIDDEN * sizeof(float));
    model->checkpoint_position = model->position;
    model->checkpoint_mtp_position = model->mtp_position;
}

static void q38_restore_target_state(Q38Model *model)
{
    memcpy(model->conv_state, model->checkpoint_conv_state,
           q38_conv_state_count() * sizeof(float));
    memcpy(model->delta_state, model->checkpoint_delta_state,
           q38_delta_state_count() * sizeof(float));
    model->position = model->checkpoint_position;
}

static void q38_restore_mtp_state(Q38Model *model)
{
    memcpy(model->mtp_input_hidden, model->checkpoint_mtp_hidden,
           Q38_HIDDEN * sizeof(float));
    model->mtp_position = model->checkpoint_mtp_position;
}

static uint32_t q38_argmax(const float *values, uint32_t count)
{
    uint32_t best = 0;
    float maximum = values[0];
    for (uint32_t i = 1; i < count; ++i) {
        if (values[i] > maximum) {
            maximum = values[i];
            best = i;
        }
    }
    return best;
}

static int q38_batch_target_eval(Q38Model *model, Q38BatchScratch *scratch,
                                 const uint32_t *tokens, uint32_t token_count,
                                 float *batch_logits)
{
    int ok = 1;
    for (uint32_t token = 0; ok && token < token_count; ++token) {
        if (tokens[token] >= Q38_VOCAB ||
            !q38_tensor_row_f32(scratch->hidden +
                                     (uint64_t)token * Q38_HIDDEN,
                                 model->embedding, tokens[token])) ok = 0;
    }
    const uint32_t base_position = model->position;
    for (int layer = 0; ok && layer < Q38_LAYERS; ++layer)
        ok = q38_batch_layer(model, scratch, layer, token_count, base_position);
    if (ok) {
        q38_batch_norm(scratch->target_norm, scratch->hidden,
                       model->output_norm, token_count);
        ok = q38_batch_quantize(scratch, scratch->target_norm, token_count,
                                Q38_HIDDEN) &&
             q38_batch_project(batch_logits, scratch->target_norm,
                               scratch->quantized, token_count, Q38_HIDDEN,
                               model->output);
    }
    if (ok) model->position = base_position + token_count;
    return ok;
}

static int q38_mtp_catchup(Q38Model *model, Q38BatchScratch *scratch,
                           const uint32_t *tokens, uint32_t count,
                           uint32_t base_position)
{
    for (uint32_t token = 0; token < count; ++token) {
        float *embedding = scratch->hidden + (uint64_t)token * Q38_HIDDEN;
        float *input = scratch->wide0 + (uint64_t)token * (2u * Q38_HIDDEN);
        if (!q38_tensor_row_f32(embedding, model->mtp_embedding, tokens[token]))
            return 0;
        q38_rmsnorm(input, embedding, model->mtp.enorm, Q38_HIDDEN);
        const float *previous = token == 0
            ? model->mtp_input_hidden
            : scratch->target_norm + (uint64_t)(token - 1) * Q38_HIDDEN;
        q38_rmsnorm(input + Q38_HIDDEN, previous, model->mtp.hnorm, Q38_HIDDEN);
    }
    if (!q38_batch_quantize(scratch, scratch->wide0, count,
                            2u * Q38_HIDDEN) ||
        !q38_batch_project(scratch->hidden, scratch->wide0,
                           scratch->quantized, count, 2u * Q38_HIDDEN,
                           model->mtp.eh_proj) ||
        !q38_batch_mtp_layer(model, scratch, count, base_position)) return 0;
    model->mtp_position = base_position + count;
    memcpy(model->mtp_input_hidden,
           scratch->target_norm + (uint64_t)(count - 1) * Q38_HIDDEN,
           Q38_HIDDEN * sizeof(float));
    return 1;
}

static int q38_alloc_scratch(Q38Model *model)
{
#define ALLOC(field, count) do { \
    model->scratch.field = (float *)calloc((size_t)(count), sizeof(float)); \
    if (!model->scratch.field) return 0; \
} while (0)
    ALLOC(hidden, Q38_HIDDEN);
    ALLOC(norm, Q38_HIDDEN);
    ALLOC(branch, Q38_HIDDEN);
    ALLOC(wide0, Q38_FFN);
    ALLOC(wide1, Q38_FFN);
    ALLOC(q, Q38_ATTN_Q_DIM);
    ALLOC(gate, Q38_ATTN_Q_DIM);
    ALLOC(k, Q38_ATTN_KV_DIM);
    ALLOC(v, Q38_ATTN_KV_DIM);
    ALLOC(attn, Q38_LINEAR_V_DIM);
    ALLOC(scores, (size_t)Q38_ATTN_HEADS * model->context_length);
    ALLOC(beta, Q38_LINEAR_V_HEADS);
    ALLOC(alpha, Q38_LINEAR_V_HEADS);
    ALLOC(logits, Q38_VOCAB);
#undef ALLOC
    model->scratch.quantized = (Q38Q8KBlock *)calloc(
        Q38_FFN / Q38_Q8_K_BLOCK_SIZE, sizeof(Q38Q8KBlock));
    if (!model->scratch.quantized) return 0;
    return 1;
}

#if defined(__AVX2__)
static uint64_t q38_available_memory(void)
{
    FILE *file = fopen("/proc/meminfo", "r");
    if (file) {
        char line[256];
        while (fgets(line, sizeof(line), file)) {
            unsigned long long kib = 0;
            if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) {
                fclose(file);
                return kib <= UINT64_MAX / 1024u ? (uint64_t)kib * 1024u : 0;
            }
        }
        fclose(file);
    }

    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0 ||
        (uint64_t)pages > UINT64_MAX / (uint64_t)page_size) return 0;
    return (uint64_t)pages * (uint64_t)page_size;
}
#endif

static int q38_iq1_repack_enabled(void)
{
#if !defined(__AVX2__)
    return 0;
#else
    const char *setting = getenv("QWEN38_REPACK");
    if (setting && *setting) {
        if (strcmp(setting, "0") == 0 || strcmp(setting, "false") == 0 ||
            strcmp(setting, "no") == 0) return 0;
        if (strcmp(setting, "1") == 0 || strcmp(setting, "true") == 0 ||
            strcmp(setting, "yes") == 0) return 1;
    }

    const uint64_t minimum = UINT64_C(12) * 1024u * 1024u * 1024u;
    if (q38_available_memory() < minimum) return 0;

    struct rlimit limit;
    if (getrlimit(RLIMIT_AS, &limit) == 0 && limit.rlim_cur != RLIM_INFINITY &&
        (uint64_t)limit.rlim_cur < minimum) return 0;
    return 1;
#endif
}

Q38Model *q38_model_open_gguf(const char *path, uint32_t context_length)
{
    if (!path || context_length == 0) return NULL;
    Q38Model *model = (Q38Model *)calloc(1, sizeof(*model));
    if (!model) return NULL;
    model->gguf.fd = -1;
    model->mtp_gguf.fd = -1;
    model->context_length = context_length;
    if (!q38_gguf_open(&model->gguf, path) || !q38_bind_weights(model)) {
        q38_model_close(model);
        return NULL;
    }
    if (q38_iq1_repack_enabled() &&
        !q38_prepare_iq1_s_repacks(&model->gguf)) {
        fprintf(stderr,
                "qwen38: IQ1 runtime repack unavailable; using packed weights\n");
    }

    const size_t conv_count = (size_t)Q38_RECURRENT_LAYERS * Q38_LINEAR_QKV_DIM * 3u;
    const size_t delta_count = (size_t)Q38_RECURRENT_LAYERS * Q38_LINEAR_V_HEADS *
                               Q38_LINEAR_HEAD_DIM * Q38_LINEAR_HEAD_DIM;
    const size_t kv_count = (size_t)Q38_TOTAL_FULL_LAYERS * context_length *
                            Q38_ATTN_KV_DIM;
    model->conv_state = (float *)calloc(conv_count, sizeof(float));
    model->delta_state = (float *)calloc(delta_count, sizeof(float));
    model->key_cache = (float *)calloc(kv_count, sizeof(float));
    model->value_cache = (float *)calloc(kv_count, sizeof(float));
    model->mtp_input_hidden = (float *)calloc(Q38_HIDDEN, sizeof(float));
    if (!model->conv_state || !model->delta_state || !model->key_cache ||
        !model->value_cache || !model->mtp_input_hidden ||
        !q38_alloc_scratch(model)) {
        fprintf(stderr, "qwen38: unable to allocate inference state\n");
        q38_model_close(model);
        return NULL;
    }
    return model;
}

int q38_model_attach_mtp_gguf(Q38Model *model, const char *path)
{
    if (!model || !path || model->position != 0 || model->mtp_position != 0 ||
        model->mtp_enabled || model->mtp_weights_loaded ||
        model->mtp_gguf.fd >= 0) return 0;
    if (!q38_gguf_open(&model->mtp_gguf, path) ||
        !q38_bind_mtp_weights(model, &model->mtp_gguf)) {
        memset(&model->layers[Q38_LAYERS], 0,
               sizeof(model->layers[Q38_LAYERS]));
        memset(&model->mtp, 0, sizeof(model->mtp));
        model->mtp_embedding = NULL;
        model->mtp_output = NULL;
        model->mtp_weights_loaded = 0;
        q38_gguf_close(&model->mtp_gguf);
        return 0;
    }
    return 1;
}

void q38_model_reset(Q38Model *model)
{
    if (!model) return;
    memset(model->conv_state, 0,
           (size_t)Q38_RECURRENT_LAYERS * Q38_LINEAR_QKV_DIM * 3u * sizeof(float));
    memset(model->delta_state, 0,
           (size_t)Q38_RECURRENT_LAYERS * Q38_LINEAR_V_HEADS *
           Q38_LINEAR_HEAD_DIM * Q38_LINEAR_HEAD_DIM * sizeof(float));
    memset(model->key_cache, 0,
           (size_t)Q38_TOTAL_FULL_LAYERS * model->context_length *
           Q38_ATTN_KV_DIM * sizeof(float));
    memset(model->value_cache, 0,
           (size_t)Q38_TOTAL_FULL_LAYERS * model->context_length *
           Q38_ATTN_KV_DIM * sizeof(float));
    memset(model->mtp_input_hidden, 0, Q38_HIDDEN * sizeof(float));
    model->position = 0;
    model->mtp_position = 0;
}

void q38_model_close(Q38Model *model)
{
    if (!model) return;
    free(model->conv_state);
    free(model->delta_state);
    free(model->key_cache);
    free(model->value_cache);
    free(model->mtp_input_hidden);
    free(model->checkpoint_conv_state);
    free(model->checkpoint_delta_state);
    free(model->checkpoint_mtp_hidden);
    free(model->scratch.hidden);
    free(model->scratch.norm);
    free(model->scratch.branch);
    free(model->scratch.wide0);
    free(model->scratch.wide1);
    free(model->scratch.q);
    free(model->scratch.gate);
    free(model->scratch.k);
    free(model->scratch.v);
    free(model->scratch.attn);
    free(model->scratch.scores);
    free(model->scratch.beta);
    free(model->scratch.alpha);
    free(model->scratch.logits);
    free(model->scratch.quantized);
    q38_release_iq1_s_repacks(&model->gguf);
    q38_gguf_close(&model->mtp_gguf);
    q38_gguf_close(&model->gguf);
    free(model);
}

int q38_model_forward_token_layers(Q38Model *model, uint32_t token_id,
                                   uint32_t layer_count, float *hidden_out)
{
    if (!model || !hidden_out || token_id >= Q38_VOCAB ||
        layer_count > Q38_LAYERS || model->position >= model->context_length) return 0;
    if (!q38_tensor_row_f32(model->scratch.hidden, model->embedding, token_id)) return 0;
    q38_trace_hidden("model.input_embed", -1, model->scratch.hidden);
    for (uint32_t layer = 0; layer < layer_count; ++layer) {
        if (!q38_layer_forward(model, (int)layer)) return 0;
        q38_trace_hidden("l_out-", (int)layer, model->scratch.hidden);
    }
    memcpy(hidden_out, model->scratch.hidden, Q38_HIDDEN * sizeof(float));
    ++model->position;
    return 1;
}

int q38_model_forward_token(Q38Model *model, uint32_t token_id,
                            const float **logits)
{
    if (!model || !logits || token_id >= Q38_VOCAB ||
        model->position >= model->context_length) return 0;
    if (!q38_tensor_row_f32(model->scratch.hidden, model->embedding, token_id)) return 0;
    q38_trace_hidden("model.input_embed", -1, model->scratch.hidden);
    for (int layer = 0; layer < Q38_LAYERS; ++layer) {
        if (!q38_layer_forward(model, layer)) return 0;
        q38_trace_hidden("l_out-", layer, model->scratch.hidden);
    }
    q38_rmsnorm(model->scratch.norm, model->scratch.hidden,
                model->output_norm, Q38_HIDDEN);
    if (model->mtp_enabled)
        memcpy(model->mtp_input_hidden, model->scratch.norm,
               Q38_HIDDEN * sizeof(float));
    if (!q38_quantize_input(&model->scratch, model->scratch.norm, Q38_HIDDEN) ||
        !q38_project(model->scratch.logits, model->scratch.norm,
                     model->scratch.quantized, Q38_HIDDEN,
                     model->output)) return 0;
    ++model->position;
    *logits = model->scratch.logits;
    return 1;
}

int q38_model_prefill(Q38Model *model, const uint32_t *tokens,
                      uint32_t token_count, const float **logits)
{
    if (!model || !tokens || !token_count || !logits ||
        token_count > model->context_length - model->position) return 0;
    Q38BatchScratch scratch;
    if (!q38_batch_alloc(&scratch, token_count)) {
        fprintf(stderr, "qwen38: unable to allocate prompt batch for %u tokens\n",
                token_count);
        return 0;
    }
    int ok = 1;
    for (uint32_t token = 0; ok && token < token_count; ++token) {
        if (tokens[token] >= Q38_VOCAB ||
            !q38_tensor_row_f32(scratch.hidden + (uint64_t)token * Q38_HIDDEN,
                                 model->embedding, tokens[token])) ok = 0;
    }
    const uint32_t base_position = model->position;
    for (int layer = 0; ok && layer < Q38_LAYERS; ++layer)
        ok = q38_batch_layer(model, &scratch, layer, token_count, base_position);
    if (ok) {
        q38_batch_norm(scratch.target_norm, scratch.hidden,
                       model->output_norm, token_count);
    }
    if (ok && model->mtp_enabled)
        ok = q38_mtp_catchup(model, &scratch, tokens, token_count,
                             base_position);
    if (ok) {
        const float *last = scratch.target_norm +
                            (uint64_t)(token_count - 1) * Q38_HIDDEN;
        memcpy(model->scratch.norm, last, Q38_HIDDEN * sizeof(float));
        ok = q38_quantize_input(&model->scratch, last, Q38_HIDDEN) &&
             q38_project(model->scratch.logits, model->scratch.norm,
                          model->scratch.quantized, Q38_HIDDEN, model->output);
    }
    if (ok) {
        model->position = base_position + token_count;
        *logits = model->scratch.logits;
    }
    q38_batch_free(&scratch);
    return ok;
}

int q38_model_enable_mtp(Q38Model *model)
{
    if (!model || model->position != 0 || model->mtp_position != 0) return 0;
    if (!model->mtp_weights_loaded) {
        fprintf(stderr,
                "qwen38: MTP weights are unavailable; attach the MTP GGUF first\n");
        return 0;
    }
    if (!model->checkpoint_conv_state)
        model->checkpoint_conv_state = (float *)malloc(
            q38_conv_state_count() * sizeof(float));
    if (!model->checkpoint_delta_state)
        model->checkpoint_delta_state = (float *)malloc(
            q38_delta_state_count() * sizeof(float));
    if (!model->checkpoint_mtp_hidden)
        model->checkpoint_mtp_hidden = (float *)malloc(
            Q38_HIDDEN * sizeof(float));
    if (!model->checkpoint_conv_state || !model->checkpoint_delta_state ||
        !model->checkpoint_mtp_hidden) {
        fprintf(stderr, "qwen38: unable to allocate speculative checkpoint\n");
        return 0;
    }
    model->mtp_enabled = 1;
    return 1;
}

int q38_model_mtp_forward(Q38Model *model, uint32_t token_id,
                          const float **logits)
{
    if (!model || !model->mtp_enabled || !logits || token_id >= Q38_VOCAB ||
        model->mtp_position >= model->context_length) return 0;
    Q38Scratch *scratch = &model->scratch;
    if (!q38_tensor_row_f32(scratch->hidden, model->mtp_embedding, token_id))
        return 0;
    q38_rmsnorm(scratch->wide0, scratch->hidden, model->mtp.enorm, Q38_HIDDEN);
    q38_rmsnorm(scratch->wide0 + Q38_HIDDEN, model->mtp_input_hidden,
                model->mtp.hnorm, Q38_HIDDEN);
    if (!q38_quantize_input(scratch, scratch->wide0, 2u * Q38_HIDDEN) ||
        !q38_project(scratch->hidden, scratch->wide0, scratch->quantized,
                     2u * Q38_HIDDEN, model->mtp.eh_proj) ||
        !q38_mtp_layer_forward(model, model->mtp_position)) return 0;
    q38_rmsnorm(scratch->norm, scratch->hidden,
                model->mtp.shared_head_norm, Q38_HIDDEN);
    memcpy(model->mtp_input_hidden, scratch->norm, Q38_HIDDEN * sizeof(float));
    if (!q38_quantize_input(scratch, scratch->norm, Q38_HIDDEN) ||
        !q38_project(scratch->logits, scratch->norm, scratch->quantized,
                     Q38_HIDDEN, model->mtp_output)) return 0;
    ++model->mtp_position;
    *logits = scratch->logits;
    return 1;
}

int q38_model_speculative_greedy(Q38Model *model, uint32_t token_id,
                                 uint32_t stop_token, uint32_t draft_count,
                                 uint32_t *accepted_tokens,
                                 uint32_t *accepted_count,
                                 const float **next_logits)
{
    if (!model || !model->mtp_enabled || !accepted_tokens ||
        !accepted_count || !next_logits || token_id >= Q38_VOCAB ||
        draft_count == 0 || model->position != model->mtp_position ||
        draft_count > model->context_length - model->position) return 0;

    uint32_t *drafts = (uint32_t *)malloc(draft_count * sizeof(uint32_t));
    uint32_t *inputs = (uint32_t *)malloc(draft_count * sizeof(uint32_t));
    float *batch_logits = (float *)malloc(
        (size_t)draft_count * Q38_VOCAB * sizeof(float));
    if (!drafts || !inputs || !batch_logits) {
        free(drafts);
        free(inputs);
        free(batch_logits);
        return 0;
    }

    q38_checkpoint_state(model);
    uint32_t actual_count = 0;
    uint32_t draft_input = token_id;
    int ok = 1;
    while (actual_count < draft_count) {
        const float *draft_logits = NULL;
        if (!q38_model_mtp_forward(model, draft_input, &draft_logits)) {
            ok = 0;
            break;
        }
        draft_input = q38_argmax(draft_logits, Q38_VOCAB);
        drafts[actual_count++] = draft_input;
        if (draft_input == stop_token) break;
    }
    if (!ok) {
        q38_restore_mtp_state(model);
        free(drafts);
        free(inputs);
        free(batch_logits);
        return 0;
    }

    inputs[0] = token_id;
    for (uint32_t i = 1; i < actual_count; ++i) inputs[i] = drafts[i - 1];
    Q38BatchScratch scratch;
    if (!q38_batch_alloc(&scratch, actual_count)) ok = 0;
    if (ok) ok = q38_batch_target_eval(model, &scratch, inputs,
                                       actual_count, batch_logits);

    uint32_t matched = 0;
    uint32_t next = 0;
    if (ok) {
        while (matched < actual_count) {
            const uint32_t target = q38_argmax(
                batch_logits + (uint64_t)matched * Q38_VOCAB, Q38_VOCAB);
            if (target != drafts[matched]) {
                next = target;
                break;
            }
            ++matched;
        }
        if (matched == actual_count) next = drafts[actual_count - 1];
    }

    const uint32_t consumed = matched < actual_count
        ? matched + 1u
        : actual_count;
    if (ok && consumed < actual_count) {
        q38_batch_free(&scratch);
        q38_restore_target_state(model);
        if (!q38_batch_alloc(&scratch, consumed)) ok = 0;
        if (ok) ok = q38_batch_target_eval(model, &scratch, inputs,
                                           consumed, batch_logits);
    }
    if (ok) {
        const float *last_logits = batch_logits +
                                   (uint64_t)(consumed - 1u) * Q38_VOCAB;
        memcpy(model->scratch.logits, last_logits,
               Q38_VOCAB * sizeof(float));
        q38_restore_mtp_state(model);
        ok = q38_mtp_catchup(model, &scratch, inputs, consumed,
                             model->checkpoint_mtp_position);
    }
    if (ok) {
        for (uint32_t i = 0; i < matched; ++i) accepted_tokens[i] = drafts[i];
        if (matched < actual_count) accepted_tokens[matched] = next;
        *accepted_count = consumed;
        *next_logits = model->scratch.logits;
    } else {
        q38_restore_target_state(model);
        q38_restore_mtp_state(model);
    }

    q38_batch_free(&scratch);
    free(drafts);
    free(inputs);
    free(batch_logits);
    return ok;
}

uint32_t q38_model_vocab_size(const Q38Model *model)
{
    return model ? Q38_VOCAB : 0;
}

uint32_t q38_model_position(const Q38Model *model)
{
    return model ? model->position : 0;
}

uint32_t q38_model_context_length(const Q38Model *model)
{
    return model ? model->context_length : 0;
}
