#include "qwen38_quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-const-variable"
#endif
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#undef GGML_COMMON_IMPL_C
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#define CHECK(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

static void set_half(uint8_t *data, uint16_t value)
{
    data[0] = (uint8_t)value;
    data[1] = (uint8_t)(value >> 8);
}

static int close_to(float actual, float expected)
{
    return fabsf(actual - expected) <= 1e-5f * fmaxf(1.0f, fabsf(expected));
}

static uint32_t next_random(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void set_random_block_scales(uint8_t *block, uint32_t type,
                                    uint16_t scale, uint16_t minimum)
{
    switch (type) {
    case Q38_GGML_Q2_K:
        set_half(block + 80, scale);
        set_half(block + 82, minimum);
        break;
    case Q38_GGML_Q3_K:
        set_half(block + 108, scale);
        break;
    case Q38_GGML_Q4_K:
    case Q38_GGML_Q5_K:
        set_half(block, scale);
        set_half(block + 2, minimum);
        break;
    case Q38_GGML_Q6_K:
        set_half(block + 208, scale);
        break;
    case Q38_GGML_IQ2_S:
    case Q38_GGML_IQ2_XXS:
    case Q38_GGML_IQ2_XS:
    case Q38_GGML_IQ3_XXS:
    case Q38_GGML_IQ3_S:
    case Q38_GGML_IQ4_XS:
    case Q38_GGML_IQ1_S:
        set_half(block, scale);
        break;
    case Q38_GGML_IQ4_NL:
        for (int group = 0; group < 8; ++group)
            set_half(block + group * 18, scale);
        break;
    case Q38_GGML_IQ1_M:
        set_half(block + 48, (uint16_t)((scale & 0x000fu) << 12));
        set_half(block + 50, (uint16_t)((scale & 0x00f0u) << 8));
        set_half(block + 52, (uint16_t)((scale & 0x0f00u) << 4));
        set_half(block + 54, (uint16_t)(scale & 0xf000u));
        break;
    default:
        break;
    }
}

static float reference_iq2_s_q8_k(const uint8_t block[82],
                                   const Q38Q8KBlock *input)
{
    const float d = 0.125f * q38_f16_to_f32(
        (uint16_t)block[0] | (uint16_t)block[1] << 8) * input->scale;
    const uint8_t *indices = block + 2;
    const uint8_t *signs = block + 34;
    const uint8_t *high = block + 66;
    const uint8_t *scales = block + 74;
    int weighted = 0;
    for (int group = 0; group < 8; ++group) {
        int subtotal[2] = {0, 0};
        for (int section = 0; section < 4; ++section) {
            const uint16_t grid_index = (uint16_t)indices[section] |
                (uint16_t)((high[group] << (8 - 2 * section)) & 0x300);
            const uint8_t *grid = (const uint8_t *)(iq2s_grid + grid_index);
            for (int lane = 0; lane < 8; ++lane) {
                const int sign = signs[section] & (1u << lane) ? -1 : 1;
                subtotal[section / 2] +=
                    input->quants[group * 32 + section * 8 + lane] *
                    grid[lane] * sign;
            }
        }
        weighted += (1 + 2 * (scales[group] & 15u)) * subtotal[0] +
                    (1 + 2 * (scales[group] >> 4)) * subtotal[1];
        indices += 4;
        signs += 4;
    }
    return d * weighted;
}

static float reference_iq2_xxs_q8_k(const uint8_t data[66],
                                     const Q38Q8KBlock *input)
{
    const float d = 0.125f * q38_f16_to_f32(
        (uint16_t)data[0] | (uint16_t)data[1] << 8) * input->scale;
    const uint8_t *codes = data + 2;
    int weighted = 0;
    for (int group = 0; group < 8; ++group, codes += 8) {
        const uint32_t metadata = (uint32_t)codes[4] |
            (uint32_t)codes[5] << 8 | (uint32_t)codes[6] << 16 |
            (uint32_t)codes[7] << 24;
        int subtotal = 0;
        for (int section = 0; section < 4; ++section) {
            const uint8_t *grid = (const uint8_t *)(iq2xxs_grid +
                                                    codes[section]);
            const uint8_t signs = ksigns_iq2xs[
                (metadata >> (7 * section)) & 127u];
            for (int lane = 0; lane < 8; ++lane) {
                const int sign = signs & (1u << lane) ? -1 : 1;
                subtotal += input->quants[group * 32 + section * 8 + lane] *
                            grid[lane] * sign;
            }
        }
        weighted += (1 + 2 * (int)(metadata >> 28)) * subtotal;
    }
    return d * weighted;
}

static float reference_iq2_xs_q8_k(const uint8_t data[74],
                                    const Q38Q8KBlock *input)
{
    const float d = 0.125f * q38_f16_to_f32(
        (uint16_t)data[0] | (uint16_t)data[1] << 8) * input->scale;
    const uint8_t *codes = data + 2;
    const uint8_t *scales = data + 66;
    int weighted = 0;
    for (int group = 0; group < 8; ++group, codes += 8) {
        int subtotal[2] = {0, 0};
        for (int section = 0; section < 4; ++section) {
            const uint16_t code = (uint16_t)codes[section * 2] |
                                  (uint16_t)codes[section * 2 + 1] << 8;
            const uint8_t *grid = (const uint8_t *)(iq2xs_grid +
                                                    (code & 511u));
            const uint8_t signs = ksigns_iq2xs[code >> 9];
            for (int lane = 0; lane < 8; ++lane) {
                const int sign = signs & (1u << lane) ? -1 : 1;
                subtotal[section / 2] += input->quants[
                    group * 32 + section * 8 + lane] * grid[lane] * sign;
            }
        }
        weighted += (1 + 2 * (scales[group] & 15u)) * subtotal[0] +
                    (1 + 2 * (scales[group] >> 4)) * subtotal[1];
    }
    return d * weighted;
}

static float reference_iq3_xxs_q8_k(const uint8_t data[98],
                                     const Q38Q8KBlock *input)
{
    const float d = 0.25f * q38_f16_to_f32(
        (uint16_t)data[0] | (uint16_t)data[1] << 8) * input->scale;
    const uint8_t *indices = data + 2;
    const uint8_t *metadata = data + 66;
    int weighted = 0;
    for (int group = 0; group < 8; ++group) {
        const uint32_t packed = (uint32_t)metadata[group * 4] |
            (uint32_t)metadata[group * 4 + 1] << 8 |
            (uint32_t)metadata[group * 4 + 2] << 16 |
            (uint32_t)metadata[group * 4 + 3] << 24;
        int subtotal = 0;
        for (int section = 0; section < 4; ++section) {
            const uint8_t *grid0 = (const uint8_t *)(iq3xxs_grid +
                indices[group * 8 + section * 2]);
            const uint8_t *grid1 = (const uint8_t *)(iq3xxs_grid +
                indices[group * 8 + section * 2 + 1]);
            const uint8_t signs = ksigns_iq2xs[
                (packed >> (7 * section)) & 127u];
            for (int lane = 0; lane < 4; ++lane) {
                const int sign0 = signs & (1u << lane) ? -1 : 1;
                const int sign1 = signs & (1u << (lane + 4)) ? -1 : 1;
                subtotal += input->quants[group * 32 + section * 8 + lane] *
                            grid0[lane] * sign0;
                subtotal += input->quants[
                    group * 32 + section * 8 + lane + 4] * grid1[lane] *
                    sign1;
            }
        }
        weighted += (1 + 2 * (int)(packed >> 28)) * subtotal;
    }
    return d * weighted;
}

static float reference_iq3_s_q8_k(const uint8_t data[110],
                                   const Q38Q8KBlock *input)
{
    const float d = q38_f16_to_f32(
        (uint16_t)data[0] | (uint16_t)data[1] << 8) * input->scale;
    const uint8_t *indices = data + 2;
    const uint8_t *high = data + 66;
    const uint8_t *signs = data + 74;
    const uint8_t *scales = data + 106;
    int weighted = 0;
    for (int group = 0; group < 8; ++group) {
        int subtotal = 0;
        for (int section = 0; section < 4; ++section) {
            const uint16_t index0 =
                (uint16_t)indices[group * 8 + section * 2] |
                (uint16_t)((high[group] << (8 - 2 * section)) & 0x100);
            const uint16_t index1 =
                (uint16_t)indices[group * 8 + section * 2 + 1] |
                (uint16_t)((high[group] << (7 - 2 * section)) & 0x100);
            const uint8_t *grid0 = (const uint8_t *)(iq3s_grid + index0);
            const uint8_t *grid1 = (const uint8_t *)(iq3s_grid + index1);
            for (int lane = 0; lane < 4; ++lane) {
                const int sign0 = signs[group * 4 + section] &
                    (1u << lane) ? -1 : 1;
                const int sign1 = signs[group * 4 + section] &
                    (1u << (lane + 4)) ? -1 : 1;
                subtotal += input->quants[
                    group * 32 + section * 8 + lane] * grid0[lane] * sign0;
                subtotal += input->quants[
                    group * 32 + section * 8 + lane + 4] *
                    grid1[lane] * sign1;
            }
        }
        const int scale = 1 + 2 * ((scales[group / 2] >>
            (4 * (group % 2))) & 15u);
        weighted += scale * subtotal;
    }
    return d * weighted;
}

static float reference_iq4_xs_q8_k(const uint8_t data[136],
                                    const Q38Q8KBlock *input)
{
    const float d = q38_f16_to_f32(
        (uint16_t)data[0] | (uint16_t)data[1] << 8) * input->scale;
    const uint16_t scale_high = (uint16_t)data[2] | (uint16_t)data[3] << 8;
    const uint8_t *scale_low = data + 4;
    const uint8_t *quants = data + 8;
    int weighted = 0;
    for (int group = 0; group < 8; ++group) {
        const int scale = ((scale_low[group / 2] >>
            (4 * (group % 2))) & 15u) |
            (((scale_high >> (2 * group)) & 3u) << 4);
        int subtotal = 0;
        for (int lane = 0; lane < 16; ++lane) {
            const uint8_t packed = quants[group * 16 + lane];
            subtotal += input->quants[group * 32 + lane] *
                        kvalues_iq4nl[packed & 15u];
            subtotal += input->quants[group * 32 + lane + 16] *
                        kvalues_iq4nl[packed >> 4];
        }
        weighted += (scale - 32) * subtotal;
    }
    return d * weighted;
}

static float reference_iq4_nl_q8_k(const uint8_t data[144],
                                    const Q38Q8KBlock *input)
{
    float total = 0.0f;
    for (int group = 0; group < 8; ++group) {
        const uint8_t *block = data + group * 18;
        const float d = q38_f16_to_f32(
            (uint16_t)block[0] | (uint16_t)block[1] << 8) * input->scale;
        int weighted = 0;
        for (int lane = 0; lane < 16; ++lane) {
            weighted += input->quants[group * 32 + lane] *
                        kvalues_iq4nl[block[2 + lane] & 15u];
            weighted += input->quants[group * 32 + lane + 16] *
                        kvalues_iq4nl[block[2 + lane] >> 4];
        }
        total += d * weighted;
    }
    return total;
}

static float reference_iq1_m_q8_k(const uint8_t data[56],
                                   const Q38Q8KBlock *input)
{
    const uint8_t *indices = data;
    const uint8_t *high = data + 32;
    const uint8_t *scales = data + 48;
    const uint16_t scale_bits = ((uint16_t)scales[1] << 8 |
        scales[0]) >> 12 | (((uint16_t)scales[3] << 8 |
        scales[2]) >> 8 & 0x00f0u) | (((uint16_t)scales[5] << 8 |
        scales[4]) >> 4 & 0x0f00u) | (((uint16_t)scales[7] << 8 |
        scales[6]) & 0xf000u);
    const float d = q38_f16_to_f32(scale_bits) * input->scale;
    int weighted = 0;
    int correction = 0;
    for (int group = 0; group < 8; ++group) {
        int dot[2] = {0, 0};
        int correction_dot[2] = {0, 0};
        for (int section = 0; section < 4; ++section) {
            const uint16_t grid_index = indices[section] |
                (uint16_t)(((uint16_t)high[section / 2] <<
                (8 - 4 * (section % 2))) & 0x700u);
            const int8_t *grid = (const int8_t *)(iq1s_grid + grid_index);
            const int delta = high[section / 2] &
                (section % 2 ? 0x80u : 0x08u) ? -1 : 1;
            for (int lane = 0; lane < 8; ++lane) {
                const int activation = input->quants[
                    group * 32 + section * 8 + lane];
                dot[section / 2] += activation * grid[lane];
                correction_dot[section / 2] += activation * delta;
            }
        }
        const uint16_t packed_scale = (uint16_t)scales[(group / 2) * 2] |
            (uint16_t)scales[(group / 2) * 2 + 1] << 8;
        const int shift = 6 * (group % 2);
        const int scale0 = 1 + 2 * ((packed_scale >> shift) & 7u);
        const int scale1 = 1 + 2 * ((packed_scale >> (shift + 3)) & 7u);
        weighted += scale0 * dot[0] + scale1 * dot[1];
        correction += scale0 * correction_dot[0] +
                      scale1 * correction_dot[1];
        indices += 4;
        high += 2;
    }
    return d * (weighted + 0.125f * correction);
}

static float reference_iq1_s_q8_k(const uint8_t data[50],
                                   const Q38Q8KBlock *input)
{
    const float d = q38_f16_to_f32((uint16_t)data[0] |
        (uint16_t)data[1] << 8) * input->scale;
    const uint8_t *indices = data + 2;
    const uint8_t *high = data + 34;
    int weighted = 0;
    int correction = 0;
    for (int group = 0; group < 8; ++group) {
        const uint16_t packed = (uint16_t)high[group * 2] |
            (uint16_t)high[group * 2 + 1] << 8;
        const int scale = 1 + 2 * ((packed >> 12) & 7u);
        const int delta = packed & 0x8000u ? -1 : 1;
        int subtotal = 0;
        for (int section = 0; section < 4; ++section) {
            const uint16_t grid_index = indices[section] |
                (uint16_t)(((packed >> (3 * section)) & 7u) << 8);
            const int8_t *grid = (const int8_t *)(iq1s_grid + grid_index);
            for (int lane = 0; lane < 8; ++lane) {
                subtotal += input->quants[
                    group * 32 + section * 8 + lane] * grid[lane];
            }
        }
        weighted += scale * subtotal;
        correction += scale * delta *
            (input->sums[group * 2] + input->sums[group * 2 + 1]);
        indices += 4;
    }
    return d * (weighted + 0.125f * correction);
}

typedef float (*IQReference)(const uint8_t *, const Q38Q8KBlock *);

static int test_iq_kernel(uint32_t type, uint64_t bytes, IQReference reference)
{
    uint8_t data[144] = {0};
    float activation[256];
    uint32_t random = 0xf012ca65u + type;
    Q38GGUFTensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.n_dims = 2;
    tensor.shape[0] = 256;
    tensor.shape[1] = 1;
    tensor.type = type;
    tensor.nbytes = bytes;
    tensor.data = data;
    for (int trial = 0; trial < 200; ++trial) {
        for (uint64_t i = 0; i < bytes; ++i)
            data[i] = (uint8_t)(next_random(&random) >> 24);
        set_random_block_scales(data, type, 0x34cdu, 0);
        for (int i = 0; i < 256; ++i)
            activation[i] =
                (float)((int)(next_random(&random) >> 20) - 2048) /
                (float)(89 + i % 13);
        Q38Q8KBlock quantized;
        CHECK(q38_quantize_q8_k(&quantized, activation, 256),
              "IQ activation quantization");
        float actual = 0.0f;
        CHECK(q38_tensor_gemv_q8_k(&actual, &quantized, 256, &tensor),
              "IQ integer GEMV dispatch");
        const float expected = reference(data, &quantized);
        const float tolerance = 1e-4f * fmaxf(1.0f, fabsf(expected));
        if (fabsf(actual - expected) > tolerance) {
            fprintf(stderr,
                    "FAIL: IQ type=%u trial=%d actual=%.9g expected=%.9g "
                    "difference=%.9g tolerance=%.9g\n",
                    type, trial, actual, expected, actual - expected,
                    tolerance);
            return 1;
        }
    }
    return 0;
}

static int test_iq2_s_kernel(void)
{
    uint8_t block[82] = {0};
    float activation[256];
    uint32_t random = 0x7c11a295u;
    for (uint64_t i = 0; i < sizeof(block); ++i)
        block[i] = (uint8_t)(next_random(&random) >> 24);
    set_half(block, 0x34cdu);
    for (int i = 0; i < 256; ++i)
        activation[i] = (float)((int)(next_random(&random) >> 20) - 2048) /
                        (float)(97 + i % 11);
    Q38Q8KBlock quantized;
    CHECK(q38_quantize_q8_k(&quantized, activation, 256),
          "IQ2_S activation quantization");

    Q38GGUFTensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.n_dims = 2;
    tensor.shape[0] = 256;
    tensor.shape[1] = 1;
    tensor.type = Q38_GGML_IQ2_S;
    tensor.nbytes = sizeof(block);
    tensor.data = block;
    float actual = 0.0f;
    CHECK(q38_tensor_gemv_q8_k(&actual, &quantized, 256, &tensor),
          "IQ2_S integer GEMV dispatch");
    CHECK(close_to(actual, reference_iq2_s_q8_k(block, &quantized)),
          "IQ2_S integer kernel agrees with independent reference");
    return 0;
}

static int test_iq2_s_known_block(void)
{
    uint8_t block[82] = {0};
    set_half(block, 0x3c00u);
    float activation[256];
    for (int i = 0; i < 256; ++i) activation[i] = 1.0f;
    Q38Q8KBlock quantized;
    CHECK(q38_quantize_q8_k(&quantized, activation, 256),
          "known IQ2_S activation quantization");
    Q38GGUFTensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.n_dims = 2;
    tensor.shape[0] = 256;
    tensor.shape[1] = 1;
    tensor.type = Q38_GGML_IQ2_S;
    tensor.nbytes = sizeof(block);
    tensor.data = block;
    float actual = 0.0f;
    CHECK(q38_tensor_gemv_q8_k(&actual, &quantized, 256, &tensor),
          "known IQ2_S integer GEMV dispatch");
    CHECK(close_to(actual, 256.0f), "known IQ2_S value");
    return 0;
}

static int test_iq_known_block(uint32_t type, uint64_t bytes, float expected)
{
    uint8_t data[144] = {0};
    set_random_block_scales(data, type, 0x3c00u, 0);
    float activation[256];
    for (int i = 0; i < 256; ++i) activation[i] = 1.0f;
    Q38Q8KBlock quantized;
    CHECK(q38_quantize_q8_k(&quantized, activation, 256),
          "known IQ activation quantization");
    Q38GGUFTensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.n_dims = 2;
    tensor.shape[0] = 256;
    tensor.shape[1] = 1;
    tensor.type = type;
    tensor.nbytes = bytes;
    tensor.data = data;
    float actual = 0.0f;
    CHECK(q38_tensor_gemv_q8_k(&actual, &quantized, 256, &tensor),
          "known IQ integer GEMV dispatch");
    CHECK(close_to(actual, expected), "known IQ value");
    return 0;
}

static float reference_q2_k_dot(const uint8_t block[84],
                                const float input[256])
{
    const float d = q38_f16_to_f32((uint16_t)block[80] |
                                    (uint16_t)block[81] << 8);
    const float dmin = q38_f16_to_f32((uint16_t)block[82] |
                                      (uint16_t)block[83] << 8);
    float total = 0.0f;
    int scale_index = 0;
    const uint8_t *quants = block + 16;
    for (int half = 0; half < 2; ++half) {
        int shift = 0;
        for (int field = 0; field < 4; ++field) {
            for (int group = 0; group < 2; ++group) {
                const uint8_t packed_scale = block[scale_index++];
                const float scale = d * (packed_scale & 15u);
                const float minimum = dmin * (packed_scale >> 4);
                for (int lane = 0; lane < 16; ++lane) {
                    const int packed_index = half * 32 + group * 16 + lane;
                    const int input_index = half * 128 + field * 32 +
                                            group * 16 + lane;
                    const int quant = (quants[packed_index] >> shift) & 3;
                    total = fmaf(scale * quant - minimum,
                                 input[input_index], total);
                }
            }
            shift += 2;
        }
    }
    return total;
}

static float reference_q3_k_dot(const uint8_t block[110],
                                const float input[256])
{
    const uint32_t mask2 = 0x03030303u;
    const uint32_t mask4 = 0x0f0f0f0fu;
    uint32_t packed_scales[4] = {0};
    memcpy(packed_scales, block + 96, 12);
    const uint32_t high = packed_scales[2];
    packed_scales[2] = ((packed_scales[0] >> 4) & mask4) |
                       (((high >> 4) & mask2) << 4);
    packed_scales[3] = ((packed_scales[1] >> 4) & mask4) |
                       (((high >> 6) & mask2) << 4);
    packed_scales[0] = (packed_scales[0] & mask4) |
                       (((high >> 0) & mask2) << 4);
    packed_scales[1] = (packed_scales[1] & mask4) |
                       (((high >> 2) & mask2) << 4);
    const int8_t *scales = (const int8_t *)packed_scales;
    const float d = q38_f16_to_f32((uint16_t)block[108] |
                                    (uint16_t)block[109] << 8);
    const uint8_t *mask = block;
    const uint8_t *quants = block + 32;
    float total = 0.0f;
    int scale_index = 0;
    uint8_t mask_bit = 1;
    for (int half = 0; half < 2; ++half) {
        int shift = 0;
        for (int field = 0; field < 4; ++field) {
            for (int group = 0; group < 2; ++group) {
                const float scale = d * (scales[scale_index++] - 32);
                for (int lane = 0; lane < 16; ++lane) {
                    const int packed_index = half * 32 + group * 16 + lane;
                    const int input_index = half * 128 + field * 32 +
                                            group * 16 + lane;
                    const int quant = ((quants[packed_index] >> shift) & 3) -
                        ((mask[group * 16 + lane] & mask_bit) ? 0 : 4);
                    total = fmaf(scale * quant, input[input_index], total);
                }
            }
            shift += 2;
            mask_bit <<= 1;
        }
    }
    return total;
}

static int test_low_bit_decode_against_reference(uint32_t type,
                                                 uint64_t bytes)
{
    uint8_t block[110] = {0};
    float input[256];
    float actual = 0.0f;
    uint32_t random = 0xe20a87b5u + type;
    for (uint64_t i = 0; i < bytes; ++i)
        block[i] = (uint8_t)(next_random(&random) >> 24);
    set_random_block_scales(block, type, 0x34cdu, 0x30cdu);
    for (int i = 0; i < 256; ++i)
        input[i] = (float)((int)(next_random(&random) >> 20) - 2048) / 131.0f;

    Q38GGUFTensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.n_dims = 2;
    tensor.shape[0] = 256;
    tensor.shape[1] = 1;
    tensor.type = type;
    tensor.nbytes = bytes;
    tensor.data = block;
    CHECK(q38_tensor_gemv_f32(&actual, input, &tensor),
          "low-bit decoded GEMV dispatch");
    const float expected = type == Q38_GGML_Q2_K
        ? reference_q2_k_dot(block, input)
        : reference_q3_k_dot(block, input);
    CHECK(close_to(actual, expected),
          "low-bit decode agrees with independent reference");
    return 0;
}

static int test_integer_kernel_against_decoded(uint32_t type, uint64_t bytes)
{
    uint8_t block[210] = {0};
    float input[256];
    float reconstructed[256];
    float decoded = 0.0f;
    float integer = 0.0f;
    uint32_t random = 0x38c99a4du + type;
    for (uint64_t i = 0; i < bytes; ++i)
        block[i] = (uint8_t)(next_random(&random) >> 24);

    set_random_block_scales(block, type, 0x2e66u, 0x28cdu);
    for (int i = 0; i < 256; ++i)
        input[i] = (float)((int)(next_random(&random) >> 20) - 2048) / 257.0f;

    Q38Q8KBlock q8k;
    CHECK(q38_quantize_q8_k(&q8k, input, 256),
          "nonuniform Q8_K activation quantization");
    for (int i = 0; i < 256; ++i)
        reconstructed[i] = q8k.scale * q8k.quants[i];

    Q38GGUFTensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.n_dims = 2;
    tensor.shape[0] = 256;
    tensor.shape[1] = 1;
    tensor.type = type;
    tensor.nbytes = bytes;
    tensor.data = block;
    CHECK(q38_tensor_gemv_f32(&decoded, reconstructed, &tensor),
          "decoded nonuniform GEMV");
    CHECK(q38_tensor_gemv_q8_k(&integer, &q8k, 256, &tensor),
          "integer nonuniform GEMV");
    CHECK(fabsf(decoded - integer) <= 2e-4f * fmaxf(1.0f, fabsf(decoded)),
          "integer kernel agrees with decoded nonuniform weights");
    return 0;
}

static int test_integer_batch_exact(uint32_t type, uint64_t block_bytes)
{
    enum { WIDTH = 512, ROWS = 3, MAX_TOKENS = 5 };
    uint8_t weights[3 * 2 * 210] = {0};
    float activations[MAX_TOKENS * WIDTH];
    Q38Q8KBlock quantized[MAX_TOKENS * 2];
    float batch[MAX_TOKENS * ROWS];
    float single[ROWS];
    uint32_t random = 0x91e10da5u + type;
    const uint64_t bytes = ROWS * 2u * block_bytes;
    for (uint64_t i = 0; i < bytes; ++i)
        weights[i] = (uint8_t)(next_random(&random) >> 24);
    for (int row = 0; row < ROWS; ++row) {
        for (int block = 0; block < 2; ++block) {
            uint8_t *data = weights + (row * 2 + block) * block_bytes;
            set_random_block_scales(
                data, type, (uint16_t)(0x2800u + row * 0x80u + block),
                (uint16_t)(0x2400u + row * 0x40u + block));
        }
    }
    for (int i = 0; i < MAX_TOKENS * WIDTH; ++i)
        activations[i] = (float)((int)(next_random(&random) >> 19) - 4096) /
                         (float)(101 + (i % 7));
    CHECK(q38_quantize_q8_k(quantized, activations, MAX_TOKENS * WIDTH),
          "batched exact Q8_K activation quantization");

    Q38GGUFTensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.n_dims = 2;
    tensor.shape[0] = WIDTH;
    tensor.shape[1] = ROWS;
    tensor.type = type;
    tensor.nbytes = bytes;
    tensor.data = weights;
    for (int token_count = 1; token_count <= MAX_TOKENS; ++token_count) {
        CHECK(q38_tensor_gemm_q8_k(batch, quantized, (uint32_t)token_count,
                                    WIDTH, &tensor),
              "integer tiled GEMM dispatch");
        for (int token = 0; token < token_count; ++token) {
            CHECK(q38_tensor_gemv_q8_k(single, quantized + token * 2,
                                        WIDTH, &tensor),
                  "integer reference GEMV dispatch");
            CHECK(memcmp(batch + token * ROWS, single, sizeof(single)) == 0,
                  "integer tiled GEMM is bit-exact with GEMV");
        }
    }
    return 0;
}

static int test_iq1_s_repack_exact(void)
{
    enum { WIDTH = 512, ROWS = 3, BLOCK_BYTES = 50 };
    uint8_t weights[ROWS * 2 * BLOCK_BYTES];
    float activation[WIDTH];
    Q38Q8KBlock quantized[2];
    float packed_output[ROWS];
    float repacked_output[ROWS];
    uint32_t random = 0x7638a11du;
    for (size_t i = 0; i < sizeof(weights); ++i)
        weights[i] = (uint8_t)(next_random(&random) >> 24);
    for (int row = 0; row < ROWS; ++row) {
        for (int block = 0; block < 2; ++block) {
            set_random_block_scales(
                weights + (row * 2 + block) * BLOCK_BYTES,
                Q38_GGML_IQ1_S,
                (uint16_t)(0x2a00u + row * 0x40u + block), 0);
        }
    }
    for (int i = 0; i < WIDTH; ++i)
        activation[i] = (float)((int)(next_random(&random) >> 20) - 2048) /
                        (float)(83 + i % 11);
    CHECK(q38_quantize_q8_k(quantized, activation, WIDTH),
          "IQ1_S repack activation quantization");

    Q38GGUFTensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.n_dims = 2;
    tensor.shape[0] = WIDTH;
    tensor.shape[1] = ROWS;
    tensor.type = Q38_GGML_IQ1_S;
    tensor.nbytes = sizeof(weights);
    tensor.data = weights;
    CHECK(q38_tensor_gemv_q8_k(packed_output, quantized, WIDTH, &tensor),
          "packed IQ1_S GEMV");

    Q38GGUF gguf;
    memset(&gguf, 0, sizeof(gguf));
    gguf.tensor_count = 1;
    gguf.tensors = &tensor;
    CHECK(q38_prepare_iq1_s_repacks(&gguf), "prepare IQ1_S runtime repack");
    CHECK(q38_tensor_gemv_q8_k(repacked_output, quantized, WIDTH, &tensor),
          "repacked IQ1_S GEMV");
    CHECK(memcmp(packed_output, repacked_output, sizeof(packed_output)) == 0,
          "repacked IQ1_S GEMV is bit-exact");
    q38_release_iq1_s_repacks(&gguf);
    CHECK(tensor.iq1_s_repack == NULL, "release IQ1_S runtime repack");
    return 0;
}

static int test_block(uint8_t *block, uint32_t type, uint64_t bytes,
                      float expected)
{
    float input[256], output = 0.0f;
    for (int i = 0; i < 256; ++i) input[i] = 1.0f;
    Q38GGUFTensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.n_dims = 2;
    tensor.shape[0] = type == Q38_GGML_Q8_0 ? 32 : 256;
    tensor.shape[1] = 1;
    tensor.type = type;
    tensor.nbytes = bytes;
    tensor.data = block;
    CHECK(q38_tensor_gemv_f32(&output, input, &tensor), "quantized GEMV dispatch");
    CHECK(close_to(output, expected), "quantized GEMV value");
    if (type == Q38_GGML_Q2_K || type == Q38_GGML_Q3_K ||
        type == Q38_GGML_Q4_K || type == Q38_GGML_Q5_K ||
        type == Q38_GGML_Q6_K) {
        Q38Q8KBlock quantized;
        float integer_output = 0.0f;
        CHECK(q38_quantize_q8_k(&quantized, input, 256),
              "Q8_K activation quantization");
        CHECK(q38_tensor_gemv_q8_k(&integer_output, &quantized, 256, &tensor),
              "integer GEMV dispatch");
        CHECK(close_to(integer_output, expected), "integer GEMV value");
    }
    return 0;
}

int main(void)
{
    CHECK(q38_f16_to_f32(0x3c00u) == 1.0f, "f16 one");
    CHECK(q38_f16_to_f32(0xc000u) == -2.0f, "f16 negative two");
    CHECK(q38_f16_to_f32(0x0001u) == 0x1p-24f, "f16 subnormal");

    float activation[256];
    for (int i = 0; i < 256; ++i) activation[i] = (float)(i - 128) / 128.0f;
    Q38Q8KBlock q8k;
    CHECK(q38_quantize_q8_k(&q8k, activation, 256), "Q8_K block");
    CHECK(q8k.quants[0] == -127 && q8k.quants[128] == 0,
          "Q8_K signed range");

    uint8_t q2[84] = {0};
    memset(q2, 0x01, 16);
    memset(q2 + 16, 0x55, 64);
    set_half(q2 + 80, 0x3c00u);
    CHECK(test_block(q2, Q38_GGML_Q2_K, sizeof(q2), 256.0f) == 0,
          "Q2_K block");

    uint8_t q3[110] = {0};
    memset(q3, 0xff, 32);
    memset(q3 + 32, 0x55, 64);
    memset(q3 + 96, 0x11, 8);
    memset(q3 + 104, 0xaa, 4);
    set_half(q3 + 108, 0x3c00u);
    CHECK(test_block(q3, Q38_GGML_Q3_K, sizeof(q3), 256.0f) == 0,
          "Q3_K block");

    uint8_t q4[144] = {0};
    set_half(q4, 0x3c00u);
    set_half(q4 + 2, 0x3800u);
    for (int i = 0; i < 4; ++i) q4[4 + i] = 1;
    for (int i = 0; i < 4; ++i) q4[8 + i] = 2;
    memset(q4 + 16, 0x21, 128);
    CHECK(test_block(q4, Q38_GGML_Q4_K, sizeof(q4), 64.0f) == 0, "Q4_K block");

    uint8_t q4_matrix[288];
    memcpy(q4_matrix, q4, sizeof(q4));
    memcpy(q4_matrix + sizeof(q4), q4, sizeof(q4));
    Q38GGUFTensor q4_tensor;
    memset(&q4_tensor, 0, sizeof(q4_tensor));
    q4_tensor.n_dims = 2;
    q4_tensor.shape[0] = 256;
    q4_tensor.shape[1] = 2;
    q4_tensor.type = Q38_GGML_Q4_K;
    q4_tensor.nbytes = sizeof(q4_matrix);
    q4_tensor.data = q4_matrix;
    float q8_inputs[512];
    for (int i = 0; i < 256; ++i) {
        q8_inputs[i] = 1.0f;
        q8_inputs[256 + i] = 2.0f;
    }
    Q38Q8KBlock q8_batches[2];
    float q4_batch_output[4] = {0};
    CHECK(q38_quantize_q8_k(q8_batches, q8_inputs, 512),
          "batched Q8_K activation quantization");
    CHECK(q38_tensor_gemm_q8_k(q4_batch_output, q8_batches, 2, 256,
                                &q4_tensor),
          "Q4_K integer GEMM dispatch");
    CHECK(close_to(q4_batch_output[0], 64.0f) &&
          close_to(q4_batch_output[1], 64.0f) &&
          close_to(q4_batch_output[2], 128.0f) &&
          close_to(q4_batch_output[3], 128.0f),
          "Q4_K integer GEMM batch-major output");

    uint8_t q5[176] = {0};
    set_half(q5, 0x3c00u);
    set_half(q5 + 2, 0x3800u);
    for (int i = 0; i < 4; ++i) q5[4 + i] = 1;
    for (int i = 0; i < 4; ++i) q5[8 + i] = 2;
    memset(q5 + 48, 0x21, 128);
    CHECK(test_block(q5, Q38_GGML_Q5_K, sizeof(q5), 64.0f) == 0, "Q5_K block");

    uint8_t q6[210] = {0};
    memset(q6 + 192, 1, 16);
    set_half(q6 + 208, 0x3c00u);
    CHECK(test_block(q6, Q38_GGML_Q6_K, sizeof(q6), -8192.0f) == 0, "Q6_K block");

    uint8_t q8[34] = {0};
    set_half(q8, 0x3800u);
    memset(q8 + 2, 2, 32);
    CHECK(test_block(q8, Q38_GGML_Q8_0, sizeof(q8), 32.0f) == 0, "Q8_0 block");

    CHECK(test_integer_kernel_against_decoded(Q38_GGML_Q2_K, 84) == 0,
          "Q2_K nonuniform block");
    CHECK(test_integer_kernel_against_decoded(Q38_GGML_Q3_K, 110) == 0,
          "Q3_K nonuniform block");
    CHECK(test_low_bit_decode_against_reference(Q38_GGML_Q2_K, 84) == 0,
          "Q2_K independent decode reference");
    CHECK(test_low_bit_decode_against_reference(Q38_GGML_Q3_K, 110) == 0,
          "Q3_K independent decode reference");
    CHECK(test_integer_kernel_against_decoded(Q38_GGML_Q4_K, 144) == 0,
          "Q4_K nonuniform block");
    CHECK(test_integer_kernel_against_decoded(Q38_GGML_Q5_K, 176) == 0,
          "Q5_K nonuniform block");
    CHECK(test_integer_kernel_against_decoded(Q38_GGML_Q6_K, 210) == 0,
          "Q6_K nonuniform block");
    CHECK(test_iq2_s_known_block() == 0, "IQ2_S known block");
    CHECK(test_iq2_s_kernel() == 0, "IQ2_S nonuniform block");
    CHECK(test_iq_kernel(Q38_GGML_IQ2_S, 82,
                         reference_iq2_s_q8_k) == 0,
          "IQ2_S randomized blocks");
    CHECK(test_iq_known_block(Q38_GGML_IQ2_XXS, 66, 256.0f) == 0,
          "IQ2_XXS known block");
    CHECK(test_iq_known_block(Q38_GGML_IQ2_XS, 74, 256.0f) == 0,
          "IQ2_XS known block");
    CHECK(test_iq_known_block(Q38_GGML_IQ3_XXS, 98, 256.0f) == 0,
          "IQ3_XXS known block");
    CHECK(test_iq_known_block(Q38_GGML_IQ3_S, 110, 256.0f) == 0,
          "IQ3_S known block");
    CHECK(test_iq_known_block(Q38_GGML_IQ4_XS, 136, 1040384.0f) == 0,
          "IQ4_XS known block");
    CHECK(test_iq_known_block(Q38_GGML_IQ4_NL, 144, -32512.0f) == 0,
          "IQ4_NL known block");
    CHECK(test_iq_known_block(Q38_GGML_IQ1_S, 50, -224.0f) == 0,
          "IQ1_S known block");
    CHECK(test_iq_known_block(Q38_GGML_IQ1_M, 56, -224.0f) == 0,
          "IQ1_M known block");
    CHECK(test_iq_kernel(Q38_GGML_IQ2_XXS, 66,
                         reference_iq2_xxs_q8_k) == 0,
          "IQ2_XXS nonuniform block");
    CHECK(test_iq_kernel(Q38_GGML_IQ2_XS, 74,
                         reference_iq2_xs_q8_k) == 0,
          "IQ2_XS nonuniform block");
    CHECK(test_iq_kernel(Q38_GGML_IQ3_XXS, 98,
                         reference_iq3_xxs_q8_k) == 0,
          "IQ3_XXS nonuniform block");
    CHECK(test_iq_kernel(Q38_GGML_IQ3_S, 110,
                         reference_iq3_s_q8_k) == 0,
          "IQ3_S nonuniform block");
    CHECK(test_iq_kernel(Q38_GGML_IQ4_XS, 136,
                         reference_iq4_xs_q8_k) == 0,
          "IQ4_XS nonuniform block");
    CHECK(test_iq_kernel(Q38_GGML_IQ4_NL, 144,
                         reference_iq4_nl_q8_k) == 0,
          "IQ4_NL nonuniform block");
    CHECK(test_iq_kernel(Q38_GGML_IQ1_S, 50,
                         reference_iq1_s_q8_k) == 0,
          "IQ1_S nonuniform block");
    CHECK(test_iq_kernel(Q38_GGML_IQ1_M, 56,
                         reference_iq1_m_q8_k) == 0,
          "IQ1_M nonuniform block");
    CHECK(test_integer_batch_exact(Q38_GGML_Q2_K, 84) == 0,
          "Q2_K tiled integer batch");
    CHECK(test_integer_batch_exact(Q38_GGML_Q3_K, 110) == 0,
          "Q3_K tiled integer batch");
    CHECK(test_integer_batch_exact(Q38_GGML_Q4_K, 144) == 0,
          "Q4_K tiled integer batch");
    CHECK(test_integer_batch_exact(Q38_GGML_Q5_K, 176) == 0,
          "Q5_K tiled integer batch");
    CHECK(test_integer_batch_exact(Q38_GGML_Q6_K, 210) == 0,
          "Q6_K tiled integer batch");
    CHECK(test_integer_batch_exact(Q38_GGML_IQ2_S, 82) == 0,
          "IQ2_S tiled integer batch");
    CHECK(test_integer_batch_exact(Q38_GGML_IQ2_XXS, 66) == 0,
          "IQ2_XXS tiled integer batch");
    CHECK(test_integer_batch_exact(Q38_GGML_IQ2_XS, 74) == 0,
          "IQ2_XS tiled integer batch");
    CHECK(test_integer_batch_exact(Q38_GGML_IQ3_XXS, 98) == 0,
          "IQ3_XXS tiled integer batch");
    CHECK(test_integer_batch_exact(Q38_GGML_IQ3_S, 110) == 0,
          "IQ3_S tiled integer batch");
    CHECK(test_integer_batch_exact(Q38_GGML_IQ4_XS, 136) == 0,
          "IQ4_XS tiled integer batch");
    CHECK(test_integer_batch_exact(Q38_GGML_IQ4_NL, 144) == 0,
          "IQ4_NL tiled integer batch");
    CHECK(test_integer_batch_exact(Q38_GGML_IQ1_S, 50) == 0,
          "IQ1_S tiled integer batch");
    CHECK(test_integer_batch_exact(Q38_GGML_IQ1_M, 56) == 0,
          "IQ1_M tiled integer batch");
    CHECK(test_iq1_s_repack_exact() == 0, "IQ1_S runtime repack");

    float matrix[8] = {1, 2, 3, 4, -1, -2, -3, -4};
    float input[4] = {1, 1, 1, 1};
    float output[2] = {0};
    Q38GGUFTensor f32;
    memset(&f32, 0, sizeof(f32));
    f32.n_dims = 2;
    f32.shape[0] = 4;
    f32.shape[1] = 2;
    f32.type = Q38_GGML_F32;
    f32.data = (const uint8_t *)matrix;
    CHECK(q38_tensor_gemv_f32(output, input, &f32), "F32 GEMV dispatch");
    CHECK(output[0] == 10.0f && output[1] == -10.0f, "F32 GEMV value");

    float batch_input[8] = {1, 1, 1, 1, 2, 2, 2, 2};
    float batch_output[4] = {0};
    CHECK(q38_tensor_gemm_f32(batch_output, batch_input, 2, 4, &f32),
          "F32 GEMM dispatch");
    CHECK(batch_output[0] == 10.0f && batch_output[1] == -10.0f &&
          batch_output[2] == 20.0f && batch_output[3] == -20.0f,
          "F32 GEMM batch-major output");

    puts("qwen38 quant: ok");
    return 0;
}
