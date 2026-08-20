/* Packed K/IQ decoding techniques are adapted from ggml's MIT-licensed
 * reference kernels. Qwen-specific integration and multi-token kernels are
 * modified here. See NOTICE and third_party/GGML-LICENSE.txt.
 */
#include "qwen38_quant.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
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

#if defined(__AVX2__) || defined(__F16C__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#define Q38_UNUSED_HELPER __attribute__((unused))
#endif

static uint16_t q38_load_u16(const uint8_t *data)
{
    return (uint16_t)data[0] | (uint16_t)data[1] << 8;
}

static uint32_t q38_load_u32(const uint8_t *data)
{
    return (uint32_t)data[0] | (uint32_t)data[1] << 8 |
           (uint32_t)data[2] << 16 | (uint32_t)data[3] << 24;
}

static int q38_nearest_int(float value)
{
    const float shifted = value + 12582912.0f;
    uint32_t bits;
    memcpy(&bits, &shifted, sizeof(bits));
    return (int)(bits & 0x007fffffu) - 0x00400000;
}

int q38_quantize_q8_k(Q38Q8KBlock *output, const float *input,
                      uint64_t length)
{
    if (!output || !input || length % Q38_Q8_K_BLOCK_SIZE != 0) return 0;
    const uint64_t blocks = length / Q38_Q8_K_BLOCK_SIZE;
    for (uint64_t block = 0; block < blocks; ++block) {
        const float *values = input + block * Q38_Q8_K_BLOCK_SIZE;
        Q38Q8KBlock *quantized = output + block;
        float maximum = 0.0f;
        float absolute_maximum = 0.0f;
        for (int i = 0; i < Q38_Q8_K_BLOCK_SIZE; ++i) {
            const float absolute = fabsf(values[i]);
            if (absolute > absolute_maximum) {
                absolute_maximum = absolute;
                maximum = values[i];
            }
        }
        if (absolute_maximum == 0.0f) {
            quantized->scale = 0.0f;
            memset(quantized->quants, 0, sizeof(quantized->quants));
            memset(quantized->sums, 0, sizeof(quantized->sums));
            continue;
        }
        const float inverse_scale = -127.0f / maximum;
        for (int i = 0; i < Q38_Q8_K_BLOCK_SIZE; ++i) {
            int value = q38_nearest_int(inverse_scale * values[i]);
            if (value > 127) value = 127;
            quantized->quants[i] = (int8_t)value;
        }
        for (int group = 0; group < Q38_Q8_K_BLOCK_SIZE / 16; ++group) {
            int sum = 0;
            for (int i = 0; i < 16; ++i)
                sum += quantized->quants[group * 16 + i];
            quantized->sums[group] = (int16_t)sum;
        }
        quantized->scale = 1.0f / inverse_scale;
    }
    return 1;
}

float q38_f16_to_f32(uint16_t value)
{
#if defined(__F16C__)
    return _cvtsh_ss(value);
#else
    const uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 31u;
    const uint32_t mantissa = value & 1023u;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            uint32_t normalized = mantissa;
            int shift = 0;
            while ((normalized & 0x400u) == 0) {
                normalized <<= 1;
                ++shift;
            }
            normalized &= 0x3ffu;
            bits = sign | (uint32_t)(127 - 14 - shift) << 23 |
                   normalized << 13;
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | mantissa << 13;
    } else {
        bits = sign | (exponent + 112u) << 23 | mantissa << 13;
    }
    union { uint32_t u; float f; } result;
    result.u = bits;
    return result.f;
#endif
}

static float q38_bf16_to_f32_local(const uint8_t *data)
{
    union { uint32_t u; float f; } result;
    result.u = (uint32_t)q38_load_u16(data) << 16;
    return result.f;
}

static void q38_scale_min_k4(int index, const uint8_t *packed,
                             uint8_t *scale, uint8_t *minimum)
{
    if (index < 4) {
        *scale = packed[index] & 63u;
        *minimum = packed[index + 4] & 63u;
    } else {
        *scale = (packed[index + 4] & 15u) |
                 (uint8_t)((packed[index - 4] >> 6) << 4);
        *minimum = (packed[index + 4] >> 4) |
                   (uint8_t)((packed[index] >> 6) << 4);
    }
}

#if defined(__AVX2__)
static __m256i q38_products_u8_s8_32(__m256i unsigned_values,
                                     __m256i signed_values)
{
#if defined(__AVXVNNI__)
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), unsigned_values,
                               signed_values);
#else
    const __m256i pair16 = _mm256_maddubs_epi16(unsigned_values, signed_values);
    return _mm256_madd_epi16(pair16, _mm256_set1_epi16(1));
#endif
}

static int Q38_UNUSED_HELPER q38_sum_i32_8(__m256i sum32)
{
    int32_t lanes[8];
    _mm256_storeu_si256((__m256i *)lanes, sum32);
    int total = 0;
    for (int i = 0; i < 8; ++i) total += lanes[i];
    return total;
}

static __m128i Q38_UNUSED_HELPER q38_products_s8_s8_16(__m128i left,
                                                       __m128i right)
{
    const __m128i magnitudes = _mm_abs_epi8(left);
    const __m128i signed_right = _mm_sign_epi8(right, left);
    const __m128i pair16 = _mm_maddubs_epi16(magnitudes, signed_right);
    return _mm_madd_epi16(pair16, _mm_set1_epi16(1));
}

static __m256i q38_products_s8_s8_32(__m256i left, __m256i right)
{
    const __m256i magnitudes = _mm256_abs_epi8(left);
    const __m256i signed_right = _mm256_sign_epi8(right, left);
    return q38_products_u8_s8_32(magnitudes, signed_right);
}

static __m256i q38_products_s8_s8_scaled_32(__m256i left, __m256i right,
                                             int scale)
{
    const __m256i magnitudes = _mm256_abs_epi8(left);
    const __m256i signed_right = _mm256_sign_epi8(right, left);
#if defined(__AVXVNNI__)
    const __m256i dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(),
                                             magnitudes, signed_right);
    return _mm256_mullo_epi32(dot, _mm256_set1_epi32(scale));
#else
    const __m256i pair16 = _mm256_maddubs_epi16(magnitudes, signed_right);
    return _mm256_madd_epi16(pair16, _mm256_set1_epi16((short)scale));
#endif
}

static __m256i q38_iq1_s_grid32(const uint8_t indices[4], uint16_t high)
{
#if defined(__BMI2__)
    const uint64_t packed =
        _pdep_u64(q38_load_u32(indices), 0x00ff00ff00ff00ffULL) |
        _pdep_u64(high, 0x0700070007000700ULL);
    return _mm256_set_epi64x(
        (long long)iq1s_grid[(uint16_t)(packed >> 48)],
        (long long)iq1s_grid[(uint16_t)(packed >> 32)],
        (long long)iq1s_grid[(uint16_t)(packed >> 16)],
        (long long)iq1s_grid[(uint16_t)packed]);
#else
    return _mm256_set_epi64x(
        (long long)iq1s_grid[indices[3] | ((high >> 1) & 0x700u)],
        (long long)iq1s_grid[indices[2] | ((high << 2) & 0x700u)],
        (long long)iq1s_grid[indices[1] | ((high << 5) & 0x700u)],
        (long long)iq1s_grid[indices[0] | ((high << 8) & 0x700u)]);
#endif
}

static __m256i q38_iq_sign_mask(uint32_t sign0, uint32_t sign1,
                                uint32_t sign2, uint32_t sign3)
{
    return _mm256_set_epi64x(
        (long long)ksigns64[sign3], (long long)ksigns64[sign2],
        (long long)ksigns64[sign1], (long long)ksigns64[sign0]);
}

static __m256i q38_apply_iq_signs(__m256i activation, uint32_t sign0,
                                  uint32_t sign1, uint32_t sign2,
                                  uint32_t sign3)
{
    const __m256i negative = q38_iq_sign_mask(sign0, sign1, sign2, sign3);
    return _mm256_sub_epi8(_mm256_xor_si256(negative, activation), negative);
}

static __m256i q38_raw_iq_sign_mask(const uint8_t signs[4])
{
    static const uint8_t sign_source[32] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3
    };
    static const uint8_t sign_bit[32] = {
        1, 2, 4, 8, 16, 32, 64, 128,
        1, 2, 4, 8, 16, 32, 64, 128,
        1, 2, 4, 8, 16, 32, 64, 128,
        1, 2, 4, 8, 16, 32, 64, 128
    };
    const __m256i packed = _mm256_set1_epi32((int)q38_load_u32(signs));
    const __m256i source = _mm256_loadu_si256(
        (const __m256i *)sign_source);
    const __m256i bits = _mm256_loadu_si256((const __m256i *)sign_bit);
    const __m256i selected = _mm256_and_si256(
        _mm256_shuffle_epi8(packed, source), bits);
    return _mm256_cmpeq_epi8(selected, bits);
}

static __m256i q38_apply_raw_iq_signs(__m256i activation,
                                      const uint8_t signs[4])
{
    const __m256i negative = q38_raw_iq_sign_mask(signs);
    return _mm256_sub_epi8(_mm256_xor_si256(negative, activation), negative);
}

static inline __attribute__((always_inline))
void q38_iq3_s_grid_vectors(__m256i output[2],
                            const uint8_t codes[16],
                            const uint8_t high[2])
{
    const __m128i packed = _mm_loadu_si128((const __m128i *)codes);
    const __m256i shifts = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i high_mask = _mm256_set1_epi32(0x100);
    uint32_t indices[16];

    const __m256i low0 = _mm256_cvtepu8_epi32(packed);
    const __m256i low1 = _mm256_cvtepu8_epi32(_mm_srli_si128(packed, 8));
    const __m256i high0 = _mm256_and_si256(
        _mm256_sllv_epi32(_mm256_set1_epi32(high[0]), shifts), high_mask);
    const __m256i high1 = _mm256_and_si256(
        _mm256_sllv_epi32(_mm256_set1_epi32(high[1]), shifts), high_mask);
    _mm256_storeu_si256((__m256i *)indices, _mm256_or_si256(low0, high0));
    _mm256_storeu_si256((__m256i *)(indices + 8),
                        _mm256_or_si256(low1, high1));

    output[0] = _mm256_set_epi32(
        (int)iq3s_grid[indices[7]], (int)iq3s_grid[indices[6]],
        (int)iq3s_grid[indices[5]], (int)iq3s_grid[indices[4]],
        (int)iq3s_grid[indices[3]], (int)iq3s_grid[indices[2]],
        (int)iq3s_grid[indices[1]], (int)iq3s_grid[indices[0]]);
    output[1] = _mm256_set_epi32(
        (int)iq3s_grid[indices[15]], (int)iq3s_grid[indices[14]],
        (int)iq3s_grid[indices[13]], (int)iq3s_grid[indices[12]],
        (int)iq3s_grid[indices[11]], (int)iq3s_grid[indices[10]],
        (int)iq3s_grid[indices[9]], (int)iq3s_grid[indices[8]]);
}

static int Q38_UNUSED_HELPER q38_sum_i32_4(__m128i sum32)
{
    int32_t lanes[4];
    _mm_storeu_si128((__m128i *)lanes, sum32);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

static float q38_sum_f32_8(__m256 values)
{
    __m128 sum = _mm256_extractf128_ps(values, 1);
    sum = _mm_add_ps(sum, _mm256_castps256_ps128(values));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

static __m256i q38_q6_scale_pair(const int8_t *scales, int index)
{
    return _mm256_set_m128i(_mm_set1_epi16(scales[index + 1]),
                            _mm_set1_epi16(scales[index]));
}
#endif

static void q38_unpack_q3_scales(const uint8_t *packed, int8_t scales[16])
{
    for (int index = 0; index < 16; ++index) {
        const uint8_t low = index < 8
            ? packed[index] & 15u
            : packed[index - 8] >> 4;
        const uint8_t high = (packed[8 + index % 4] >>
                              (2 * (index / 4))) & 3u;
        scales[index] = (int8_t)((low | high << 4) - 32);
    }
}

static float q38_dot_q2_k(const uint8_t *blocks, const float *input,
                          uint64_t length)
{
    float total = 0.0f;
    for (uint64_t base = 0; base < length; base += 256, blocks += 84) {
        const uint8_t *scales = blocks;
        const uint8_t *quants = blocks + 16;
        const float d = q38_f16_to_f32(q38_load_u16(blocks + 80));
        const float dmin = q38_f16_to_f32(q38_load_u16(blocks + 82));
        int scale_index = 0;
        for (int half = 0; half < 2; ++half) {
            for (int field = 0; field < 4; ++field) {
                const int shift = field * 2;
                for (int group = 0; group < 2; ++group) {
                    const uint8_t packed_scale = scales[scale_index++];
                    const float scale = d * (packed_scale & 15u);
                    const float minimum = dmin * (packed_scale >> 4);
                    const uint64_t offset = base + (uint64_t)half * 128u +
                                            (uint64_t)field * 32u +
                                            (uint64_t)group * 16u;
                    const uint8_t *packed = quants + half * 32 + group * 16;
                    for (int lane = 0; lane < 16; ++lane) {
                        const int q = (packed[lane] >> shift) & 3;
                        total = fmaf(scale * q - minimum,
                                     input[offset + (uint64_t)lane], total);
                    }
                }
            }
        }
    }
    return total;
}

static float q38_dot_q3_k(const uint8_t *blocks, const float *input,
                          uint64_t length)
{
    float total = 0.0f;
    for (uint64_t base = 0; base < length; base += 256, blocks += 110) {
        const uint8_t *high = blocks;
        const uint8_t *low = blocks + 32;
        int8_t scales[16];
        q38_unpack_q3_scales(blocks + 96, scales);
        const float d = q38_f16_to_f32(q38_load_u16(blocks + 108));
        int group = 0;
        uint8_t high_bit = 1;
        for (int half = 0; half < 2; ++half) {
            for (int field = 0; field < 4; ++field, ++group) {
                const int shift = field * 2;
                const uint8_t *packed = low + half * 32;
                for (int lane = 0; lane < 32; ++lane) {
                    const int q = ((packed[lane] >> shift) & 3) -
                                  ((high[lane] & high_bit) ? 0 : 4);
                    const int scale = scales[group * 2 + lane / 16];
                    const uint64_t offset = base + (uint64_t)half * 128u +
                                            (uint64_t)field * 32u +
                                            (uint64_t)lane;
                    total = fmaf(d * scale * q, input[offset], total);
                }
                high_bit <<= 1;
            }
        }
    }
    return total;
}

static float q38_dot_q4_k(const uint8_t *blocks, const float *input,
                          uint64_t length)
{
    float total = 0.0f;
    for (uint64_t base = 0; base < length; base += 256, blocks += 144) {
        const float d = q38_f16_to_f32(q38_load_u16(blocks));
        const float dmin = q38_f16_to_f32(q38_load_u16(blocks + 2));
        const uint8_t *scales = blocks + 4;
        const uint8_t *quants = blocks + 16;
        int scale_index = 0;
        for (int chunk = 0; chunk < 4; ++chunk) {
            uint8_t scale0, min0, scale1, min1;
            q38_scale_min_k4(scale_index, scales, &scale0, &min0);
            q38_scale_min_k4(scale_index + 1, scales, &scale1, &min1);
            const float d0 = d * scale0;
            const float m0 = dmin * min0;
            const float d1 = d * scale1;
            const float m1 = dmin * min1;
            const float *x = input + base + (uint64_t)chunk * 64u;
            for (int i = 0; i < 32; ++i) {
                total = fmaf(d0 * (quants[i] & 15u) - m0, x[i], total);
            }
            for (int i = 0; i < 32; ++i) {
                total = fmaf(d1 * (quants[i] >> 4) - m1, x[i + 32], total);
            }
            quants += 32;
            scale_index += 2;
        }
    }
    return total;
}

static float q38_dot_q5_k(const uint8_t *blocks, const float *input,
                          uint64_t length)
{
    float total = 0.0f;
    for (uint64_t base = 0; base < length; base += 256, blocks += 176) {
        const float d = q38_f16_to_f32(q38_load_u16(blocks));
        const float dmin = q38_f16_to_f32(q38_load_u16(blocks + 2));
        const uint8_t *scales = blocks + 4;
        const uint8_t *high = blocks + 16;
        const uint8_t *low = blocks + 48;
        int scale_index = 0;
        uint8_t high0 = 1, high1 = 2;
        for (int chunk = 0; chunk < 4; ++chunk) {
            uint8_t scale0, min0, scale1, min1;
            q38_scale_min_k4(scale_index, scales, &scale0, &min0);
            q38_scale_min_k4(scale_index + 1, scales, &scale1, &min1);
            const float d0 = d * scale0;
            const float m0 = dmin * min0;
            const float d1 = d * scale1;
            const float m1 = dmin * min1;
            const float *x = input + base + (uint64_t)chunk * 64u;
            for (int i = 0; i < 32; ++i) {
                const int q = (low[i] & 15u) + ((high[i] & high0) ? 16 : 0);
                total = fmaf(d0 * q - m0, x[i], total);
            }
            for (int i = 0; i < 32; ++i) {
                const int q = (low[i] >> 4) + ((high[i] & high1) ? 16 : 0);
                total = fmaf(d1 * q - m1, x[i + 32], total);
            }
            low += 32;
            scale_index += 2;
            high0 <<= 2;
            high1 <<= 2;
        }
    }
    return total;
}

static float q38_dot_q6_k(const uint8_t *blocks, const float *input,
                          uint64_t length)
{
    float total = 0.0f;
    for (uint64_t base = 0; base < length; base += 256, blocks += 210) {
        const uint8_t *low = blocks;
        const uint8_t *high = blocks + 128;
        const int8_t *scales = (const int8_t *)(blocks + 192);
        const float d = q38_f16_to_f32(q38_load_u16(blocks + 208));
        for (int half = 0; half < 2; ++half) {
            const float *x = input + base + (uint64_t)half * 128u;
            for (int i = 0; i < 32; ++i) {
                const int scale_index = i / 16;
                const int q0 = ((low[i] & 15u) | ((high[i] & 3u) << 4)) - 32;
                const int q1 = ((low[i + 32] & 15u) | (((high[i] >> 2) & 3u) << 4)) - 32;
                const int q2 = ((low[i] >> 4) | (((high[i] >> 4) & 3u) << 4)) - 32;
                const int q3 = ((low[i + 32] >> 4) | (((high[i] >> 6) & 3u) << 4)) - 32;
                total = fmaf(d * scales[scale_index] * q0, x[i], total);
                total = fmaf(d * scales[scale_index + 2] * q1, x[i + 32], total);
                total = fmaf(d * scales[scale_index + 4] * q2, x[i + 64], total);
                total = fmaf(d * scales[scale_index + 6] * q3, x[i + 96], total);
            }
            low += 64;
            high += 32;
            scales += 8;
        }
    }
    return total;
}

static float q38_dot_q8_0(const uint8_t *blocks, const float *input,
                          uint64_t length)
{
    float total = 0.0f;
    for (uint64_t base = 0; base < length; base += 32, blocks += 34) {
        const float d = q38_f16_to_f32(q38_load_u16(blocks));
        const int8_t *quants = (const int8_t *)(blocks + 2);
        for (int i = 0; i < 32; ++i) {
            total = fmaf(d * quants[i], input[base + (uint64_t)i], total);
        }
    }
    return total;
}

static float q38_dot_q2_k_q8_k(const uint8_t *weights,
                               const Q38Q8KBlock *input,
                               uint64_t blocks)
{
#if defined(__AVX2__)
    __m256 accumulated = _mm256_setzero_ps();
    float accumulated_min = 0.0f;
#else
    float total = 0.0f;
#endif
    for (uint64_t block = 0; block < blocks; ++block, weights += 84) {
        const uint8_t *scales = weights;
        const uint8_t *quants = weights + 16;
        const float d = q38_f16_to_f32(q38_load_u16(weights + 80)) *
                        input[block].scale;
        const float dmin = q38_f16_to_f32(q38_load_u16(weights + 82)) *
                           input[block].scale;
#if defined(__AVX2__)
        const __m256i mask = _mm256_set1_epi8(3);
        __m256i weighted_lanes = _mm256_setzero_si256();
#else
        int weighted = 0;
#endif
        int minimum = 0;
        for (int half = 0; half < 2; ++half) {
#if defined(__AVX2__)
            const __m256i packed = _mm256_loadu_si256(
                (const __m256i *)(quants + half * 32));
#endif
            for (int field = 0; field < 4; ++field) {
                const int scale_index = half * 8 + field * 2;
                const int scale0 = scales[scale_index] & 15u;
                const int scale1 = scales[scale_index + 1] & 15u;
#if defined(__AVX2__)
                const __m256i q = _mm256_and_si256(
                    _mm256_srli_epi16(packed, field * 2), mask);
                const __m256i activation = _mm256_loadu_si256(
                    (const __m256i *)(input[block].quants +
                    half * 128 + field * 32));
                const __m256i dot = q38_products_u8_s8_32(q, activation);
                const __m256i scale = _mm256_set_m128i(
                    _mm_set1_epi32(scale1), _mm_set1_epi32(scale0));
                weighted_lanes = _mm256_add_epi32(weighted_lanes,
                    _mm256_mullo_epi32(dot, scale));
#else
                int dot0 = 0;
                int dot1 = 0;
                for (int lane = 0; lane < 16; ++lane) {
                    const int shift = field * 2;
                    dot0 += ((quants[half * 32 + lane] >> shift) & 3) *
                            input[block].quants[half * 128 + field * 32 + lane];
                    dot1 += ((quants[half * 32 + 16 + lane] >> shift) & 3) *
                            input[block].quants[half * 128 + field * 32 + 16 + lane];
                }
                weighted += scale0 * dot0 + scale1 * dot1;
#endif
                minimum += (scales[scale_index] >> 4) *
                           input[block].sums[scale_index];
                minimum += (scales[scale_index + 1] >> 4) *
                           input[block].sums[scale_index + 1];
            }
        }
#if defined(__AVX2__)
        accumulated = _mm256_fmadd_ps(_mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes), accumulated);
        accumulated_min = fmaf(-dmin, (float)minimum, accumulated_min);
#else
        total += d * weighted - dmin * minimum;
#endif
    }
#if defined(__AVX2__)
    return q38_sum_f32_8(accumulated) + accumulated_min;
#else
    return total;
#endif
}

static float q38_dot_q3_k_q8_k(const uint8_t *weights,
                               const Q38Q8KBlock *input,
                               uint64_t blocks)
{
#if defined(__AVX2__)
    __m256 accumulated = _mm256_setzero_ps();
#else
    float total = 0.0f;
#endif
    for (uint64_t block = 0; block < blocks; ++block, weights += 110) {
        const uint8_t *high = weights;
        const uint8_t *low = weights + 32;
        int8_t scales[16];
        q38_unpack_q3_scales(weights + 96, scales);
        const float d = q38_f16_to_f32(q38_load_u16(weights + 108)) *
                        input[block].scale;
#if defined(__AVX2__)
        const __m256i hbits = _mm256_loadu_si256((const __m256i *)high);
        const __m256i mask = _mm256_set1_epi8(3);
        const __m256i four = _mm256_set1_epi8(4);
        const __m256i zero = _mm256_setzero_si256();
        __m256i weighted_lanes = _mm256_setzero_si256();
#else
        int weighted = 0;
#endif
        int group = 0;
        for (int half = 0; half < 2; ++half) {
#if defined(__AVX2__)
            const __m256i packed = _mm256_loadu_si256(
                (const __m256i *)(low + half * 32));
#endif
            for (int field = 0; field < 4; ++field, ++group) {
                const uint8_t high_bit = (uint8_t)(1u << group);
                const int scale0 = scales[group * 2];
                const int scale1 = scales[group * 2 + 1];
#if defined(__AVX2__)
                const __m256i low_values = _mm256_and_si256(
                    _mm256_srli_epi16(packed, field * 2), mask);
                const __m256i absent = _mm256_cmpeq_epi8(
                    _mm256_and_si256(hbits,
                        _mm256_set1_epi8((char)high_bit)), zero);
                const __m256i q = _mm256_sub_epi8(low_values,
                    _mm256_and_si256(absent, four));
                const __m256i activation = _mm256_loadu_si256(
                    (const __m256i *)(input[block].quants +
                    half * 128 + field * 32));
                const __m256i dot = q38_products_s8_s8_32(q, activation);
                const __m256i scale = _mm256_set_m128i(
                    _mm_set1_epi32(scale1), _mm_set1_epi32(scale0));
                weighted_lanes = _mm256_add_epi32(weighted_lanes,
                    _mm256_mullo_epi32(dot, scale));
#else
                for (int lane = 0; lane < 32; ++lane) {
                    const int q = ((low[half * 32 + lane] >>
                                    (field * 2)) & 3) -
                                  ((high[lane] & high_bit) ? 0 : 4);
                    const int scale = lane < 16 ? scale0 : scale1;
                    weighted += scale * q *
                        input[block].quants[half * 128 + field * 32 + lane];
                }
#endif
            }
        }
#if defined(__AVX2__)
        accumulated = _mm256_fmadd_ps(_mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes), accumulated);
#else
        total += d * weighted;
#endif
    }
#if defined(__AVX2__)
    return q38_sum_f32_8(accumulated);
#else
    return total;
#endif
}

static float q38_dot_iq2_s_q8_k(const uint8_t *weights,
                                const Q38Q8KBlock *input,
                                uint64_t blocks)
{
#if defined(__AVX2__)
    __m256 accumulated = _mm256_setzero_ps();
    const __m128i scale_mask = _mm_set1_epi8(15);
    const __m128i scale_one = _mm_set1_epi8(1);
#else
    float total = 0.0f;
#endif
    for (uint64_t block = 0; block < blocks; ++block, weights += 82) {
        const float d = 0.125f *
            q38_f16_to_f32(q38_load_u16(weights)) * input[block].scale;
        const uint8_t *indices = weights + 2;
        const uint8_t *signs = weights + 34;
        const uint8_t *high = weights + 66;
        const uint8_t *scales = weights + 74;
#if defined(__AVX2__)
        uint64_t packed_scales;
        memcpy(&packed_scales, scales, sizeof(packed_scales));
        __m128i scale_bytes = _mm_set_epi64x(
            (long long)(packed_scales >> 4), (long long)packed_scales);
        scale_bytes = _mm_add_epi8(
            _mm_slli_epi16(_mm_and_si128(scale_bytes, scale_mask), 1),
            scale_one);
        const __m256i scale_words = _mm256_cvtepu8_epi16(scale_bytes);
        __m256i weighted_lanes = _mm256_setzero_si256();
#else
        int weighted = 0;
#endif
#if defined(__AVX2__)
        for (int group = 0; group < 8; ++group) {
            const uint8_t *code = indices + group * 4;
            const uint8_t high_bits = high[group];
            const __m256i quant = _mm256_set_epi64x(
                (long long)iq2s_grid[code[3] |
                    ((high_bits << 2) & 0x300)],
                (long long)iq2s_grid[code[2] |
                    ((high_bits << 4) & 0x300)],
                (long long)iq2s_grid[code[1] |
                    ((high_bits << 6) & 0x300)],
                (long long)iq2s_grid[code[0] |
                    ((high_bits << 8) & 0x300)]);
            const __m256i activation = _mm256_loadu_si256(
                (const __m256i *)(input[block].quants + group * 32));
            const __m256i signed_activation = q38_apply_raw_iq_signs(
                activation, signs + group * 4);
            const __m256i dot16 = _mm256_maddubs_epi16(
                quant, signed_activation);
            const int control = (2 * group) | ((2 * group + 1) << 8);
            const __m256i scale = _mm256_shuffle_epi8(
                scale_words, _mm256_set1_epi16((short)control));
            weighted_lanes = _mm256_add_epi32(weighted_lanes,
                _mm256_madd_epi16(dot16, scale));
        }
#else
        for (int group = 0; group < 8; ++group) {
            const int scale0 = 1 + 2 * (scales[group] & 15u);
            const int scale1 = 1 + 2 * (scales[group] >> 4);
            int subtotal0 = 0;
            int subtotal1 = 0;
            for (int lane = 0; lane < 32; ++lane) {
                const int section = lane / 8;
                const uint16_t grid_index = (uint16_t)indices[section] |
                    (uint16_t)((high[group] << (8 - 2 * section)) & 0x300);
                const uint8_t *grid = (const uint8_t *)(iq2s_grid +
                                                        grid_index);
                const int sign = signs[section] & kmask_iq2xs[lane % 8]
                    ? -1 : 1;
                const int product = input[block].quants[group * 32 + lane] *
                                    grid[lane % 8] * sign;
                if (section < 2) subtotal0 += product;
                else subtotal1 += product;
            }
            weighted += scale0 * subtotal0 + scale1 * subtotal1;
            indices += 4;
            signs += 4;
        }
#endif
#if defined(__AVX2__)
        accumulated = _mm256_fmadd_ps(_mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes), accumulated);
#else
        total += d * weighted;
#endif
    }
#if defined(__AVX2__)
    return q38_sum_f32_8(accumulated);
#else
    return total;
#endif
}

static float q38_dot_iq2_xxs_q8_k(const uint8_t *weights,
                                   const Q38Q8KBlock *input,
                                   uint64_t blocks)
{
#if defined(__AVX2__)
    __m256 accumulated = _mm256_setzero_ps();
#else
    float total = 0.0f;
#endif
    for (uint64_t block = 0; block < blocks; ++block, weights += 66) {
        const float d = 0.125f *
            q38_f16_to_f32(q38_load_u16(weights)) * input[block].scale;
        const uint8_t *codes = weights + 2;
#if defined(__AVX2__)
        __m256i weighted_lanes = _mm256_setzero_si256();
#else
        int weighted = 0;
#endif
        for (int group = 0; group < 8; ++group, codes += 8) {
            const uint32_t metadata = q38_load_u32(codes + 4);
            const int scale = 1 + 2 * (int)(metadata >> 28);
#if defined(__AVX2__)
            const __m256i quant = _mm256_set_epi64x(
                (long long)iq2xxs_grid[codes[3]],
                (long long)iq2xxs_grid[codes[2]],
                (long long)iq2xxs_grid[codes[1]],
                (long long)iq2xxs_grid[codes[0]]);
            const __m256i activation = _mm256_loadu_si256(
                (const __m256i *)(input[block].quants + group * 32));
            const __m256i signed_activation = q38_apply_iq_signs(
                activation, metadata & 127u, (metadata >> 7) & 127u,
                (metadata >> 14) & 127u, (metadata >> 21) & 127u);
            const __m256i dot = q38_products_u8_s8_32(
                quant, signed_activation);
            weighted_lanes = _mm256_add_epi32(weighted_lanes,
                _mm256_mullo_epi32(dot, _mm256_set1_epi32(scale)));
#else
            int subtotal = 0;
            for (int section = 0; section < 4; ++section) {
                const uint8_t *grid = (const uint8_t *)(iq2xxs_grid +
                                                        codes[section]);
                const uint8_t signs = ksigns_iq2xs[
                    (metadata >> (7 * section)) & 127u];
                for (int lane = 0; lane < 8; ++lane) {
                    const int sign = signs & kmask_iq2xs[lane] ? -1 : 1;
                    subtotal += input[block].quants[
                        group * 32 + section * 8 + lane] * grid[lane] * sign;
                }
            }
            weighted += scale * subtotal;
#endif
        }
#if defined(__AVX2__)
        accumulated = _mm256_fmadd_ps(_mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes), accumulated);
#else
        total += d * weighted;
#endif
    }
#if defined(__AVX2__)
    return q38_sum_f32_8(accumulated);
#else
    return total;
#endif
}

static float q38_dot_iq2_xs_q8_k(const uint8_t *weights,
                                 const Q38Q8KBlock *input,
                                 uint64_t blocks)
{
#if defined(__AVX2__)
    __m256 accumulated = _mm256_setzero_ps();
#else
    float total = 0.0f;
#endif
    for (uint64_t block = 0; block < blocks; ++block, weights += 74) {
        const float d = 0.125f *
            q38_f16_to_f32(q38_load_u16(weights)) * input[block].scale;
        const uint8_t *codes = weights + 2;
        const uint8_t *scales = weights + 66;
#if defined(__AVX2__)
        __m256i weighted_lanes = _mm256_setzero_si256();
#else
        int weighted = 0;
#endif
        for (int group = 0; group < 8; ++group, codes += 8) {
            uint16_t code[4];
            for (int section = 0; section < 4; ++section)
                code[section] = q38_load_u16(codes + section * 2);
            const int scale0 = 1 + 2 * (scales[group] & 15u);
            const int scale1 = 1 + 2 * (scales[group] >> 4);
#if defined(__AVX2__)
            const __m256i quant = _mm256_set_epi64x(
                (long long)iq2xs_grid[code[3] & 511u],
                (long long)iq2xs_grid[code[2] & 511u],
                (long long)iq2xs_grid[code[1] & 511u],
                (long long)iq2xs_grid[code[0] & 511u]);
            const __m256i activation = _mm256_loadu_si256(
                (const __m256i *)(input[block].quants + group * 32));
            const __m256i signed_activation = q38_apply_iq_signs(
                activation, code[0] >> 9, code[1] >> 9,
                code[2] >> 9, code[3] >> 9);
            const __m256i dot = q38_products_u8_s8_32(
                quant, signed_activation);
            const __m256i scale = _mm256_set_m128i(
                _mm_set1_epi32(scale1), _mm_set1_epi32(scale0));
            weighted_lanes = _mm256_add_epi32(weighted_lanes,
                _mm256_mullo_epi32(dot, scale));
#else
            int subtotal[2] = {0, 0};
            for (int section = 0; section < 4; ++section) {
                const uint8_t *grid = (const uint8_t *)(iq2xs_grid +
                                                        (code[section] & 511u));
                const uint8_t signs = ksigns_iq2xs[code[section] >> 9];
                for (int lane = 0; lane < 8; ++lane) {
                    const int sign = signs & kmask_iq2xs[lane] ? -1 : 1;
                    subtotal[section / 2] += input[block].quants[
                        group * 32 + section * 8 + lane] * grid[lane] * sign;
                }
            }
            weighted += scale0 * subtotal[0] + scale1 * subtotal[1];
#endif
        }
#if defined(__AVX2__)
        accumulated = _mm256_fmadd_ps(_mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes), accumulated);
#else
        total += d * weighted;
#endif
    }
#if defined(__AVX2__)
    return q38_sum_f32_8(accumulated);
#else
    return total;
#endif
}

static float q38_dot_iq3_xxs_q8_k(const uint8_t *weights,
                                   const Q38Q8KBlock *input,
                                   uint64_t blocks)
{
#if defined(__AVX2__)
    __m256 accumulated = _mm256_setzero_ps();
#else
    float total = 0.0f;
#endif
    for (uint64_t block = 0; block < blocks; ++block, weights += 98) {
        const float d = 0.25f *
            q38_f16_to_f32(q38_load_u16(weights)) * input[block].scale;
        const uint8_t *indices = weights + 2;
        const uint8_t *metadata = weights + 66;
#if defined(__AVX2__)
        __m256i weighted_lanes[2] = {
            _mm256_setzero_si256(), _mm256_setzero_si256()
        };
#else
        int weighted = 0;
#endif
#if defined(__AVX2__)
        for (int group = 0; group < 8; ++group) {
            const uint32_t packed = q38_load_u32(metadata + group * 4);
            const int scale = 1 + 2 * (int)(packed >> 28);
            const uint8_t *code = indices + group * 8;
            const __m256i quant = _mm256_set_epi32(
                (int)iq3xxs_grid[code[7]], (int)iq3xxs_grid[code[6]],
                (int)iq3xxs_grid[code[5]], (int)iq3xxs_grid[code[4]],
                (int)iq3xxs_grid[code[3]], (int)iq3xxs_grid[code[2]],
                (int)iq3xxs_grid[code[1]], (int)iq3xxs_grid[code[0]]);
            const __m256i activation = _mm256_loadu_si256(
                (const __m256i *)(input[block].quants + group * 32));
            const __m256i signed_activation = q38_apply_iq_signs(
                activation, packed & 127u, (packed >> 7) & 127u,
                (packed >> 14) & 127u, (packed >> 21) & 127u);
            const __m256i dot = q38_products_u8_s8_32(
                quant, signed_activation);
            weighted_lanes[group & 1] = _mm256_add_epi32(
                weighted_lanes[group & 1],
                _mm256_mullo_epi32(dot, _mm256_set1_epi32(scale)));
        }
#else
        for (int group = 0; group < 8; ++group) {
            const uint32_t packed = q38_load_u32(metadata + group * 4);
            const int scale = 1 + 2 * (int)(packed >> 28);
            const uint8_t *code = indices + group * 8;
            int subtotal = 0;
            for (int section = 0; section < 4; ++section) {
                const uint8_t *grid0 = (const uint8_t *)(iq3xxs_grid +
                                                         code[section * 2]);
                const uint8_t *grid1 = (const uint8_t *)(iq3xxs_grid +
                                                         code[section * 2 + 1]);
                const uint8_t signs = ksigns_iq2xs[
                    (packed >> (7 * section)) & 127u];
                for (int lane = 0; lane < 4; ++lane) {
                    const int sign0 = signs & kmask_iq2xs[lane] ? -1 : 1;
                    const int sign1 = signs & kmask_iq2xs[lane + 4] ? -1 : 1;
                    subtotal += input[block].quants[
                        group * 32 + section * 8 + lane] * grid0[lane] * sign0;
                    subtotal += input[block].quants[
                        group * 32 + section * 8 + lane + 4] * grid1[lane] *
                        sign1;
                }
            }
            weighted += scale * subtotal;
        }
#endif
#if defined(__AVX2__)
        accumulated = _mm256_fmadd_ps(_mm256_set1_ps(d),
            _mm256_cvtepi32_ps(_mm256_add_epi32(
                weighted_lanes[0], weighted_lanes[1])), accumulated);
#else
        total += d * weighted;
#endif
    }
#if defined(__AVX2__)
    return q38_sum_f32_8(accumulated);
#else
    return total;
#endif
}

static float q38_dot_iq3_s_q8_k(const uint8_t *weights,
                                const Q38Q8KBlock *input,
                                uint64_t blocks)
{
#if defined(__AVX2__)
    __m256 accumulated = _mm256_setzero_ps();
#else
    float total = 0.0f;
#endif
    for (uint64_t block = 0; block < blocks; ++block, weights += 110) {
        const float d = q38_f16_to_f32(q38_load_u16(weights)) *
                        input[block].scale;
        const uint8_t *indices = weights + 2;
        const uint8_t *high = weights + 66;
        const uint8_t *signs = weights + 74;
        const uint8_t *scales = weights + 106;
#if defined(__AVX2__)
        __m256i weighted_lanes = _mm256_setzero_si256();
#else
        int weighted = 0;
#endif
#if defined(__AVX2__)
        for (int group = 0; group < 8; group += 2) {
            __m256i quant[2];
            q38_iq3_s_grid_vectors(quant, indices + group * 8, high + group);
            for (int pair = 0; pair < 2; ++pair) {
                const int current = group + pair;
                const __m256i activation = _mm256_loadu_si256(
                    (const __m256i *)(input[block].quants + current * 32));
                const __m256i signed_activation = q38_apply_raw_iq_signs(
                    activation, signs + current * 4);
                const __m256i dot = q38_products_u8_s8_32(
                    quant[pair], signed_activation);
                const int scale = 1 + 2 * ((scales[group / 2] >>
                    (4 * pair)) & 15u);
                weighted_lanes = _mm256_add_epi32(weighted_lanes,
                    _mm256_mullo_epi32(dot, _mm256_set1_epi32(scale)));
            }
        }
#else
        for (int group = 0; group < 8; ++group) {
            const uint8_t *code = indices + group * 8;
            const uint8_t high_bits = high[group];
            const uint8_t *group_signs = signs + group * 4;
            const int scale = 1 + 2 * ((scales[group / 2] >>
                (4 * (group % 2))) & 15u);
            int subtotal = 0;
            for (int section = 0; section < 4; ++section) {
                const uint16_t index0 = (uint16_t)code[section * 2] |
                    (uint16_t)((high_bits << (8 - 2 * section)) & 0x100);
                const uint16_t index1 =
                    (uint16_t)code[section * 2 + 1] |
                    (uint16_t)((high_bits << (7 - 2 * section)) & 0x100);
                const uint8_t *grid0 = (const uint8_t *)(iq3s_grid + index0);
                const uint8_t *grid1 = (const uint8_t *)(iq3s_grid + index1);
                for (int lane = 0; lane < 4; ++lane) {
                    const int sign0 = group_signs[section] & (1u << lane)
                        ? -1 : 1;
                    const int sign1 = group_signs[section] &
                        (1u << (lane + 4)) ? -1 : 1;
                    subtotal += input[block].quants[
                        group * 32 + section * 8 + lane] *
                        grid0[lane] * sign0;
                    subtotal += input[block].quants[
                        group * 32 + section * 8 + lane + 4] *
                        grid1[lane] * sign1;
                }
            }
            weighted += scale * subtotal;
        }
#endif
#if defined(__AVX2__)
        accumulated = _mm256_fmadd_ps(_mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes), accumulated);
#else
        total += d * weighted;
#endif
    }
#if defined(__AVX2__)
    return q38_sum_f32_8(accumulated);
#else
    return total;
#endif
}

static float q38_dot_iq4_xs_q8_k(const uint8_t *weights,
                                 const Q38Q8KBlock *input,
                                 uint64_t blocks)
{
#if defined(__AVX2__)
    const __m128i table = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
    const __m128i nibble_mask = _mm_set1_epi8(15);
    __m256 accumulated = _mm256_setzero_ps();
#else
    float total = 0.0f;
#endif
    for (uint64_t block = 0; block < blocks; ++block, weights += 136) {
        const float d = q38_f16_to_f32(q38_load_u16(weights)) *
                        input[block].scale;
        const uint16_t scale_high = q38_load_u16(weights + 2);
        const uint8_t *scale_low = weights + 4;
        const uint8_t *quants = weights + 8;
#if defined(__AVX2__)
        __m256i weighted_lanes = _mm256_setzero_si256();
#else
        int weighted = 0;
#endif
        for (int group = 0; group < 8; ++group) {
            const int scale = ((scale_low[group / 2] >>
                (4 * (group % 2))) & 15u) |
                (((scale_high >> (2 * group)) & 3u) << 4);
#if defined(__AVX2__)
            const __m128i packed = _mm_loadu_si128(
                (const __m128i *)(quants + group * 16));
            const __m128i low = _mm_shuffle_epi8(
                table, _mm_and_si128(packed, nibble_mask));
            const __m128i high = _mm_shuffle_epi8(table,
                _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask));
            const __m256i quant = _mm256_set_m128i(high, low);
            const __m256i activation = _mm256_loadu_si256(
                (const __m256i *)(input[block].quants + group * 32));
            const __m256i dot = q38_products_s8_s8_32(quant, activation);
            weighted_lanes = _mm256_add_epi32(weighted_lanes,
                _mm256_mullo_epi32(dot,
                    _mm256_set1_epi32(scale - 32)));
#else
            int subtotal = 0;
            for (int lane = 0; lane < 16; ++lane) {
                const uint8_t packed = quants[group * 16 + lane];
                subtotal += input[block].quants[group * 32 + lane] *
                            kvalues_iq4nl[packed & 15u];
                subtotal += input[block].quants[group * 32 + lane + 16] *
                            kvalues_iq4nl[packed >> 4];
            }
            weighted += (scale - 32) * subtotal;
#endif
        }
#if defined(__AVX2__)
        accumulated = _mm256_fmadd_ps(_mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes), accumulated);
#else
        total += d * weighted;
#endif
    }
#if defined(__AVX2__)
    return q38_sum_f32_8(accumulated);
#else
    return total;
#endif
}

static float q38_dot_iq4_nl_q8_k(const uint8_t *weights,
                                 const Q38Q8KBlock *input,
                                 uint64_t blocks)
{
#if defined(__AVX2__)
    const __m128i table = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
    const __m128i nibble_mask = _mm_set1_epi8(15);
    __m256 accumulated = _mm256_setzero_ps();
#else
    float total = 0.0f;
#endif
    for (uint64_t block = 0; block < blocks; ++block, weights += 144) {
        for (int group = 0; group < 8; ++group) {
            const uint8_t *packed_block = weights + group * 18;
            const float d = q38_f16_to_f32(q38_load_u16(packed_block)) *
                            input[block].scale;
            const uint8_t *quants = packed_block + 2;
#if defined(__AVX2__)
            const __m128i packed = _mm_loadu_si128((const __m128i *)quants);
            const __m128i low = _mm_shuffle_epi8(
                table, _mm_and_si128(packed, nibble_mask));
            const __m128i high = _mm_shuffle_epi8(table,
                _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask));
            const __m256i quant = _mm256_set_m128i(high, low);
            const __m256i activation = _mm256_loadu_si256(
                (const __m256i *)(input[block].quants + group * 32));
            const __m256i dot = q38_products_s8_s8_32(quant, activation);
            accumulated = _mm256_fmadd_ps(_mm256_set1_ps(d),
                _mm256_cvtepi32_ps(dot), accumulated);
#else
            int weighted = 0;
            for (int lane = 0; lane < 16; ++lane) {
                weighted += input[block].quants[group * 32 + lane] *
                            kvalues_iq4nl[quants[lane] & 15u];
                weighted += input[block].quants[group * 32 + lane + 16] *
                            kvalues_iq4nl[quants[lane] >> 4];
            }
            total += d * weighted;
#endif
        }
    }
#if defined(__AVX2__)
    return q38_sum_f32_8(accumulated);
#else
    return total;
#endif
}

static float q38_dot_iq1_s_q8_k(const uint8_t *weights,
                                const Q38Q8KBlock *input,
                                uint64_t blocks)
{
#if defined(__AVX2__)
    __m256 accumulated = _mm256_setzero_ps();
    float accumulated_correction = 0.0f;
#else
    float total = 0.0f;
#endif
    for (uint64_t block = 0; block < blocks; ++block, weights += 50) {
        const float d = q38_f16_to_f32(q38_load_u16(weights)) *
                        input[block].scale;
        const uint8_t *indices = weights + 2;
        const uint8_t *high = weights + 34;
#if defined(__AVX2__)
        __m256i weighted_lanes = _mm256_setzero_si256();
        int correction = 0;
        for (int group = 0; group < 8; group += 2) {
            const uint16_t high0 = q38_load_u16(high + group * 2);
            const uint16_t high1 = q38_load_u16(high + (group + 1) * 2);
            const __m256i quant0 = q38_iq1_s_grid32(indices, high0);
            const __m256i quant1 = q38_iq1_s_grid32(indices + 4, high1);
            const __m256i activation0 = _mm256_loadu_si256(
                (const __m256i *)(input[block].quants + group * 32));
            const __m256i activation1 = _mm256_loadu_si256(
                (const __m256i *)(input[block].quants + (group + 1) * 32));
            const int scale0 = 1 + 2 * ((high0 >> 12) & 7u);
            const int scale1 = 1 + 2 * ((high1 >> 12) & 7u);
            weighted_lanes = _mm256_add_epi32(weighted_lanes,
                q38_products_s8_s8_scaled_32(quant0, activation0, scale0));
            weighted_lanes = _mm256_add_epi32(weighted_lanes,
                q38_products_s8_s8_scaled_32(quant1, activation1, scale1));
            correction += scale0 * (high0 & 0x8000u ? -1 : 1) *
                (input[block].sums[group * 2] +
                 input[block].sums[group * 2 + 1]);
            correction += scale1 * (high1 & 0x8000u ? -1 : 1) *
                (input[block].sums[(group + 1) * 2] +
                 input[block].sums[(group + 1) * 2 + 1]);
            indices += 8;
        }
        accumulated = _mm256_fmadd_ps(_mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes), accumulated);
        accumulated_correction += d * correction;
#else
        int weighted = 0;
        int correction = 0;
        for (int group = 0; group < 8; ++group) {
            const uint16_t packed = q38_load_u16(high + group * 2);
            const int scale = 1 + 2 * ((packed >> 12) & 7u);
            const int delta = packed & 0x8000u ? -1 : 1;
            int subtotal = 0;
            for (int section = 0; section < 4; ++section) {
                const uint16_t grid_index = indices[section] |
                    (uint16_t)(((packed >> (3 * section)) & 7u) << 8);
                const int8_t *grid = (const int8_t *)(iq1s_grid + grid_index);
                for (int lane = 0; lane < 8; ++lane) {
                    subtotal += input[block].quants[
                        group * 32 + section * 8 + lane] * grid[lane];
                }
            }
            weighted += scale * subtotal;
            correction += scale * delta *
                (input[block].sums[group * 2] +
                 input[block].sums[group * 2 + 1]);
            indices += 4;
        }
        total += d * (weighted + 0.125f * correction);
#endif
    }
#if defined(__AVX2__)
    return q38_sum_f32_8(accumulated) +
           0.125f * accumulated_correction;
#else
    return total;
#endif
}

typedef struct {
    float base_d;
    uint16_t grid_index[32];
    int8_t signed_scale[8];
} Q38IQ1SRepackedBlock;

#if defined(__AVX2__)
static void q38_repack_iq1_s_block(Q38IQ1SRepackedBlock *output,
                                   const uint8_t input[50])
{
    output->base_d = q38_f16_to_f32(q38_load_u16(input));
    const uint8_t *low = input + 2;
    const uint8_t *high = input + 34;
    for (int group = 0; group < 8; ++group) {
        const uint16_t packed = q38_load_u16(high + group * 2);
        const int scale = 1 + 2 * ((packed >> 12) & 7u);
        output->signed_scale[group] = (int8_t)(
            packed & 0x8000u ? -scale : scale);
        for (int section = 0; section < 4; ++section) {
            output->grid_index[group * 4 + section] =
                low[group * 4 + section] |
                (uint16_t)(((packed >> (3 * section)) & 7u) << 8);
        }
    }
}
#endif

static float q38_dot_iq1_s_repacked_q8_k(
                                const Q38IQ1SRepackedBlock *weights,
                                const Q38Q8KBlock *input, uint64_t blocks)
{
#if defined(__AVX2__)
    __m256 accumulated = _mm256_setzero_ps();
    float accumulated_correction = 0.0f;
    for (uint64_t block = 0; block < blocks; ++block) {
        const Q38IQ1SRepackedBlock *weight = weights + block;
        __m256i weighted_lanes = _mm256_setzero_si256();
        int correction = 0;
        for (int group = 0; group < 8; group += 2) {
            const uint16_t *index0 = weight->grid_index + group * 4;
            const uint16_t *index1 = index0 + 4;
            const __m256i quant0 = _mm256_set_epi64x(
                (long long)iq1s_grid[index0[3]],
                (long long)iq1s_grid[index0[2]],
                (long long)iq1s_grid[index0[1]],
                (long long)iq1s_grid[index0[0]]);
            const __m256i quant1 = _mm256_set_epi64x(
                (long long)iq1s_grid[index1[3]],
                (long long)iq1s_grid[index1[2]],
                (long long)iq1s_grid[index1[1]],
                (long long)iq1s_grid[index1[0]]);
            const __m256i activation0 = _mm256_loadu_si256(
                (const __m256i *)(input[block].quants + group * 32));
            const __m256i activation1 = _mm256_loadu_si256(
                (const __m256i *)(input[block].quants + (group + 1) * 32));
            const int signed_scale0 = weight->signed_scale[group];
            const int signed_scale1 = weight->signed_scale[group + 1];
            const int scale0 = signed_scale0 < 0
                ? -signed_scale0 : signed_scale0;
            const int scale1 = signed_scale1 < 0
                ? -signed_scale1 : signed_scale1;
            weighted_lanes = _mm256_add_epi32(weighted_lanes,
                q38_products_s8_s8_scaled_32(quant0, activation0, scale0));
            weighted_lanes = _mm256_add_epi32(weighted_lanes,
                q38_products_s8_s8_scaled_32(quant1, activation1, scale1));
            correction += signed_scale0 *
                (input[block].sums[group * 2] +
                 input[block].sums[group * 2 + 1]);
            correction += signed_scale1 *
                (input[block].sums[(group + 1) * 2] +
                 input[block].sums[(group + 1) * 2 + 1]);
        }
        const float d = weight->base_d * input[block].scale;
        accumulated = _mm256_fmadd_ps(_mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes), accumulated);
        accumulated_correction += d * correction;
    }
    return q38_sum_f32_8(accumulated) +
           0.125f * accumulated_correction;
#else
    (void)weights;
    (void)input;
    (void)blocks;
    return 0.0f;
#endif
}

int q38_prepare_iq1_s_repacks(Q38GGUF *gguf)
{
    if (!gguf) return 0;
#if defined(__AVX2__)
    for (uint64_t i = 0; i < gguf->tensor_count; ++i) {
        Q38GGUFTensor *tensor = &gguf->tensors[i];
        if (tensor->type != Q38_GGML_IQ1_S || tensor->n_dims != 2 ||
            tensor->shape[0] % Q38_Q8_K_BLOCK_SIZE != 0 ||
            tensor->iq1_s_repack) continue;
        const uint64_t blocks = tensor->shape[0] / Q38_Q8_K_BLOCK_SIZE;
        const uint64_t block_count = blocks * tensor->shape[1];
        if (blocks != 0 && block_count / blocks != tensor->shape[1]) {
            q38_release_iq1_s_repacks(gguf);
            return 0;
        }
        if (block_count > SIZE_MAX / sizeof(Q38IQ1SRepackedBlock)) {
            q38_release_iq1_s_repacks(gguf);
            return 0;
        }
        Q38IQ1SRepackedBlock *data = malloc(
            (size_t)block_count * sizeof(*data));
        if (!data) {
            q38_release_iq1_s_repacks(gguf);
            return 0;
        }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (uint64_t block = 0; block < block_count; ++block)
            q38_repack_iq1_s_block(data + block,
                                   tensor->data + block * 50u);
        tensor->iq1_s_repack = data;
    }
#endif
    return 1;
}

void q38_release_iq1_s_repacks(Q38GGUF *gguf)
{
    if (!gguf || !gguf->tensors) return;
    for (uint64_t i = 0; i < gguf->tensor_count; ++i) {
        free(gguf->tensors[i].iq1_s_repack);
        gguf->tensors[i].iq1_s_repack = NULL;
    }
}

static float q38_dot_iq1_m_q8_k(const uint8_t *weights,
                                const Q38Q8KBlock *input,
                                uint64_t blocks)
{
    float total = 0.0f;
#if defined(__AVX2__)
    const __m256i one = _mm256_set1_epi8(1);
#endif
    for (uint64_t block = 0; block < blocks; ++block, weights += 56) {
        const uint8_t *indices = weights;
        const uint8_t *high = weights + 32;
        const uint8_t *packed_scales = weights + 48;
        const uint16_t scale_bits = (q38_load_u16(packed_scales) >> 12) |
             ((q38_load_u16(packed_scales + 2) >> 8) & 0x00f0u) |
             ((q38_load_u16(packed_scales + 4) >> 4) & 0x0f00u) |
             (q38_load_u16(packed_scales + 6) & 0xf000u);
        const float d = q38_f16_to_f32(scale_bits) * input[block].scale;
#if defined(__AVX2__)
        __m256i weighted_lanes = _mm256_setzero_si256();
        __m256i correction_lanes = _mm256_setzero_si256();
        for (int group = 0; group < 8; group += 2) {
#if defined(__BMI2__)
            const __m256i activation0 = _mm256_loadu_si256(
                (const __m256i *)(input[block].quants + group * 32));
            const __m256i activation1 = _mm256_loadu_si256(
                (const __m256i *)(input[block].quants + (group + 1) * 32));
            const uint64_t packed0 =
                _pdep_u64(q38_load_u32(indices), 0x00ff00ff00ff00ffULL) |
                _pdep_u64(q38_load_u16(high) & 0x7777u,
                          0x0f000f000f000f00ULL);
            const uint64_t packed1 =
                _pdep_u64(q38_load_u32(indices + 4),
                          0x00ff00ff00ff00ffULL) |
                _pdep_u64(q38_load_u16(high + 2) & 0x7777u,
                          0x0f000f000f000f00ULL);
            const __m256i quant0 = _mm256_set_epi64x(
                (long long)iq1s_grid[(uint16_t)(packed0 >> 48)],
                (long long)iq1s_grid[(uint16_t)(packed0 >> 32)],
                (long long)iq1s_grid[(uint16_t)(packed0 >> 16)],
                (long long)iq1s_grid[(uint16_t)packed0]);
            const __m256i quant1 = _mm256_set_epi64x(
                (long long)iq1s_grid[(uint16_t)(packed1 >> 48)],
                (long long)iq1s_grid[(uint16_t)(packed1 >> 32)],
                (long long)iq1s_grid[(uint16_t)(packed1 >> 16)],
                (long long)iq1s_grid[(uint16_t)packed1]);
            const uint64_t delta_sign = _pdep_u64(
                q38_load_u32(high) & 0x88888888u,
                0xf0f0f0f0f0f0f0f0ULL);
            const __m256i delta0 = _mm256_or_si256(one,
                _mm256_cvtepi8_epi64(_mm_set1_epi32((int)delta_sign)));
            const __m256i delta1 = _mm256_or_si256(one,
                _mm256_cvtepi8_epi64(
                    _mm_set1_epi32((int)(delta_sign >> 32))));
#else
            const __m256i quant0 = _mm256_set_epi64x(
                (long long)iq1s_grid[indices[3] |
                    (((uint16_t)high[1] << 4) & 0x700)],
                (long long)iq1s_grid[indices[2] |
                    (((uint16_t)high[1] << 8) & 0x700)],
                (long long)iq1s_grid[indices[1] |
                    (((uint16_t)high[0] << 4) & 0x700)],
                (long long)iq1s_grid[indices[0] |
                    (((uint16_t)high[0] << 8) & 0x700)]);
            const __m256i quant1 = _mm256_set_epi64x(
                (long long)iq1s_grid[indices[7] |
                    (((uint16_t)high[3] << 4) & 0x700)],
                (long long)iq1s_grid[indices[6] |
                    (((uint16_t)high[3] << 8) & 0x700)],
                (long long)iq1s_grid[indices[5] |
                    (((uint16_t)high[2] << 4) & 0x700)],
                (long long)iq1s_grid[indices[4] |
                    (((uint16_t)high[2] << 8) & 0x700)]);
            const __m256i activation0 = _mm256_loadu_si256(
                (const __m256i *)(input[block].quants + group * 32));
            const __m256i activation1 = _mm256_loadu_si256(
                (const __m256i *)(input[block].quants + (group + 1) * 32));
            const __m256i delta0 = _mm256_set_epi64x(
                high[1] & 0x80u ? -1LL : 0x0101010101010101LL,
                high[1] & 0x08u ? -1LL : 0x0101010101010101LL,
                high[0] & 0x80u ? -1LL : 0x0101010101010101LL,
                high[0] & 0x08u ? -1LL : 0x0101010101010101LL);
            const __m256i delta1 = _mm256_set_epi64x(
                high[3] & 0x80u ? -1LL : 0x0101010101010101LL,
                high[3] & 0x08u ? -1LL : 0x0101010101010101LL,
                high[2] & 0x80u ? -1LL : 0x0101010101010101LL,
                high[2] & 0x08u ? -1LL : 0x0101010101010101LL);
#endif
            const uint16_t scales = q38_load_u16(
                packed_scales + (group / 2) * 2);
            const int scale0 = 1 + 2 * ((scales >> 0) & 7u);
            const int scale1 = 1 + 2 * ((scales >> 3) & 7u);
            const int scale2 = 1 + 2 * ((scales >> 6) & 7u);
            const int scale3 = 1 + 2 * ((scales >> 9) & 7u);
            const __m256i scales0 = _mm256_set_m128i(
                _mm_set1_epi16((short)scale1),
                _mm_set1_epi16((short)scale0));
            const __m256i scales1 = _mm256_set_m128i(
                _mm_set1_epi16((short)scale3),
                _mm_set1_epi16((short)scale2));
            const __m256i dot0 = _mm256_maddubs_epi16(
                _mm256_abs_epi8(quant0),
                _mm256_sign_epi8(activation0, quant0));
            const __m256i dot1 = _mm256_maddubs_epi16(
                _mm256_abs_epi8(quant1),
                _mm256_sign_epi8(activation1, quant1));
            const __m256i correction0 = _mm256_maddubs_epi16(
                one, _mm256_sign_epi8(activation0, delta0));
            const __m256i correction1 = _mm256_maddubs_epi16(
                one, _mm256_sign_epi8(activation1, delta1));
            weighted_lanes = _mm256_add_epi32(weighted_lanes,
                _mm256_add_epi32(_mm256_madd_epi16(dot0, scales0),
                                 _mm256_madd_epi16(dot1, scales1)));
            correction_lanes = _mm256_add_epi32(correction_lanes,
                _mm256_add_epi32(
                    _mm256_madd_epi16(correction0, scales0),
                    _mm256_madd_epi16(correction1, scales1)));
            indices += 8;
            high += 4;
        }
        const int weighted = q38_sum_i32_8(weighted_lanes);
        const int correction = q38_sum_i32_8(correction_lanes);
#else
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
                const uint8_t delta_bit = section % 2 ? 0x80u : 0x08u;
                const int delta = high[section / 2] & delta_bit ? -1 : 1;
                for (int lane = 0; lane < 8; ++lane) {
                    const int activation = input[block].quants[
                        group * 32 + section * 8 + lane];
                    dot[section / 2] += activation * grid[lane];
                    correction_dot[section / 2] += activation * delta;
                }
            }
            const uint16_t scales = q38_load_u16(
                packed_scales + (group / 2) * 2);
            const int shift = 6 * (group % 2);
            const int scale0 = 1 + 2 * ((scales >> shift) & 7u);
            const int scale1 = 1 + 2 * ((scales >> (shift + 3)) & 7u);
            weighted += scale0 * dot[0] + scale1 * dot[1];
            correction += scale0 * correction_dot[0] +
                          scale1 * correction_dot[1];
            indices += 4;
            high += 2;
        }
#endif
        total += d * (weighted + 0.125f * correction);
    }
    return total;
}

static float q38_dot_q4_k_q8_k(const uint8_t *weights,
                               const Q38Q8KBlock *input,
                               uint64_t blocks)
{
#if defined(__AVX2__)
    __m256 accumulated = _mm256_setzero_ps();
    __m128 accumulated_min = _mm_setzero_ps();
#else
    float total = 0.0f;
#endif
    for (uint64_t block = 0; block < blocks; ++block, weights += 144) {
        const float d = q38_f16_to_f32(q38_load_u16(weights)) * input[block].scale;
        const float dmin = q38_f16_to_f32(q38_load_u16(weights + 2)) *
                           input[block].scale;
        const uint8_t *scales = weights + 4;
        const uint8_t *quants = weights + 16;
#if !defined(__AVX2__)
        int weighted = 0;
        int minimum = 0;
#else
        __m256i weighted_lanes = _mm256_setzero_si256();
        int16_t minimum_scales[8];
#endif
        for (int chunk = 0; chunk < 4; ++chunk) {
            uint8_t scale0, min0, scale1, min1;
            q38_scale_min_k4(chunk * 2, scales, &scale0, &min0);
            q38_scale_min_k4(chunk * 2 + 1, scales, &scale1, &min1);
#if defined(__AVX2__)
            const __m256i packed = _mm256_loadu_si256(
                (const __m256i *)(quants + chunk * 32));
            const __m256i mask = _mm256_set1_epi8(15);
            const __m256i low = _mm256_and_si256(packed, mask);
            const __m256i high = _mm256_and_si256(
                _mm256_srli_epi16(packed, 4), mask);
            const __m256i dot0 = q38_products_u8_s8_32(low,
                _mm256_loadu_si256((const __m256i *)(
                    input[block].quants + chunk * 64)));
            const __m256i dot1 = q38_products_u8_s8_32(high,
                _mm256_loadu_si256((const __m256i *)(
                    input[block].quants + chunk * 64 + 32)));
            weighted_lanes = _mm256_add_epi32(weighted_lanes,
                _mm256_mullo_epi32(dot0, _mm256_set1_epi32(scale0)));
            weighted_lanes = _mm256_add_epi32(weighted_lanes,
                _mm256_mullo_epi32(dot1, _mm256_set1_epi32(scale1)));
            minimum_scales[chunk * 2] = min0;
            minimum_scales[chunk * 2 + 1] = min1;
#else
            int dot0 = 0;
            int dot1 = 0;
            for (int i = 0; i < 32; ++i) {
                dot0 += (quants[chunk * 32 + i] & 15u) *
                        input[block].quants[chunk * 64 + i];
                dot1 += (quants[chunk * 32 + i] >> 4) *
                        input[block].quants[chunk * 64 + 32 + i];
            }
            weighted += scale0 * dot0 + scale1 * dot1;
            minimum += min0 * (input[block].sums[chunk * 4] +
                               input[block].sums[chunk * 4 + 1]);
            minimum += min1 * (input[block].sums[chunk * 4 + 2] +
                               input[block].sums[chunk * 4 + 3]);
#endif
        }
#if defined(__AVX2__)
        accumulated = _mm256_fmadd_ps(_mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes), accumulated);
        const __m128i sums0 = _mm_loadu_si128(
            (const __m128i *)(input[block].sums));
        const __m128i sums1 = _mm_loadu_si128(
            (const __m128i *)(input[block].sums + 8));
        const __m128i paired_sums = _mm_hadd_epi16(sums0, sums1);
        const __m128i mins = _mm_loadu_si128(
            (const __m128i *)minimum_scales);
        const __m128i minimum_products = _mm_madd_epi16(mins, paired_sums);
        accumulated_min = _mm_fmadd_ps(_mm_set1_ps(-dmin),
            _mm_cvtepi32_ps(minimum_products), accumulated_min);
#else
        total += d * weighted - dmin * minimum;
#endif
    }
#if defined(__AVX2__)
    accumulated_min = _mm_add_ps(accumulated_min,
                                  _mm_movehl_ps(accumulated_min, accumulated_min));
    accumulated_min = _mm_add_ss(accumulated_min,
                                  _mm_movehdup_ps(accumulated_min));
    return q38_sum_f32_8(accumulated) + _mm_cvtss_f32(accumulated_min);
#else
    return total;
#endif
}

static float q38_dot_q5_k_q8_k(const uint8_t *weights,
                               const Q38Q8KBlock *input,
                               uint64_t blocks)
{
#if defined(__AVX2__)
    __m256 accumulated = _mm256_setzero_ps();
    __m128 accumulated_min = _mm_setzero_ps();
#else
    float total = 0.0f;
#endif
    for (uint64_t block = 0; block < blocks; ++block, weights += 176) {
        const float d = q38_f16_to_f32(q38_load_u16(weights)) * input[block].scale;
        const float dmin = q38_f16_to_f32(q38_load_u16(weights + 2)) *
                           input[block].scale;
        const uint8_t *scales = weights + 4;
        const uint8_t *high = weights + 16;
        const uint8_t *low = weights + 48;
        int minimum = 0;
        uint8_t high0 = 1;
        uint8_t high1 = 2;
#if defined(__AVX2__)
        __m256i weighted_lanes = _mm256_setzero_si256();
#else
        int weighted = 0;
#endif
        for (int chunk = 0; chunk < 4; ++chunk) {
            uint8_t scale0, min0, scale1, min1;
            q38_scale_min_k4(chunk * 2, scales, &scale0, &min0);
            q38_scale_min_k4(chunk * 2 + 1, scales, &scale1, &min1);
#if defined(__AVX2__)
            const __m256i packed = _mm256_loadu_si256(
                (const __m256i *)(low + chunk * 32));
            const __m256i high_bits = _mm256_loadu_si256((const __m256i *)high);
            const __m256i nibble_mask = _mm256_set1_epi8(15);
            const __m256i zero = _mm256_setzero_si256();
            const __m256i sixteen = _mm256_set1_epi8(16);
            __m256i q0 = _mm256_and_si256(packed, nibble_mask);
            __m256i q1 = _mm256_and_si256(
                _mm256_srli_epi16(packed, 4), nibble_mask);
            const __m256i present0 = _mm256_cmpeq_epi8(
                _mm256_and_si256(high_bits, _mm256_set1_epi8((char)high0)), zero);
            const __m256i present1 = _mm256_cmpeq_epi8(
                _mm256_and_si256(high_bits, _mm256_set1_epi8((char)high1)), zero);
            q0 = _mm256_add_epi8(q0, _mm256_andnot_si256(present0, sixteen));
            q1 = _mm256_add_epi8(q1, _mm256_andnot_si256(present1, sixteen));
            const __m256i dot0 = q38_products_u8_s8_32(q0,
                _mm256_loadu_si256((const __m256i *)(
                    input[block].quants + chunk * 64)));
            const __m256i dot1 = q38_products_u8_s8_32(q1,
                _mm256_loadu_si256((const __m256i *)(
                    input[block].quants + chunk * 64 + 32)));
            weighted_lanes = _mm256_add_epi32(weighted_lanes,
                _mm256_mullo_epi32(dot0, _mm256_set1_epi32(scale0)));
            weighted_lanes = _mm256_add_epi32(weighted_lanes,
                _mm256_mullo_epi32(dot1, _mm256_set1_epi32(scale1)));
#else
            int dot0 = 0;
            int dot1 = 0;
            for (int i = 0; i < 32; ++i) {
                const int q0 = (low[chunk * 32 + i] & 15u) +
                               ((high[i] & high0) ? 16 : 0);
                const int q1 = (low[chunk * 32 + i] >> 4) +
                               ((high[i] & high1) ? 16 : 0);
                dot0 += q0 * input[block].quants[chunk * 64 + i];
                dot1 += q1 * input[block].quants[chunk * 64 + 32 + i];
            }
            weighted += scale0 * dot0 + scale1 * dot1;
#endif
            minimum += min0 * (input[block].sums[chunk * 4] +
                               input[block].sums[chunk * 4 + 1]);
            minimum += min1 * (input[block].sums[chunk * 4 + 2] +
                               input[block].sums[chunk * 4 + 3]);
            high0 <<= 2;
            high1 <<= 2;
        }
#if defined(__AVX2__)
        accumulated = _mm256_fmadd_ps(_mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes), accumulated);
        accumulated_min = _mm_fmadd_ss(_mm_set_ss(-dmin),
            _mm_set_ss((float)minimum), accumulated_min);
#else
        total += d * weighted - dmin * minimum;
#endif
    }
#if defined(__AVX2__)
    return q38_sum_f32_8(accumulated) + _mm_cvtss_f32(accumulated_min);
#else
    return total;
#endif
}

static float q38_dot_q6_k_q8_k(const uint8_t *weights,
                               const Q38Q8KBlock *input,
                               uint64_t blocks)
{
#if defined(__AVX2__)
    __m256 accumulated = _mm256_setzero_ps();
#else
    float total = 0.0f;
#endif
    for (uint64_t block = 0; block < blocks; ++block, weights += 210) {
        const uint8_t *low = weights;
        const uint8_t *high = weights + 128;
        const int8_t *scales = (const int8_t *)(weights + 192);
        const float d = q38_f16_to_f32(q38_load_u16(weights + 208)) *
                        input[block].scale;
#if defined(__AVX2__)
        const __m256i mask2 = _mm256_set1_epi8(3);
        const __m256i mask4 = _mm256_set1_epi8(15);
        const __m256i input_sums = _mm256_loadu_si256(
            (const __m256i *)input[block].sums);
        const __m128i packed_scales = _mm_loadu_si128(
            (const __m128i *)scales);
        const __m256i wide_scales = _mm256_cvtepi8_epi16(packed_scales);
        const __m256i offset_products = _mm256_slli_epi32(
            _mm256_madd_epi16(input_sums, wide_scales), 5);
        __m256i weighted_lanes = _mm256_setzero_si256();
#else
        int weighted = 0;
#endif
        for (int half = 0; half < 2; ++half) {
#if defined(__AVX2__)
            const __m256i low0 = _mm256_loadu_si256(
                (const __m256i *)(low + half * 64));
            const __m256i low1 = _mm256_loadu_si256(
                (const __m256i *)(low + half * 64 + 32));
            const __m256i upper = _mm256_loadu_si256(
                (const __m256i *)(high + half * 32));
            const __m256i q0 = _mm256_or_si256(
                _mm256_and_si256(low0, mask4),
                _mm256_slli_epi16(_mm256_and_si256(upper, mask2), 4));
            const __m256i q1 = _mm256_or_si256(
                _mm256_and_si256(low1, mask4),
                _mm256_slli_epi16(_mm256_and_si256(
                    _mm256_srli_epi16(upper, 2), mask2), 4));
            const __m256i q2 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(low0, 4), mask4),
                _mm256_slli_epi16(_mm256_and_si256(
                    _mm256_srli_epi16(upper, 4), mask2), 4));
            const __m256i q3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(low1, 4), mask4),
                _mm256_srli_epi16(_mm256_and_si256(
                    upper, _mm256_set1_epi8((char)0xc0)), 2));
            const int base = half * 128;
            __m256i product0 = _mm256_maddubs_epi16(q0,
                _mm256_loadu_si256((const __m256i *)(input[block].quants + base)));
            __m256i product1 = _mm256_maddubs_epi16(q1,
                _mm256_loadu_si256((const __m256i *)(input[block].quants + base + 32)));
            __m256i product2 = _mm256_maddubs_epi16(q2,
                _mm256_loadu_si256((const __m256i *)(input[block].quants + base + 64)));
            __m256i product3 = _mm256_maddubs_epi16(q3,
                _mm256_loadu_si256((const __m256i *)(input[block].quants + base + 96)));
            const int scale_index = half * 8;
            product0 = _mm256_madd_epi16(
                q38_q6_scale_pair(scales, scale_index), product0);
            product1 = _mm256_madd_epi16(
                q38_q6_scale_pair(scales, scale_index + 2), product1);
            product2 = _mm256_madd_epi16(
                q38_q6_scale_pair(scales, scale_index + 4), product2);
            product3 = _mm256_madd_epi16(
                q38_q6_scale_pair(scales, scale_index + 6), product3);
            weighted_lanes = _mm256_add_epi32(weighted_lanes,
                _mm256_add_epi32(product0, product1));
            weighted_lanes = _mm256_add_epi32(weighted_lanes,
                _mm256_add_epi32(product2, product3));
#else
            for (int i = 0; i < 32; ++i) {
                const int scale_index = half * 8 + i / 16;
                const int q0 = ((low[half * 64 + i] & 15u) |
                                ((high[half * 32 + i] & 3u) << 4)) - 32;
                const int q1 = ((low[half * 64 + i + 32] & 15u) |
                                (((high[half * 32 + i] >> 2) & 3u) << 4)) - 32;
                const int q2 = ((low[half * 64 + i] >> 4) |
                                (((high[half * 32 + i] >> 4) & 3u) << 4)) - 32;
                const int q3 = ((low[half * 64 + i + 32] >> 4) |
                                (((high[half * 32 + i] >> 6) & 3u) << 4)) - 32;
                const int base = half * 128;
                weighted += scales[scale_index] * q0 * input[block].quants[base + i];
                weighted += scales[scale_index + 2] * q1 * input[block].quants[base + 32 + i];
                weighted += scales[scale_index + 4] * q2 * input[block].quants[base + 64 + i];
                weighted += scales[scale_index + 6] * q3 * input[block].quants[base + 96 + i];
            }
#endif
        }
#if defined(__AVX2__)
        weighted_lanes = _mm256_sub_epi32(weighted_lanes, offset_products);
        accumulated = _mm256_fmadd_ps(_mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes), accumulated);
#else
        total += d * weighted;
#endif
    }
#if defined(__AVX2__)
    return q38_sum_f32_8(accumulated);
#else
    return total;
#endif
}

#if defined(__AVX2__)
#ifndef Q38_Q8_K_TILE
#define Q38_Q8_K_TILE 4u
#endif

static void q38_dot_q2_k_q8_k_tile(float output[Q38_Q8_K_TILE],
                                    const uint8_t *weights,
                                    const Q38Q8KBlock *input,
                                    uint32_t batch_size, uint64_t blocks)
{
    __m256 accumulated[Q38_Q8_K_TILE];
    float accumulated_min[Q38_Q8_K_TILE] = {0};
    for (uint32_t token = 0; token < batch_size; ++token)
        accumulated[token] = _mm256_setzero_ps();

    for (uint64_t block = 0; block < blocks; ++block, weights += 84) {
        const uint8_t *scales = weights;
        const uint8_t *quants = weights + 16;
        const float base_d = q38_f16_to_f32(q38_load_u16(weights + 80));
        const float base_dmin = q38_f16_to_f32(q38_load_u16(weights + 82));
        const __m256i mask = _mm256_set1_epi8(3);
        __m256i weighted[Q38_Q8_K_TILE];
        int minimum[Q38_Q8_K_TILE] = {0};
        for (uint32_t token = 0; token < batch_size; ++token)
            weighted[token] = _mm256_setzero_si256();

        for (int half = 0; half < 2; ++half) {
            const __m256i packed = _mm256_loadu_si256(
                (const __m256i *)(quants + half * 32));
            for (int field = 0; field < 4; ++field) {
                const int scale_index = half * 8 + field * 2;
                const int scale0 = scales[scale_index] & 15u;
                const int scale1 = scales[scale_index + 1] & 15u;
                const __m256i q = _mm256_and_si256(
                    _mm256_srli_epi16(packed, field * 2), mask);
                const __m256i scale = _mm256_set_m128i(
                    _mm_set1_epi32(scale1), _mm_set1_epi32(scale0));
                for (uint32_t token = 0; token < batch_size; ++token) {
                    const Q38Q8KBlock *activation = input +
                        (uint64_t)token * blocks + block;
                    const __m256i values = _mm256_loadu_si256(
                        (const __m256i *)(activation->quants +
                        half * 128 + field * 32));
                    const __m256i dot = q38_products_u8_s8_32(q, values);
                    weighted[token] = _mm256_add_epi32(weighted[token],
                        _mm256_mullo_epi32(dot, scale));
                    minimum[token] += (scales[scale_index] >> 4) *
                                      activation->sums[scale_index];
                    minimum[token] += (scales[scale_index + 1] >> 4) *
                                      activation->sums[scale_index + 1];
                }
            }
        }

        for (uint32_t token = 0; token < batch_size; ++token) {
            const Q38Q8KBlock *activation = input +
                (uint64_t)token * blocks + block;
            const float d = base_d * activation->scale;
            const float dmin = base_dmin * activation->scale;
            accumulated[token] = _mm256_fmadd_ps(_mm256_set1_ps(d),
                _mm256_cvtepi32_ps(weighted[token]), accumulated[token]);
            accumulated_min[token] = fmaf(-dmin, (float)minimum[token],
                                           accumulated_min[token]);
        }
    }

    for (uint32_t token = 0; token < batch_size; ++token)
        output[token] = q38_sum_f32_8(accumulated[token]) +
                        accumulated_min[token];
}

static void q38_dot_q3_k_q8_k_tile(float output[Q38_Q8_K_TILE],
                                    const uint8_t *weights,
                                    const Q38Q8KBlock *input,
                                    uint32_t batch_size, uint64_t blocks)
{
    __m256 accumulated[Q38_Q8_K_TILE];
    for (uint32_t token = 0; token < batch_size; ++token)
        accumulated[token] = _mm256_setzero_ps();

    for (uint64_t block = 0; block < blocks; ++block, weights += 110) {
        const uint8_t *high = weights;
        const uint8_t *low = weights + 32;
        int8_t scales[16];
        q38_unpack_q3_scales(weights + 96, scales);
        const float base_d = q38_f16_to_f32(q38_load_u16(weights + 108));
        const __m256i hbits = _mm256_loadu_si256((const __m256i *)high);
        const __m256i mask = _mm256_set1_epi8(3);
        const __m256i four = _mm256_set1_epi8(4);
        const __m256i zero = _mm256_setzero_si256();
        __m256i weighted[Q38_Q8_K_TILE];
        for (uint32_t token = 0; token < batch_size; ++token)
            weighted[token] = _mm256_setzero_si256();

        int group = 0;
        for (int half = 0; half < 2; ++half) {
            const __m256i packed = _mm256_loadu_si256(
                (const __m256i *)(low + half * 32));
            for (int field = 0; field < 4; ++field, ++group) {
                const uint8_t high_bit = (uint8_t)(1u << group);
                const int scale0 = scales[group * 2];
                const int scale1 = scales[group * 2 + 1];
                const __m256i low_values = _mm256_and_si256(
                    _mm256_srli_epi16(packed, field * 2), mask);
                const __m256i absent = _mm256_cmpeq_epi8(
                    _mm256_and_si256(hbits,
                        _mm256_set1_epi8((char)high_bit)), zero);
                const __m256i q = _mm256_sub_epi8(low_values,
                    _mm256_and_si256(absent, four));
                const __m256i scale = _mm256_set_m128i(
                    _mm_set1_epi32(scale1), _mm_set1_epi32(scale0));
                for (uint32_t token = 0; token < batch_size; ++token) {
                    const Q38Q8KBlock *activation = input +
                        (uint64_t)token * blocks + block;
                    const __m256i values = _mm256_loadu_si256(
                        (const __m256i *)(activation->quants +
                        half * 128 + field * 32));
                    const __m256i dot = q38_products_s8_s8_32(q, values);
                    weighted[token] = _mm256_add_epi32(weighted[token],
                        _mm256_mullo_epi32(dot, scale));
                }
            }
        }

        for (uint32_t token = 0; token < batch_size; ++token) {
            const Q38Q8KBlock *activation = input +
                (uint64_t)token * blocks + block;
            const float d = base_d * activation->scale;
            accumulated[token] = _mm256_fmadd_ps(_mm256_set1_ps(d),
                _mm256_cvtepi32_ps(weighted[token]), accumulated[token]);
        }
    }

    for (uint32_t token = 0; token < batch_size; ++token)
        output[token] = q38_sum_f32_8(accumulated[token]);
}

static void q38_dot_iq2_s_q8_k_tile(float output[Q38_Q8_K_TILE],
                                     const uint8_t *weights,
                                     const Q38Q8KBlock *input,
                                     uint32_t batch_size, uint64_t blocks)
{
    static const uint8_t sign_source[32] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3
    };
    static const uint8_t sign_bit[32] = {
        1, 2, 4, 8, 16, 32, 64, 128,
        1, 2, 4, 8, 16, 32, 64, 128,
        1, 2, 4, 8, 16, 32, 64, 128,
        1, 2, 4, 8, 16, 32, 64, 128
    };
    const __m256i source_mask = _mm256_loadu_si256(
        (const __m256i *)sign_source);
    const __m256i bit_mask = _mm256_loadu_si256(
        (const __m256i *)sign_bit);
    __m256 accumulated[Q38_Q8_K_TILE];
    for (uint32_t token = 0; token < batch_size; ++token)
        accumulated[token] = _mm256_setzero_ps();

    for (uint64_t block = 0; block < blocks; ++block, weights += 82) {
        const float base_d = 0.125f *
            q38_f16_to_f32(q38_load_u16(weights));
        const uint8_t *indices = weights + 2;
        const uint8_t *signs = weights + 34;
        const uint8_t *high = weights + 66;
        const uint8_t *scales = weights + 74;
        __m256i weighted[Q38_Q8_K_TILE];
        for (uint32_t token = 0; token < batch_size; ++token)
            weighted[token] = _mm256_setzero_si256();

        for (int group = 0; group < 8; ++group) {
            const int scale0 = 1 + 2 * (scales[group] & 15u);
            const int scale1 = 1 + 2 * (scales[group] >> 4);
            const __m256i quant = _mm256_set_epi64x(
                (long long)iq2s_grid[indices[3] |
                    ((high[group] << 2) & 0x300)],
                (long long)iq2s_grid[indices[2] |
                    ((high[group] << 4) & 0x300)],
                (long long)iq2s_grid[indices[1] |
                    ((high[group] << 6) & 0x300)],
                (long long)iq2s_grid[indices[0] |
                    ((high[group] << 8) & 0x300)]);
            const uint32_t packed_signs = (uint32_t)signs[0] |
                (uint32_t)signs[1] << 8 | (uint32_t)signs[2] << 16 |
                (uint32_t)signs[3] << 24;
            __m256i negative = _mm256_set1_epi32((int)packed_signs);
            negative = _mm256_and_si256(
                _mm256_shuffle_epi8(negative, source_mask), bit_mask);
            negative = _mm256_cmpeq_epi8(negative, bit_mask);
            const __m256i scale = _mm256_set_m128i(
                _mm_set1_epi32(scale1), _mm_set1_epi32(scale0));

            for (uint32_t token = 0; token < batch_size; ++token) {
                const Q38Q8KBlock *activation = input +
                    (uint64_t)token * blocks + block;
                const __m256i values = _mm256_loadu_si256(
                    (const __m256i *)(activation->quants + group * 32));
                const __m256i signed_activation = _mm256_sub_epi8(
                    _mm256_xor_si256(negative, values), negative);
                const __m256i dot = q38_products_u8_s8_32(
                    quant, signed_activation);
                weighted[token] = _mm256_add_epi32(weighted[token],
                    _mm256_mullo_epi32(dot, scale));
            }
            indices += 4;
            signs += 4;
        }

        for (uint32_t token = 0; token < batch_size; ++token) {
            const Q38Q8KBlock *activation = input +
                (uint64_t)token * blocks + block;
            const float d = base_d * activation->scale;
            accumulated[token] = _mm256_fmadd_ps(_mm256_set1_ps(d),
                _mm256_cvtepi32_ps(weighted[token]), accumulated[token]);
        }
    }

    for (uint32_t token = 0; token < batch_size; ++token)
        output[token] = q38_sum_f32_8(accumulated[token]);
}

static void q38_dot_iq2_xxs_q8_k_tile(float output[Q38_Q8_K_TILE],
                                       const uint8_t *weights,
                                       const Q38Q8KBlock *input,
                                       uint32_t batch_size, uint64_t blocks)
{
    __m256 accumulated[Q38_Q8_K_TILE];
    for (uint32_t token = 0; token < batch_size; ++token)
        accumulated[token] = _mm256_setzero_ps();

    for (uint64_t block = 0; block < blocks; ++block, weights += 66) {
        const float base_d = 0.125f *
            q38_f16_to_f32(q38_load_u16(weights));
        const uint8_t *codes = weights + 2;
        __m256i weighted[Q38_Q8_K_TILE];
        for (uint32_t token = 0; token < batch_size; ++token)
            weighted[token] = _mm256_setzero_si256();

        for (int group = 0; group < 8; ++group, codes += 8) {
            const uint32_t metadata = q38_load_u32(codes + 4);
            const int scale = 1 + 2 * (int)(metadata >> 28);
            const __m256i quant = _mm256_set_epi64x(
                (long long)iq2xxs_grid[codes[3]],
                (long long)iq2xxs_grid[codes[2]],
                (long long)iq2xxs_grid[codes[1]],
                (long long)iq2xxs_grid[codes[0]]);
            const __m256i negative = q38_iq_sign_mask(
                metadata & 127u, (metadata >> 7) & 127u,
                (metadata >> 14) & 127u, (metadata >> 21) & 127u);
            const __m256i scale_vector = _mm256_set1_epi32(scale);
            for (uint32_t token = 0; token < batch_size; ++token) {
                const Q38Q8KBlock *activation = input +
                    (uint64_t)token * blocks + block;
                const __m256i values = _mm256_loadu_si256(
                    (const __m256i *)(activation->quants + group * 32));
                const __m256i signed_activation = _mm256_sub_epi8(
                    _mm256_xor_si256(negative, values), negative);
                const __m256i dot = q38_products_u8_s8_32(
                    quant, signed_activation);
                weighted[token] = _mm256_add_epi32(weighted[token],
                    _mm256_mullo_epi32(dot, scale_vector));
            }
        }

        for (uint32_t token = 0; token < batch_size; ++token) {
            const Q38Q8KBlock *activation = input +
                (uint64_t)token * blocks + block;
            const float d = base_d * activation->scale;
            accumulated[token] = _mm256_fmadd_ps(_mm256_set1_ps(d),
                _mm256_cvtepi32_ps(weighted[token]), accumulated[token]);
        }
    }

    for (uint32_t token = 0; token < batch_size; ++token)
        output[token] = q38_sum_f32_8(accumulated[token]);
}

static void q38_dot_iq2_xs_q8_k_tile(float output[Q38_Q8_K_TILE],
                                      const uint8_t *weights,
                                      const Q38Q8KBlock *input,
                                      uint32_t batch_size, uint64_t blocks)
{
    __m256 accumulated[Q38_Q8_K_TILE];
    for (uint32_t token = 0; token < batch_size; ++token)
        accumulated[token] = _mm256_setzero_ps();

    for (uint64_t block = 0; block < blocks; ++block, weights += 74) {
        const float base_d = 0.125f *
            q38_f16_to_f32(q38_load_u16(weights));
        const uint8_t *codes = weights + 2;
        const uint8_t *scales = weights + 66;
        __m256i weighted[Q38_Q8_K_TILE];
        for (uint32_t token = 0; token < batch_size; ++token)
            weighted[token] = _mm256_setzero_si256();

        for (int group = 0; group < 8; ++group, codes += 8) {
            uint16_t code[4];
            for (int section = 0; section < 4; ++section)
                code[section] = q38_load_u16(codes + section * 2);
            const int scale0 = 1 + 2 * (scales[group] & 15u);
            const int scale1 = 1 + 2 * (scales[group] >> 4);
            const __m256i quant = _mm256_set_epi64x(
                (long long)iq2xs_grid[code[3] & 511u],
                (long long)iq2xs_grid[code[2] & 511u],
                (long long)iq2xs_grid[code[1] & 511u],
                (long long)iq2xs_grid[code[0] & 511u]);
            const __m256i negative = q38_iq_sign_mask(
                code[0] >> 9, code[1] >> 9, code[2] >> 9, code[3] >> 9);
            const __m256i scale = _mm256_set_m128i(
                _mm_set1_epi32(scale1), _mm_set1_epi32(scale0));
            for (uint32_t token = 0; token < batch_size; ++token) {
                const Q38Q8KBlock *activation = input +
                    (uint64_t)token * blocks + block;
                const __m256i values = _mm256_loadu_si256(
                    (const __m256i *)(activation->quants + group * 32));
                const __m256i signed_activation = _mm256_sub_epi8(
                    _mm256_xor_si256(negative, values), negative);
                const __m256i dot = q38_products_u8_s8_32(
                    quant, signed_activation);
                weighted[token] = _mm256_add_epi32(weighted[token],
                    _mm256_mullo_epi32(dot, scale));
            }
        }

        for (uint32_t token = 0; token < batch_size; ++token) {
            const Q38Q8KBlock *activation = input +
                (uint64_t)token * blocks + block;
            const float d = base_d * activation->scale;
            accumulated[token] = _mm256_fmadd_ps(_mm256_set1_ps(d),
                _mm256_cvtepi32_ps(weighted[token]), accumulated[token]);
        }
    }

    for (uint32_t token = 0; token < batch_size; ++token)
        output[token] = q38_sum_f32_8(accumulated[token]);
}

static void q38_dot_iq3_xxs_q8_k_tile(float output[Q38_Q8_K_TILE],
                                       const uint8_t *weights,
                                       const Q38Q8KBlock *input,
                                       uint32_t batch_size, uint64_t blocks)
{
    __m256 accumulated[Q38_Q8_K_TILE];
    for (uint32_t token = 0; token < batch_size; ++token)
        accumulated[token] = _mm256_setzero_ps();

    for (uint64_t block = 0; block < blocks; ++block, weights += 98) {
        const float base_d = 0.25f *
            q38_f16_to_f32(q38_load_u16(weights));
        const uint8_t *indices = weights + 2;
        const uint8_t *metadata = weights + 66;
        __m256i weighted[Q38_Q8_K_TILE];
        for (uint32_t token = 0; token < batch_size; ++token)
            weighted[token] = _mm256_setzero_si256();

        for (int group = 0; group < 8; ++group) {
            const uint32_t packed = q38_load_u32(metadata + group * 4);
            const int scale = 1 + 2 * (int)(packed >> 28);
            const uint8_t *code = indices + group * 8;
            const __m256i quant = _mm256_set_epi32(
                (int)iq3xxs_grid[code[7]], (int)iq3xxs_grid[code[6]],
                (int)iq3xxs_grid[code[5]], (int)iq3xxs_grid[code[4]],
                (int)iq3xxs_grid[code[3]], (int)iq3xxs_grid[code[2]],
                (int)iq3xxs_grid[code[1]], (int)iq3xxs_grid[code[0]]);
            const __m256i negative = q38_iq_sign_mask(
                packed & 127u, (packed >> 7) & 127u,
                (packed >> 14) & 127u, (packed >> 21) & 127u);
            const __m256i scale_vector = _mm256_set1_epi32(scale);
            for (uint32_t token = 0; token < batch_size; ++token) {
                const Q38Q8KBlock *activation = input +
                    (uint64_t)token * blocks + block;
                const __m256i values = _mm256_loadu_si256(
                    (const __m256i *)(activation->quants + group * 32));
                const __m256i signed_activation = _mm256_sub_epi8(
                    _mm256_xor_si256(negative, values), negative);
                const __m256i dot = q38_products_u8_s8_32(
                    quant, signed_activation);
                weighted[token] = _mm256_add_epi32(weighted[token],
                    _mm256_mullo_epi32(dot, scale_vector));
            }
        }

        for (uint32_t token = 0; token < batch_size; ++token) {
            const Q38Q8KBlock *activation = input +
                (uint64_t)token * blocks + block;
            const float d = base_d * activation->scale;
            accumulated[token] = _mm256_fmadd_ps(_mm256_set1_ps(d),
                _mm256_cvtepi32_ps(weighted[token]), accumulated[token]);
        }
    }

    for (uint32_t token = 0; token < batch_size; ++token)
        output[token] = q38_sum_f32_8(accumulated[token]);
}

static void q38_dot_iq3_s_q8_k_tile(float output[Q38_Q8_K_TILE],
                                     const uint8_t *weights,
                                     const Q38Q8KBlock *input,
                                     uint32_t batch_size, uint64_t blocks)
{
    __m256 accumulated[Q38_Q8_K_TILE];
    for (uint32_t token = 0; token < batch_size; ++token)
        accumulated[token] = _mm256_setzero_ps();

    for (uint64_t block = 0; block < blocks; ++block, weights += 110) {
        const float base_d = q38_f16_to_f32(q38_load_u16(weights));
        const uint8_t *indices = weights + 2;
        const uint8_t *high = weights + 66;
        const uint8_t *signs = weights + 74;
        const uint8_t *scales = weights + 106;
        __m256i weighted[Q38_Q8_K_TILE];
        for (uint32_t token = 0; token < batch_size; ++token)
            weighted[token] = _mm256_setzero_si256();

        for (int group = 0; group < 8; group += 2) {
            __m256i quant[2];
            q38_iq3_s_grid_vectors(quant, indices + group * 8, high + group);
            for (int pair = 0; pair < 2; ++pair) {
                const int current = group + pair;
                const __m256i negative = q38_raw_iq_sign_mask(
                    signs + current * 4);
                const int scale = 1 + 2 * ((scales[group / 2] >>
                    (4 * pair)) & 15u);
                const __m256i scale_vector = _mm256_set1_epi32(scale);
                for (uint32_t token = 0; token < batch_size; ++token) {
                    const Q38Q8KBlock *activation = input +
                        (uint64_t)token * blocks + block;
                    const __m256i values = _mm256_loadu_si256(
                        (const __m256i *)(activation->quants + current * 32));
                    const __m256i signed_activation = _mm256_sub_epi8(
                        _mm256_xor_si256(negative, values), negative);
                    const __m256i dot = q38_products_u8_s8_32(
                        quant[pair], signed_activation);
                    weighted[token] = _mm256_add_epi32(weighted[token],
                        _mm256_mullo_epi32(dot, scale_vector));
                }
            }
        }

        for (uint32_t token = 0; token < batch_size; ++token) {
            const Q38Q8KBlock *activation = input +
                (uint64_t)token * blocks + block;
            const float d = base_d * activation->scale;
            accumulated[token] = _mm256_fmadd_ps(_mm256_set1_ps(d),
                _mm256_cvtepi32_ps(weighted[token]), accumulated[token]);
        }
    }

    for (uint32_t token = 0; token < batch_size; ++token)
        output[token] = q38_sum_f32_8(accumulated[token]);
}

#define Q38_DEFINE_IQ_TILE(format) \
static void Q38_UNUSED_HELPER q38_dot_##format##_q8_k_tile( \
                                         float output[Q38_Q8_K_TILE], \
                                         const uint8_t *weights, \
                                         const Q38Q8KBlock *input, \
                                         uint32_t batch_size, uint64_t blocks) \
{ \
    for (uint32_t token = 0; token < batch_size; ++token) \
        output[token] = q38_dot_##format##_q8_k( \
            weights, input + (uint64_t)token * blocks, blocks); \
}

Q38_DEFINE_IQ_TILE(iq4_xs)
Q38_DEFINE_IQ_TILE(iq4_nl)
Q38_DEFINE_IQ_TILE(iq1_s)
Q38_DEFINE_IQ_TILE(iq1_m)

#undef Q38_DEFINE_IQ_TILE

static void q38_dot_iq1_s_q8_k_fused_tile(float output[Q38_Q8_K_TILE],
                                           const uint8_t *weights,
                                           const Q38Q8KBlock *input,
                                           uint32_t batch_size,
                                           uint64_t blocks)
{
    __m256 accumulated[Q38_Q8_K_TILE];
    float accumulated_correction[Q38_Q8_K_TILE] = {0};
    for (uint32_t token = 0; token < batch_size; ++token)
        accumulated[token] = _mm256_setzero_ps();

    for (uint64_t block = 0; block < blocks; ++block, weights += 50) {
        const float base_d = q38_f16_to_f32(q38_load_u16(weights));
        const uint8_t *indices = weights + 2;
        const uint8_t *high = weights + 34;
        __m256i weighted[Q38_Q8_K_TILE];
        int correction[Q38_Q8_K_TILE] = {0};
        for (uint32_t token = 0; token < batch_size; ++token)
            weighted[token] = _mm256_setzero_si256();

        for (int group = 0; group < 8; group += 2) {
            const uint16_t high0 = q38_load_u16(high + group * 2);
            const uint16_t high1 = q38_load_u16(high + (group + 1) * 2);
            const __m256i quant0 = q38_iq1_s_grid32(indices, high0);
            const __m256i quant1 = q38_iq1_s_grid32(indices + 4, high1);
            const int scale0 = 1 + 2 * ((high0 >> 12) & 7u);
            const int scale1 = 1 + 2 * ((high1 >> 12) & 7u);
            for (uint32_t token = 0; token < batch_size; ++token) {
                const Q38Q8KBlock *activation = input +
                    (uint64_t)token * blocks + block;
                const __m256i activation0 = _mm256_loadu_si256(
                    (const __m256i *)(activation->quants + group * 32));
                const __m256i activation1 = _mm256_loadu_si256(
                    (const __m256i *)(activation->quants +
                                      (group + 1) * 32));
                weighted[token] = _mm256_add_epi32(weighted[token],
                    q38_products_s8_s8_scaled_32(quant0, activation0,
                                                 scale0));
                weighted[token] = _mm256_add_epi32(weighted[token],
                    q38_products_s8_s8_scaled_32(quant1, activation1,
                                                 scale1));
                correction[token] += scale0 *
                    (high0 & 0x8000u ? -1 : 1) *
                    (activation->sums[group * 2] +
                     activation->sums[group * 2 + 1]);
                correction[token] += scale1 *
                    (high1 & 0x8000u ? -1 : 1) *
                    (activation->sums[(group + 1) * 2] +
                     activation->sums[(group + 1) * 2 + 1]);
            }
            indices += 8;
        }

        for (uint32_t token = 0; token < batch_size; ++token) {
            const Q38Q8KBlock *activation = input +
                (uint64_t)token * blocks + block;
            const float d = base_d * activation->scale;
            accumulated[token] = _mm256_fmadd_ps(_mm256_set1_ps(d),
                _mm256_cvtepi32_ps(weighted[token]), accumulated[token]);
            accumulated_correction[token] += d * correction[token];
        }
    }

    for (uint32_t token = 0; token < batch_size; ++token)
        output[token] = q38_sum_f32_8(accumulated[token]) +
                        0.125f * accumulated_correction[token];
}

static void q38_dot_iq1_m_q8_k_fused_tile(float output[Q38_Q8_K_TILE],
                                           const uint8_t *weights,
                                           const Q38Q8KBlock *input,
                                           uint32_t batch_size,
                                           uint64_t blocks)
{
    float total[Q38_Q8_K_TILE] = {0};
    const __m256i one = _mm256_set1_epi8(1);
    for (uint64_t block = 0; block < blocks; ++block, weights += 56) {
        const uint8_t *indices = weights;
        const uint8_t *high = weights + 32;
        const uint8_t *packed_scales = weights + 48;
        const uint16_t scale_bits = (q38_load_u16(packed_scales) >> 12) |
             ((q38_load_u16(packed_scales + 2) >> 8) & 0x00f0u) |
             ((q38_load_u16(packed_scales + 4) >> 4) & 0x0f00u) |
             (q38_load_u16(packed_scales + 6) & 0xf000u);
        const float base_d = q38_f16_to_f32(scale_bits);
        __m256i weighted[Q38_Q8_K_TILE];
        __m256i correction[Q38_Q8_K_TILE];
        for (uint32_t token = 0; token < batch_size; ++token) {
            weighted[token] = _mm256_setzero_si256();
            correction[token] = _mm256_setzero_si256();
        }

        for (int group = 0; group < 8; group += 2) {
#if defined(__BMI2__)
            const uint64_t packed0 =
                _pdep_u64(q38_load_u32(indices), 0x00ff00ff00ff00ffULL) |
                _pdep_u64(q38_load_u16(high) & 0x7777u,
                          0x0f000f000f000f00ULL);
            const uint64_t packed1 =
                _pdep_u64(q38_load_u32(indices + 4),
                          0x00ff00ff00ff00ffULL) |
                _pdep_u64(q38_load_u16(high + 2) & 0x7777u,
                          0x0f000f000f000f00ULL);
            const __m256i quant0 = _mm256_set_epi64x(
                (long long)iq1s_grid[(uint16_t)(packed0 >> 48)],
                (long long)iq1s_grid[(uint16_t)(packed0 >> 32)],
                (long long)iq1s_grid[(uint16_t)(packed0 >> 16)],
                (long long)iq1s_grid[(uint16_t)packed0]);
            const __m256i quant1 = _mm256_set_epi64x(
                (long long)iq1s_grid[(uint16_t)(packed1 >> 48)],
                (long long)iq1s_grid[(uint16_t)(packed1 >> 32)],
                (long long)iq1s_grid[(uint16_t)(packed1 >> 16)],
                (long long)iq1s_grid[(uint16_t)packed1]);
            const uint64_t delta_sign = _pdep_u64(
                q38_load_u32(high) & 0x88888888u,
                0xf0f0f0f0f0f0f0f0ULL);
            const __m256i delta0 = _mm256_or_si256(one,
                _mm256_cvtepi8_epi64(_mm_set1_epi32((int)delta_sign)));
            const __m256i delta1 = _mm256_or_si256(one,
                _mm256_cvtepi8_epi64(
                    _mm_set1_epi32((int)(delta_sign >> 32))));
#else
            const __m256i quant0 = _mm256_set_epi64x(
                (long long)iq1s_grid[indices[3] |
                    (((uint16_t)high[1] << 4) & 0x700)],
                (long long)iq1s_grid[indices[2] |
                    (((uint16_t)high[1] << 8) & 0x700)],
                (long long)iq1s_grid[indices[1] |
                    (((uint16_t)high[0] << 4) & 0x700)],
                (long long)iq1s_grid[indices[0] |
                    (((uint16_t)high[0] << 8) & 0x700)]);
            const __m256i quant1 = _mm256_set_epi64x(
                (long long)iq1s_grid[indices[7] |
                    (((uint16_t)high[3] << 4) & 0x700)],
                (long long)iq1s_grid[indices[6] |
                    (((uint16_t)high[3] << 8) & 0x700)],
                (long long)iq1s_grid[indices[5] |
                    (((uint16_t)high[2] << 4) & 0x700)],
                (long long)iq1s_grid[indices[4] |
                    (((uint16_t)high[2] << 8) & 0x700)]);
            const __m256i delta0 = _mm256_set_epi64x(
                high[1] & 0x80u ? -1LL : 0x0101010101010101LL,
                high[1] & 0x08u ? -1LL : 0x0101010101010101LL,
                high[0] & 0x80u ? -1LL : 0x0101010101010101LL,
                high[0] & 0x08u ? -1LL : 0x0101010101010101LL);
            const __m256i delta1 = _mm256_set_epi64x(
                high[3] & 0x80u ? -1LL : 0x0101010101010101LL,
                high[3] & 0x08u ? -1LL : 0x0101010101010101LL,
                high[2] & 0x80u ? -1LL : 0x0101010101010101LL,
                high[2] & 0x08u ? -1LL : 0x0101010101010101LL);
#endif
            const uint16_t scales = q38_load_u16(
                packed_scales + (group / 2) * 2);
            const int scale0 = 1 + 2 * ((scales >> 0) & 7u);
            const int scale1 = 1 + 2 * ((scales >> 3) & 7u);
            const int scale2 = 1 + 2 * ((scales >> 6) & 7u);
            const int scale3 = 1 + 2 * ((scales >> 9) & 7u);
            const __m256i scales0 = _mm256_set_m128i(
                _mm_set1_epi16((short)scale1),
                _mm_set1_epi16((short)scale0));
            const __m256i scales1 = _mm256_set_m128i(
                _mm_set1_epi16((short)scale3),
                _mm_set1_epi16((short)scale2));
            for (uint32_t token = 0; token < batch_size; ++token) {
                const Q38Q8KBlock *activation = input +
                    (uint64_t)token * blocks + block;
                const __m256i activation0 = _mm256_loadu_si256(
                    (const __m256i *)(activation->quants + group * 32));
                const __m256i activation1 = _mm256_loadu_si256(
                    (const __m256i *)(activation->quants +
                                      (group + 1) * 32));
                const __m256i dot0 = _mm256_maddubs_epi16(
                    _mm256_abs_epi8(quant0),
                    _mm256_sign_epi8(activation0, quant0));
                const __m256i dot1 = _mm256_maddubs_epi16(
                    _mm256_abs_epi8(quant1),
                    _mm256_sign_epi8(activation1, quant1));
                const __m256i correction0 = _mm256_maddubs_epi16(
                    one, _mm256_sign_epi8(activation0, delta0));
                const __m256i correction1 = _mm256_maddubs_epi16(
                    one, _mm256_sign_epi8(activation1, delta1));
                weighted[token] = _mm256_add_epi32(weighted[token],
                    _mm256_add_epi32(_mm256_madd_epi16(dot0, scales0),
                                     _mm256_madd_epi16(dot1, scales1)));
                correction[token] = _mm256_add_epi32(correction[token],
                    _mm256_add_epi32(
                        _mm256_madd_epi16(correction0, scales0),
                        _mm256_madd_epi16(correction1, scales1)));
            }
            indices += 8;
            high += 4;
        }

        for (uint32_t token = 0; token < batch_size; ++token) {
            const Q38Q8KBlock *activation = input +
                (uint64_t)token * blocks + block;
            const int weighted_sum = q38_sum_i32_8(weighted[token]);
            const int correction_sum = q38_sum_i32_8(correction[token]);
            const float d = base_d * activation->scale;
            total[token] += d *
                (weighted_sum + 0.125f * correction_sum);
        }
    }
    for (uint32_t token = 0; token < batch_size; ++token)
        output[token] = total[token];
}

static void q38_dot_q4_k_q8_k_tile(float output[Q38_Q8_K_TILE],
                                    const uint8_t *weights,
                                    const Q38Q8KBlock *input,
                                    uint32_t batch_size, uint64_t blocks)
{
    __m256 accumulated[Q38_Q8_K_TILE];
    __m128 accumulated_min[Q38_Q8_K_TILE];
    for (uint32_t token = 0; token < batch_size; ++token) {
        accumulated[token] = _mm256_setzero_ps();
        accumulated_min[token] = _mm_setzero_ps();
    }
    for (uint64_t block = 0; block < blocks; ++block, weights += 144) {
        const float base_d = q38_f16_to_f32(q38_load_u16(weights));
        const float base_dmin = q38_f16_to_f32(q38_load_u16(weights + 2));
        const uint8_t *scales = weights + 4;
        const uint8_t *quants = weights + 16;
        __m256i weighted[Q38_Q8_K_TILE];
        int16_t minimum_scales[8];
        for (uint32_t token = 0; token < batch_size; ++token)
            weighted[token] = _mm256_setzero_si256();
        for (int chunk = 0; chunk < 4; ++chunk) {
            uint8_t scale0, min0, scale1, min1;
            q38_scale_min_k4(chunk * 2, scales, &scale0, &min0);
            q38_scale_min_k4(chunk * 2 + 1, scales, &scale1, &min1);
            const __m256i packed = _mm256_loadu_si256(
                (const __m256i *)(quants + chunk * 32));
            const __m256i mask = _mm256_set1_epi8(15);
            const __m256i low = _mm256_and_si256(packed, mask);
            const __m256i high = _mm256_and_si256(
                _mm256_srli_epi16(packed, 4), mask);
            for (uint32_t token = 0; token < batch_size; ++token) {
                const Q38Q8KBlock *activation = input +
                    (uint64_t)token * blocks + block;
                const __m256i dot0 = q38_products_u8_s8_32(low,
                    _mm256_loadu_si256((const __m256i *)(
                        activation->quants + chunk * 64)));
                const __m256i dot1 = q38_products_u8_s8_32(high,
                    _mm256_loadu_si256((const __m256i *)(
                        activation->quants + chunk * 64 + 32)));
                weighted[token] = _mm256_add_epi32(weighted[token],
                    _mm256_mullo_epi32(dot0, _mm256_set1_epi32(scale0)));
                weighted[token] = _mm256_add_epi32(weighted[token],
                    _mm256_mullo_epi32(dot1, _mm256_set1_epi32(scale1)));
            }
            minimum_scales[chunk * 2] = min0;
            minimum_scales[chunk * 2 + 1] = min1;
        }
        const __m128i mins = _mm_loadu_si128(
            (const __m128i *)minimum_scales);
        for (uint32_t token = 0; token < batch_size; ++token) {
            const Q38Q8KBlock *activation = input +
                (uint64_t)token * blocks + block;
            const float d = base_d * activation->scale;
            const float dmin = base_dmin * activation->scale;
            accumulated[token] = _mm256_fmadd_ps(_mm256_set1_ps(d),
                _mm256_cvtepi32_ps(weighted[token]), accumulated[token]);
            const __m128i sums0 = _mm_loadu_si128(
                (const __m128i *)activation->sums);
            const __m128i sums1 = _mm_loadu_si128(
                (const __m128i *)(activation->sums + 8));
            const __m128i paired_sums = _mm_hadd_epi16(sums0, sums1);
            const __m128i minimum_products = _mm_madd_epi16(mins, paired_sums);
            accumulated_min[token] = _mm_fmadd_ps(_mm_set1_ps(-dmin),
                _mm_cvtepi32_ps(minimum_products), accumulated_min[token]);
        }
    }
    for (uint32_t token = 0; token < batch_size; ++token) {
        __m128 minimum = _mm_add_ps(accumulated_min[token],
            _mm_movehl_ps(accumulated_min[token], accumulated_min[token]));
        minimum = _mm_add_ss(minimum, _mm_movehdup_ps(minimum));
        output[token] = q38_sum_f32_8(accumulated[token]) +
                        _mm_cvtss_f32(minimum);
    }
}

static void q38_dot_q5_k_q8_k_tile(float output[Q38_Q8_K_TILE],
                                    const uint8_t *weights,
                                    const Q38Q8KBlock *input,
                                    uint32_t batch_size, uint64_t blocks)
{
    __m256 accumulated[Q38_Q8_K_TILE];
    __m128 accumulated_min[Q38_Q8_K_TILE];
    for (uint32_t token = 0; token < batch_size; ++token) {
        accumulated[token] = _mm256_setzero_ps();
        accumulated_min[token] = _mm_setzero_ps();
    }
    for (uint64_t block = 0; block < blocks; ++block, weights += 176) {
        const float base_d = q38_f16_to_f32(q38_load_u16(weights));
        const float base_dmin = q38_f16_to_f32(q38_load_u16(weights + 2));
        const uint8_t *scales = weights + 4;
        const uint8_t *high = weights + 16;
        const uint8_t *low = weights + 48;
        __m256i weighted[Q38_Q8_K_TILE];
        int minimum[Q38_Q8_K_TILE] = {0};
        uint8_t high0 = 1;
        uint8_t high1 = 2;
        for (uint32_t token = 0; token < batch_size; ++token)
            weighted[token] = _mm256_setzero_si256();
        const __m256i high_bits = _mm256_loadu_si256((const __m256i *)high);
        const __m256i nibble_mask = _mm256_set1_epi8(15);
        const __m256i zero = _mm256_setzero_si256();
        const __m256i sixteen = _mm256_set1_epi8(16);
        for (int chunk = 0; chunk < 4; ++chunk) {
            uint8_t scale0, min0, scale1, min1;
            q38_scale_min_k4(chunk * 2, scales, &scale0, &min0);
            q38_scale_min_k4(chunk * 2 + 1, scales, &scale1, &min1);
            const __m256i packed = _mm256_loadu_si256(
                (const __m256i *)(low + chunk * 32));
            __m256i q0 = _mm256_and_si256(packed, nibble_mask);
            __m256i q1 = _mm256_and_si256(
                _mm256_srli_epi16(packed, 4), nibble_mask);
            const __m256i present0 = _mm256_cmpeq_epi8(
                _mm256_and_si256(high_bits,
                    _mm256_set1_epi8((char)high0)), zero);
            const __m256i present1 = _mm256_cmpeq_epi8(
                _mm256_and_si256(high_bits,
                    _mm256_set1_epi8((char)high1)), zero);
            q0 = _mm256_add_epi8(q0,
                _mm256_andnot_si256(present0, sixteen));
            q1 = _mm256_add_epi8(q1,
                _mm256_andnot_si256(present1, sixteen));
            for (uint32_t token = 0; token < batch_size; ++token) {
                const Q38Q8KBlock *activation = input +
                    (uint64_t)token * blocks + block;
                const __m256i dot0 = q38_products_u8_s8_32(q0,
                    _mm256_loadu_si256((const __m256i *)(
                        activation->quants + chunk * 64)));
                const __m256i dot1 = q38_products_u8_s8_32(q1,
                    _mm256_loadu_si256((const __m256i *)(
                        activation->quants + chunk * 64 + 32)));
                weighted[token] = _mm256_add_epi32(weighted[token],
                    _mm256_mullo_epi32(dot0, _mm256_set1_epi32(scale0)));
                weighted[token] = _mm256_add_epi32(weighted[token],
                    _mm256_mullo_epi32(dot1, _mm256_set1_epi32(scale1)));
                minimum[token] += min0 *
                    (activation->sums[chunk * 4] +
                     activation->sums[chunk * 4 + 1]);
                minimum[token] += min1 *
                    (activation->sums[chunk * 4 + 2] +
                     activation->sums[chunk * 4 + 3]);
            }
            high0 <<= 2;
            high1 <<= 2;
        }
        for (uint32_t token = 0; token < batch_size; ++token) {
            const Q38Q8KBlock *activation = input +
                (uint64_t)token * blocks + block;
            const float d = base_d * activation->scale;
            const float dmin = base_dmin * activation->scale;
            accumulated[token] = _mm256_fmadd_ps(_mm256_set1_ps(d),
                _mm256_cvtepi32_ps(weighted[token]), accumulated[token]);
            accumulated_min[token] = _mm_fmadd_ss(_mm_set_ss(-dmin),
                _mm_set_ss((float)minimum[token]), accumulated_min[token]);
        }
    }
    for (uint32_t token = 0; token < batch_size; ++token)
        output[token] = q38_sum_f32_8(accumulated[token]) +
                        _mm_cvtss_f32(accumulated_min[token]);
}

static void q38_dot_q6_k_q8_k_tile(float output[Q38_Q8_K_TILE],
                                    const uint8_t *weights,
                                    const Q38Q8KBlock *input,
                                    uint32_t batch_size, uint64_t blocks)
{
    __m256 accumulated[Q38_Q8_K_TILE];
    for (uint32_t token = 0; token < batch_size; ++token)
        accumulated[token] = _mm256_setzero_ps();
    for (uint64_t block = 0; block < blocks; ++block, weights += 210) {
        const uint8_t *low = weights;
        const uint8_t *high = weights + 128;
        const int8_t *scales = (const int8_t *)(weights + 192);
        const float base_d = q38_f16_to_f32(q38_load_u16(weights + 208));
        const __m256i mask2 = _mm256_set1_epi8(3);
        const __m256i mask4 = _mm256_set1_epi8(15);
        const __m128i packed_scales = _mm_loadu_si128(
            (const __m128i *)scales);
        const __m256i wide_scales = _mm256_cvtepi8_epi16(packed_scales);
        __m256i weighted[Q38_Q8_K_TILE];
        for (uint32_t token = 0; token < batch_size; ++token)
            weighted[token] = _mm256_setzero_si256();
        for (int half = 0; half < 2; ++half) {
            const __m256i low0 = _mm256_loadu_si256(
                (const __m256i *)(low + half * 64));
            const __m256i low1 = _mm256_loadu_si256(
                (const __m256i *)(low + half * 64 + 32));
            const __m256i upper = _mm256_loadu_si256(
                (const __m256i *)(high + half * 32));
            const __m256i q0 = _mm256_or_si256(
                _mm256_and_si256(low0, mask4),
                _mm256_slli_epi16(_mm256_and_si256(upper, mask2), 4));
            const __m256i q1 = _mm256_or_si256(
                _mm256_and_si256(low1, mask4),
                _mm256_slli_epi16(_mm256_and_si256(
                    _mm256_srli_epi16(upper, 2), mask2), 4));
            const __m256i q2 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(low0, 4), mask4),
                _mm256_slli_epi16(_mm256_and_si256(
                    _mm256_srli_epi16(upper, 4), mask2), 4));
            const __m256i q3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(low1, 4), mask4),
                _mm256_srli_epi16(_mm256_and_si256(
                    upper, _mm256_set1_epi8((char)0xc0)), 2));
            const int base = half * 128;
            const int scale_index = half * 8;
            const __m256i scale0 = q38_q6_scale_pair(scales, scale_index);
            const __m256i scale1 = q38_q6_scale_pair(scales, scale_index + 2);
            const __m256i scale2 = q38_q6_scale_pair(scales, scale_index + 4);
            const __m256i scale3 = q38_q6_scale_pair(scales, scale_index + 6);
            for (uint32_t token = 0; token < batch_size; ++token) {
                const Q38Q8KBlock *activation = input +
                    (uint64_t)token * blocks + block;
                __m256i product0 = _mm256_maddubs_epi16(q0,
                    _mm256_loadu_si256((const __m256i *)(
                        activation->quants + base)));
                __m256i product1 = _mm256_maddubs_epi16(q1,
                    _mm256_loadu_si256((const __m256i *)(
                        activation->quants + base + 32)));
                __m256i product2 = _mm256_maddubs_epi16(q2,
                    _mm256_loadu_si256((const __m256i *)(
                        activation->quants + base + 64)));
                __m256i product3 = _mm256_maddubs_epi16(q3,
                    _mm256_loadu_si256((const __m256i *)(
                        activation->quants + base + 96)));
                product0 = _mm256_madd_epi16(scale0, product0);
                product1 = _mm256_madd_epi16(scale1, product1);
                product2 = _mm256_madd_epi16(scale2, product2);
                product3 = _mm256_madd_epi16(scale3, product3);
                weighted[token] = _mm256_add_epi32(weighted[token],
                    _mm256_add_epi32(product0, product1));
                weighted[token] = _mm256_add_epi32(weighted[token],
                    _mm256_add_epi32(product2, product3));
            }
        }
        for (uint32_t token = 0; token < batch_size; ++token) {
            const Q38Q8KBlock *activation = input +
                (uint64_t)token * blocks + block;
            const __m256i input_sums = _mm256_loadu_si256(
                (const __m256i *)activation->sums);
            const __m256i offset = _mm256_slli_epi32(
                _mm256_madd_epi16(input_sums, wide_scales), 5);
            weighted[token] = _mm256_sub_epi32(weighted[token], offset);
            const float d = base_d * activation->scale;
            accumulated[token] = _mm256_fmadd_ps(_mm256_set1_ps(d),
                _mm256_cvtepi32_ps(weighted[token]), accumulated[token]);
        }
    }
    for (uint32_t token = 0; token < batch_size; ++token)
        output[token] = q38_sum_f32_8(accumulated[token]);
}
#endif

static float q38_dot_row(const uint8_t *row, const float *input,
                         uint64_t length, uint32_t type)
{
    float total = 0.0f;
    switch (type) {
    case Q38_GGML_F32:
        for (uint64_t i = 0; i < length; ++i) {
            float value;
            memcpy(&value, row + i * 4, 4);
            total = fmaf(value, input[i], total);
        }
        return total;
    case Q38_GGML_F16:
        for (uint64_t i = 0; i < length; ++i) {
            total = fmaf(q38_f16_to_f32(q38_load_u16(row + i * 2)), input[i], total);
        }
        return total;
    case Q38_GGML_BF16:
        for (uint64_t i = 0; i < length; ++i) {
            total = fmaf(q38_bf16_to_f32_local(row + i * 2), input[i], total);
        }
        return total;
    case Q38_GGML_Q2_K: return q38_dot_q2_k(row, input, length);
    case Q38_GGML_Q3_K: return q38_dot_q3_k(row, input, length);
    case Q38_GGML_Q4_K: return q38_dot_q4_k(row, input, length);
    case Q38_GGML_Q5_K: return q38_dot_q5_k(row, input, length);
    case Q38_GGML_Q6_K: return q38_dot_q6_k(row, input, length);
    case Q38_GGML_Q8_0: return q38_dot_q8_0(row, input, length);
    case Q38_GGML_IQ4_NL:
        for (uint64_t base = 0; base < length; base += 32, row += 18) {
            const float d = q38_f16_to_f32(q38_load_u16(row));
            for (int lane = 0; lane < 16; ++lane) {
                const uint8_t packed = row[2 + lane];
                total = fmaf(d * kvalues_iq4nl[packed & 15u],
                             input[base + (uint64_t)lane], total);
                total = fmaf(d * kvalues_iq4nl[packed >> 4],
                             input[base + (uint64_t)lane + 16], total);
            }
        }
        return total;
    default: return NAN;
    }
}

int q38_tensor_gemv_f32(float *output, const float *input,
                        const Q38GGUFTensor *tensor)
{
    if (!output || !input || !tensor || tensor->n_dims != 2) return 0;
    const uint64_t width = tensor->shape[0];
    const uint64_t rows = tensor->shape[1];
    const uint32_t block_elements = q38_ggml_block_elements(tensor->type);
    const uint32_t block_bytes = q38_ggml_block_bytes(tensor->type);
    if (!block_elements || !block_bytes || width % block_elements != 0) return 0;
    const uint64_t row_bytes = width / block_elements * block_bytes;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(rows >= 256)
#endif
    for (uint64_t row = 0; row < rows; ++row) {
        output[row] = q38_dot_row(tensor->data + row * row_bytes,
                                  input, width, tensor->type);
    }
    return 1;
}

int q38_tensor_gemv_q8_k(float *output, const Q38Q8KBlock *input,
                         uint64_t input_length,
                         const Q38GGUFTensor *tensor)
{
    if (!output || !input || !tensor || tensor->n_dims != 2 ||
        tensor->shape[0] != input_length ||
        input_length % Q38_Q8_K_BLOCK_SIZE != 0) return 0;
    uint64_t row_bytes;
    switch (tensor->type) {
    case Q38_GGML_Q2_K: row_bytes = input_length / 256 * 84; break;
    case Q38_GGML_Q3_K: row_bytes = input_length / 256 * 110; break;
    case Q38_GGML_Q4_K: row_bytes = input_length / 256 * 144; break;
    case Q38_GGML_Q5_K: row_bytes = input_length / 256 * 176; break;
    case Q38_GGML_Q6_K: row_bytes = input_length / 256 * 210; break;
    case Q38_GGML_IQ2_XXS: row_bytes = input_length / 256 * 66; break;
    case Q38_GGML_IQ2_XS: row_bytes = input_length / 256 * 74; break;
    case Q38_GGML_IQ3_XXS: row_bytes = input_length / 256 * 98; break;
    case Q38_GGML_IQ3_S: row_bytes = input_length / 256 * 110; break;
    case Q38_GGML_IQ2_S: row_bytes = input_length / 256 * 82; break;
    case Q38_GGML_IQ4_XS: row_bytes = input_length / 256 * 136; break;
    case Q38_GGML_IQ4_NL: row_bytes = input_length / 256 * 144; break;
    case Q38_GGML_IQ1_S: row_bytes = input_length / 256 * 50; break;
    case Q38_GGML_IQ1_M: row_bytes = input_length / 256 * 56; break;
    default: return 0;
    }
    const uint64_t blocks = input_length / Q38_Q8_K_BLOCK_SIZE;
    const uint64_t rows = tensor->shape[1];
    if (tensor->type == Q38_GGML_IQ1_S && tensor->iq1_s_repack) {
        const Q38IQ1SRepackedBlock *repacked = tensor->iq1_s_repack;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(rows >= 256)
#endif
        for (uint64_t row = 0; row < rows; ++row) {
            output[row] = q38_dot_iq1_s_repacked_q8_k(
                repacked + row * blocks, input, blocks);
        }
        return 1;
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(rows >= 256)
#endif
    for (uint64_t row = 0; row < rows; ++row) {
        const uint8_t *weights = tensor->data + row * row_bytes;
        switch (tensor->type) {
        case Q38_GGML_Q2_K:
            output[row] = q38_dot_q2_k_q8_k(weights, input, blocks);
            break;
        case Q38_GGML_Q3_K:
            output[row] = q38_dot_q3_k_q8_k(weights, input, blocks);
            break;
        case Q38_GGML_IQ2_S:
            output[row] = q38_dot_iq2_s_q8_k(weights, input, blocks);
            break;
        case Q38_GGML_IQ2_XXS:
            output[row] = q38_dot_iq2_xxs_q8_k(weights, input, blocks);
            break;
        case Q38_GGML_IQ2_XS:
            output[row] = q38_dot_iq2_xs_q8_k(weights, input, blocks);
            break;
        case Q38_GGML_IQ3_XXS:
            output[row] = q38_dot_iq3_xxs_q8_k(weights, input, blocks);
            break;
        case Q38_GGML_IQ3_S:
            output[row] = q38_dot_iq3_s_q8_k(weights, input, blocks);
            break;
        case Q38_GGML_IQ4_XS:
            output[row] = q38_dot_iq4_xs_q8_k(weights, input, blocks);
            break;
        case Q38_GGML_IQ4_NL:
            output[row] = q38_dot_iq4_nl_q8_k(weights, input, blocks);
            break;
        case Q38_GGML_IQ1_S:
            output[row] = q38_dot_iq1_s_q8_k(weights, input, blocks);
            break;
        case Q38_GGML_IQ1_M:
            output[row] = q38_dot_iq1_m_q8_k(weights, input, blocks);
            break;
        case Q38_GGML_Q4_K:
            output[row] = q38_dot_q4_k_q8_k(weights, input, blocks);
            break;
        case Q38_GGML_Q5_K:
            output[row] = q38_dot_q5_k_q8_k(weights, input, blocks);
            break;
        default:
            output[row] = q38_dot_q6_k_q8_k(weights, input, blocks);
            break;
        }
    }
    return 1;
}

int q38_tensor_gemm_f32(float *output, const float *input,
                        uint32_t batch_size, uint64_t input_length,
                        const Q38GGUFTensor *tensor)
{
    if (!output || !input || !batch_size || !tensor || tensor->n_dims != 2 ||
        tensor->shape[0] != input_length) return 0;
    const uint32_t block_elements = q38_ggml_block_elements(tensor->type);
    const uint32_t block_bytes = q38_ggml_block_bytes(tensor->type);
    if (!block_elements || !block_bytes || input_length % block_elements != 0)
        return 0;
    const uint64_t row_bytes = input_length / block_elements * block_bytes;
    const uint64_t rows = tensor->shape[1];
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(rows >= 256)
#endif
    for (uint64_t row = 0; row < rows; ++row) {
        const uint8_t *weights = tensor->data + row * row_bytes;
        for (uint32_t token = 0; token < batch_size; ++token) {
            output[(uint64_t)token * rows + row] = q38_dot_row(
                weights, input + (uint64_t)token * input_length,
                input_length, tensor->type);
        }
    }
    return 1;
}

int q38_tensor_gemm_q8_k(float *output, const Q38Q8KBlock *input,
                         uint32_t batch_size, uint64_t input_length,
                         const Q38GGUFTensor *tensor)
{
    if (!output || !input || !batch_size || !tensor || tensor->n_dims != 2 ||
        tensor->shape[0] != input_length ||
        input_length % Q38_Q8_K_BLOCK_SIZE != 0) return 0;
    uint64_t row_bytes;
    switch (tensor->type) {
    case Q38_GGML_Q2_K: row_bytes = input_length / 256 * 84; break;
    case Q38_GGML_Q3_K: row_bytes = input_length / 256 * 110; break;
    case Q38_GGML_Q4_K: row_bytes = input_length / 256 * 144; break;
    case Q38_GGML_Q5_K: row_bytes = input_length / 256 * 176; break;
    case Q38_GGML_Q6_K: row_bytes = input_length / 256 * 210; break;
    case Q38_GGML_IQ2_XXS: row_bytes = input_length / 256 * 66; break;
    case Q38_GGML_IQ2_XS: row_bytes = input_length / 256 * 74; break;
    case Q38_GGML_IQ3_XXS: row_bytes = input_length / 256 * 98; break;
    case Q38_GGML_IQ3_S: row_bytes = input_length / 256 * 110; break;
    case Q38_GGML_IQ2_S: row_bytes = input_length / 256 * 82; break;
    case Q38_GGML_IQ4_XS: row_bytes = input_length / 256 * 136; break;
    case Q38_GGML_IQ4_NL: row_bytes = input_length / 256 * 144; break;
    case Q38_GGML_IQ1_S: row_bytes = input_length / 256 * 50; break;
    case Q38_GGML_IQ1_M: row_bytes = input_length / 256 * 56; break;
    default: return 0;
    }
    const uint64_t blocks = input_length / Q38_Q8_K_BLOCK_SIZE;
    const uint64_t rows = tensor->shape[1];
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(rows >= 256)
#endif
    for (uint64_t row = 0; row < rows; ++row) {
        const uint8_t *weights = tensor->data + row * row_bytes;
#if defined(__AVX2__)
        uint32_t token = 0;
        for (; token + Q38_Q8_K_TILE <= batch_size;
             token += Q38_Q8_K_TILE) {
            float values[Q38_Q8_K_TILE];
            const Q38Q8KBlock *activation = input + (uint64_t)token * blocks;
            switch (tensor->type) {
            case Q38_GGML_Q2_K:
                q38_dot_q2_k_q8_k_tile(values, weights, activation,
                                        Q38_Q8_K_TILE, blocks);
                break;
            case Q38_GGML_Q3_K:
                q38_dot_q3_k_q8_k_tile(values, weights, activation,
                                        Q38_Q8_K_TILE, blocks);
                break;
            case Q38_GGML_IQ2_S:
                q38_dot_iq2_s_q8_k_tile(values, weights, activation,
                                         Q38_Q8_K_TILE, blocks);
                break;
            case Q38_GGML_IQ2_XXS:
                q38_dot_iq2_xxs_q8_k_tile(values, weights, activation,
                                           Q38_Q8_K_TILE, blocks);
                break;
            case Q38_GGML_IQ2_XS:
                q38_dot_iq2_xs_q8_k_tile(values, weights, activation,
                                          Q38_Q8_K_TILE, blocks);
                break;
            case Q38_GGML_IQ3_XXS:
                q38_dot_iq3_xxs_q8_k_tile(values, weights, activation,
                                           Q38_Q8_K_TILE, blocks);
                break;
            case Q38_GGML_IQ3_S:
                q38_dot_iq3_s_q8_k_tile(values, weights, activation,
                                         Q38_Q8_K_TILE, blocks);
                break;
            case Q38_GGML_IQ4_XS:
                q38_dot_iq4_xs_q8_k_tile(values, weights, activation,
                                          Q38_Q8_K_TILE, blocks);
                break;
            case Q38_GGML_IQ4_NL:
                q38_dot_iq4_nl_q8_k_tile(values, weights, activation,
                                          Q38_Q8_K_TILE, blocks);
                break;
            case Q38_GGML_IQ1_S:
                q38_dot_iq1_s_q8_k_fused_tile(values, weights, activation,
                                               Q38_Q8_K_TILE, blocks);
                break;
            case Q38_GGML_IQ1_M:
                q38_dot_iq1_m_q8_k_fused_tile(values, weights, activation,
                                               Q38_Q8_K_TILE, blocks);
                break;
            case Q38_GGML_Q4_K:
                q38_dot_q4_k_q8_k_tile(values, weights, activation,
                                        Q38_Q8_K_TILE, blocks);
                break;
            case Q38_GGML_Q5_K:
                q38_dot_q5_k_q8_k_tile(values, weights, activation,
                                        Q38_Q8_K_TILE, blocks);
                break;
            default:
                q38_dot_q6_k_q8_k_tile(values, weights, activation,
                                        Q38_Q8_K_TILE, blocks);
                break;
            }
            for (uint32_t lane = 0; lane < Q38_Q8_K_TILE; ++lane)
                output[(uint64_t)(token + lane) * rows + row] = values[lane];
        }
        if (batch_size - token >= 2u) {
            const uint32_t tile = batch_size - token;
            float values[Q38_Q8_K_TILE];
            const Q38Q8KBlock *activation = input + (uint64_t)token * blocks;
            switch (tensor->type) {
            case Q38_GGML_Q2_K:
                q38_dot_q2_k_q8_k_tile(values, weights, activation,
                                        tile, blocks);
                break;
            case Q38_GGML_Q3_K:
                q38_dot_q3_k_q8_k_tile(values, weights, activation,
                                        tile, blocks);
                break;
            case Q38_GGML_IQ2_S:
                q38_dot_iq2_s_q8_k_tile(values, weights, activation,
                                         tile, blocks);
                break;
            case Q38_GGML_IQ2_XXS:
                q38_dot_iq2_xxs_q8_k_tile(values, weights, activation,
                                           tile, blocks);
                break;
            case Q38_GGML_IQ2_XS:
                q38_dot_iq2_xs_q8_k_tile(values, weights, activation,
                                          tile, blocks);
                break;
            case Q38_GGML_IQ3_XXS:
                q38_dot_iq3_xxs_q8_k_tile(values, weights, activation,
                                           tile, blocks);
                break;
            case Q38_GGML_IQ3_S:
                q38_dot_iq3_s_q8_k_tile(values, weights, activation,
                                         tile, blocks);
                break;
            case Q38_GGML_IQ4_XS:
                q38_dot_iq4_xs_q8_k_tile(values, weights, activation,
                                          tile, blocks);
                break;
            case Q38_GGML_IQ4_NL:
                q38_dot_iq4_nl_q8_k_tile(values, weights, activation,
                                          tile, blocks);
                break;
            case Q38_GGML_IQ1_S:
                q38_dot_iq1_s_q8_k_fused_tile(values, weights, activation,
                                               tile, blocks);
                break;
            case Q38_GGML_IQ1_M:
                q38_dot_iq1_m_q8_k_fused_tile(values, weights, activation,
                                               tile, blocks);
                break;
            case Q38_GGML_Q4_K:
                q38_dot_q4_k_q8_k_tile(values, weights, activation,
                                        tile, blocks);
                break;
            case Q38_GGML_Q5_K:
                q38_dot_q5_k_q8_k_tile(values, weights, activation,
                                        tile, blocks);
                break;
            default:
                q38_dot_q6_k_q8_k_tile(values, weights, activation,
                                        tile, blocks);
                break;
            }
            for (uint32_t lane = 0; lane < tile; ++lane)
                output[(uint64_t)(token + lane) * rows + row] = values[lane];
            token += tile;
        }
        for (; token < batch_size; ++token) {
#else
        for (uint32_t token = 0; token < batch_size; ++token) {
#endif
            const Q38Q8KBlock *activation = input + (uint64_t)token * blocks;
            float value;
            switch (tensor->type) {
            case Q38_GGML_Q2_K:
                value = q38_dot_q2_k_q8_k(weights, activation, blocks);
                break;
            case Q38_GGML_Q3_K:
                value = q38_dot_q3_k_q8_k(weights, activation, blocks);
                break;
            case Q38_GGML_IQ2_S:
                value = q38_dot_iq2_s_q8_k(weights, activation, blocks);
                break;
            case Q38_GGML_IQ2_XXS:
                value = q38_dot_iq2_xxs_q8_k(weights, activation, blocks);
                break;
            case Q38_GGML_IQ2_XS:
                value = q38_dot_iq2_xs_q8_k(weights, activation, blocks);
                break;
            case Q38_GGML_IQ3_XXS:
                value = q38_dot_iq3_xxs_q8_k(weights, activation, blocks);
                break;
            case Q38_GGML_IQ3_S:
                value = q38_dot_iq3_s_q8_k(weights, activation, blocks);
                break;
            case Q38_GGML_IQ4_XS:
                value = q38_dot_iq4_xs_q8_k(weights, activation, blocks);
                break;
            case Q38_GGML_IQ4_NL:
                value = q38_dot_iq4_nl_q8_k(weights, activation, blocks);
                break;
            case Q38_GGML_IQ1_S:
                value = q38_dot_iq1_s_q8_k(weights, activation, blocks);
                break;
            case Q38_GGML_IQ1_M:
                value = q38_dot_iq1_m_q8_k(weights, activation, blocks);
                break;
            case Q38_GGML_Q4_K:
                value = q38_dot_q4_k_q8_k(weights, activation, blocks);
                break;
            case Q38_GGML_Q5_K:
                value = q38_dot_q5_k_q8_k(weights, activation, blocks);
                break;
            default:
                value = q38_dot_q6_k_q8_k(weights, activation, blocks);
                break;
            }
            output[(uint64_t)token * rows + row] = value;
        }
    }
    return 1;
}

int q38_tensor_dot_row_f32(float *output, const float *input,
                            const Q38GGUFTensor *tensor, uint64_t row)
{
    if (!output || !input || !tensor || tensor->n_dims != 2 ||
        row >= tensor->shape[1]) return 0;
    const uint64_t width = tensor->shape[0];
    const uint32_t block_elements = q38_ggml_block_elements(tensor->type);
    const uint32_t block_bytes = q38_ggml_block_bytes(tensor->type);
    if (!block_elements || !block_bytes || width % block_elements != 0) return 0;
    const uint64_t row_bytes = width / block_elements * block_bytes;
    *output = q38_dot_row(tensor->data + row * row_bytes,
                          input, width, tensor->type);
    return isfinite(*output);
}

int q38_tensor_row_f32(float *output, const Q38GGUFTensor *tensor,
                       uint64_t row)
{
    if (!output || !tensor || tensor->n_dims != 2 || row >= tensor->shape[1]) return 0;
    const uint64_t width = tensor->shape[0];
    const uint32_t block_elements = q38_ggml_block_elements(tensor->type);
    const uint32_t block_bytes = q38_ggml_block_bytes(tensor->type);
    if (!block_elements || !block_bytes || width % block_elements != 0) return 0;
    const uint64_t row_bytes = width / block_elements * block_bytes;
    const uint8_t *data = tensor->data + row * row_bytes;

    switch (tensor->type) {
    case Q38_GGML_F32:
        memcpy(output, data, (size_t)width * sizeof(float));
        return 1;
    case Q38_GGML_F16:
        for (uint64_t i = 0; i < width; ++i)
            output[i] = q38_f16_to_f32(q38_load_u16(data + i * 2));
        return 1;
    case Q38_GGML_BF16:
        for (uint64_t i = 0; i < width; ++i)
            output[i] = q38_bf16_to_f32_local(data + i * 2);
        return 1;
    default: {
        float basis[256] = {0};
        const uint64_t stride = block_elements;
        for (uint64_t base = 0; base < width; base += stride) {
            for (uint64_t i = 0; i < stride; ++i) {
                basis[i] = 1.0f;
                output[base + i] = q38_dot_row(data + base / block_elements * block_bytes,
                                                basis, stride, tensor->type);
                basis[i] = 0.0f;
            }
        }
        return 1;
    }
    }
}
