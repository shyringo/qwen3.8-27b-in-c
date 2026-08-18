#include "qwen38_quant.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#define Q38_UNUSED_HELPER __attribute__((unused))
#endif

static uint16_t q38_load_u16(const uint8_t *data)
{
    return (uint16_t)data[0] | (uint16_t)data[1] << 8;
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
#define Q38_Q8_K_TILE 4u

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
        int minimum[Q38_Q8_K_TILE] = {0, 0, 0, 0};
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
    case Q38_GGML_Q4_K: return q38_dot_q4_k(row, input, length);
    case Q38_GGML_Q5_K: return q38_dot_q5_k(row, input, length);
    case Q38_GGML_Q6_K: return q38_dot_q6_k(row, input, length);
    case Q38_GGML_Q8_0: return q38_dot_q8_0(row, input, length);
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
    case Q38_GGML_Q4_K: row_bytes = input_length / 256 * 144; break;
    case Q38_GGML_Q5_K: row_bytes = input_length / 256 * 176; break;
    case Q38_GGML_Q6_K: row_bytes = input_length / 256 * 210; break;
    default: return 0;
    }
    const uint64_t blocks = input_length / Q38_Q8_K_BLOCK_SIZE;
    const uint64_t rows = tensor->shape[1];
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(rows >= 256)
#endif
    for (uint64_t row = 0; row < rows; ++row) {
        const uint8_t *weights = tensor->data + row * row_bytes;
        switch (tensor->type) {
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
    case Q38_GGML_Q4_K: row_bytes = input_length / 256 * 144; break;
    case Q38_GGML_Q5_K: row_bytes = input_length / 256 * 176; break;
    case Q38_GGML_Q6_K: row_bytes = input_length / 256 * 210; break;
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
