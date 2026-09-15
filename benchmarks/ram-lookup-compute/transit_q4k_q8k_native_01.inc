/*
Transit native Q4_K x Q8_K benchmark.

The standalone function gemv_llama_style_avx2_raw_1row is adapted from the
Q4_K x Q8_K AVX2 arithmetic path in ggml-org/llama.cpp, source snapshot:
930e2fa5995789efbf249a8bf61325bb626e417b.

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <immintrin.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#include <windows.h>
#endif

namespace transit {

constexpr int QK_K = 256;
constexpr int Q4K_BLOCK_BYTES = 144;
constexpr int Q4K_SCALE_BYTES = 12;
constexpr int Q4K_QS_BYTES = 128;
constexpr int Q8K_BSUMS = 16;

#pragma pack(push, 1)
struct Q4KRaw {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[Q4K_SCALE_BYTES];
    uint8_t qs[Q4K_QS_BYTES];
};

struct Q4KRepacked {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[8];
    uint8_t mins[8];
    uint8_t qs[Q4K_QS_BYTES];
};
#pragma pack(pop)

struct Q8K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[Q8K_BSUMS];
};

static_assert(sizeof(Q4KRaw) == 144, "Q4_K raw block must be 144 bytes");
static_assert(sizeof(Q4KRepacked) == 148, "Transit repacked Q4_K block must be 148 bytes");
static_assert(sizeof(Q8K) == 292, "Q8_K block must be 292 bytes");

static volatile double g_sink = 0.0;

static inline float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x03ffu;
            const uint32_t exp32 = static_cast<uint32_t>(e + 127);
            bits = sign | (exp32 << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        const uint32_t exp32 = exp + (127u - 15u);
        bits = sign | (exp32 << 23) | (mant << 13);
    }

    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline int nearest_int_ggml(float fval) {
    // Same bit-trick used by ggml's reference quantizer for |fval| <= 2^22-1.
    const float val = fval + 12582912.0f;
    int i;
    std::memcpy(&i, &val, sizeof(i));
    return (i & 0x007fffff) - 0x00400000;
}

static inline void get_scale_min(int j, const uint8_t * q, int & d, int & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static inline void unpack_scales_mins(const Q4KRaw & b, uint8_t scales[8], uint8_t mins[8]) {
    for (int j = 0; j < 8; ++j) {
        int s = 0;
        int m = 0;
        get_scale_min(j, b.scales, s, m);
        scales[j] = static_cast<uint8_t>(s);
        mins[j] = static_cast<uint8_t>(m);
    }
}

static inline int32_t q8_sum32(const Q8K & y, int group32) {
    const int i = group32 * 2;
    return static_cast<int32_t>(y.bsums[i]) + static_cast<int32_t>(y.bsums[i + 1]);
}

static std::vector<uint8_t> read_bytes(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("cannot open file: " + path);
    }
    const std::streamsize size = f.tellg();
    if (size < 0) {
        throw std::runtime_error("cannot determine size: " + path);
    }
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (size && !f.read(reinterpret_cast<char *>(data.data()), size)) {
        throw std::runtime_error("cannot read file: " + path);
    }
    return data;
}

static std::vector<float> read_floats(const std::string & path) {
    const auto bytes = read_bytes(path);
    if (bytes.size() % sizeof(float) != 0) {
        throw std::runtime_error("float file size is not divisible by 4: " + path);
    }
    std::vector<float> out(bytes.size() / sizeof(float));
    if (!bytes.empty()) {
        std::memcpy(out.data(), bytes.data(), bytes.size());
    }
    return out;
}

static void quantize_q8k_ref(const float * x, Q8K * y, int64_t k) {
    if (k % QK_K != 0) {
        throw std::runtime_error("Q8_K quantization requires k % 256 == 0");
    }
    const int64_t nb = k / QK_K;

    for (int64_t ib = 0; ib < nb; ++ib) {
        float maxv = 0.0f;
        float amax = 0.0f;
        const float * xb = x + ib * QK_K;
        Q8K & out = y[ib];

        for (int j = 0; j < QK_K; ++j) {
            const float ax = std::fabs(xb[j]);
            if (ax > amax) {
                amax = ax;
                maxv = xb[j];
            }
        }

        if (amax == 0.0f) {
            out.d = 0.0f;
            std::memset(out.qs, 0, sizeof(out.qs));
            std::memset(out.bsums, 0, sizeof(out.bsums));
            continue;
        }

        const float iscale = -127.0f / maxv;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int_ggml(iscale * xb[j]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            out.qs[j] = static_cast<int8_t>(v);
        }

        for (int j = 0; j < Q8K_BSUMS; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += static_cast<int>(out.qs[j * 16 + ii]);
            }
            out.bsums[j] = static_cast<int16_t>(sum);
        }
        out.d = 1.0f / iscale;
    }
}

static std::vector<Q8K> build_q8(const std::vector<float> & x) {
    if (x.size() % QK_K != 0) {
        throw std::runtime_error("activation length is not divisible by 256");
    }
    std::vector<Q8K> q8(x.size() / QK_K);
    quantize_q8k_ref(x.data(), q8.data(), static_cast<int64_t>(x.size()));
    return q8;
}

static std::vector<Q4KRepacked> repack_q4(const std::vector<uint8_t> & raw, int rows, int cols) {
    const int nb = cols / QK_K;
    const size_t expected = static_cast<size_t>(rows) * nb * sizeof(Q4KRaw);
    if (raw.size() != expected) {
        throw std::runtime_error("Q4_K byte size mismatch in repack");
    }

    std::vector<Q4KRepacked> out(static_cast<size_t>(rows) * nb);
    const Q4KRaw * src = reinterpret_cast<const Q4KRaw *>(raw.data());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i].d = src[i].d;
        out[i].dmin = src[i].dmin;
        unpack_scales_mins(src[i], out[i].scales, out[i].mins);
        std::memcpy(out[i].qs, src[i].qs, Q4K_QS_BYTES);
    }
    return out;
}

// -----------------------------------------------------------------------------
// Scalar reference: original Q4_K bytes x Q8_K activation.
// -----------------------------------------------------------------------------
static void gemv_scalar_raw(
    const uint8_t * q4raw,
    const Q8K * q8,
    float * out,
    int rows,
    int cols) {

    const int nb = cols / QK_K;
    const size_t row_bytes = static_cast<size_t>(nb) * sizeof(Q4KRaw);

    for (int row = 0; row < rows; ++row) {
        const Q4KRaw * xb = reinterpret_cast<const Q4KRaw *>(q4raw + static_cast<size_t>(row) * row_bytes);
        float acc = 0.0f;

        for (int b = 0; b < nb; ++b) {
            uint8_t scales[8];
            uint8_t mins[8];
            unpack_scales_mins(xb[b], scales, mins);

            int32_t weighted = 0;
            int32_t mincorr = 0;

            for (int chunk = 0; chunk < 4; ++chunk) {
                const uint8_t * q = xb[b].qs + chunk * 32;
                const int8_t * ylo = q8[b].qs + chunk * 64;
                const int8_t * yhi = ylo + 32;
                int32_t lo = 0;
                int32_t hi = 0;
                for (int l = 0; l < 32; ++l) {
                    const uint8_t qb = q[l];
                    lo += static_cast<int32_t>(qb & 15) * static_cast<int32_t>(ylo[l]);
                    hi += static_cast<int32_t>(qb >> 4) * static_cast<int32_t>(yhi[l]);
                }
                weighted += static_cast<int32_t>(scales[2 * chunk]) * lo;
                weighted += static_cast<int32_t>(scales[2 * chunk + 1]) * hi;
            }

            for (int g = 0; g < 8; ++g) {
                mincorr += static_cast<int32_t>(mins[g]) * q8_sum32(q8[b], g);
            }

            const float d = half_to_float(xb[b].d);
            const float dmin = half_to_float(xb[b].dmin);
            acc += q8[b].d * (d * static_cast<float>(weighted) - dmin * static_cast<float>(mincorr));
        }
        out[row] = acc;
    }
}

static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

