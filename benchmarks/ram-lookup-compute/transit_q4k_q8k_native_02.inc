static inline int32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

static inline __m256i q4_dot32_scaled(__m256i q4_unsigned, __m256i q8_signed, int scale) {
    const __m256i p16 = _mm256_maddubs_epi16(q4_unsigned, q8_signed);
    const __m256i sc = _mm256_set1_epi16(static_cast<short>(scale));
    return _mm256_madd_epi16(p16, sc);
}

#if defined(__AVXVNNI__)
static inline __m256i q4_dot32_scaled_vnni(__m256i q4_unsigned, __m256i q8_signed, int scale) {
    // VPDPBUSD: 4 unsigned-byte * signed-byte products accumulated into each i32 lane.
    const __m256i dot4 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), q4_unsigned, q8_signed);
    return _mm256_mullo_epi32(dot4, _mm256_set1_epi32(scale));
}
#endif

template<bool USE_VNNI>
static inline __m256i q4_dot32_dispatch(__m256i q4_unsigned, __m256i q8_signed, int scale) {
#if defined(__AVXVNNI__)
    if constexpr (USE_VNNI) {
        return q4_dot32_scaled_vnni(q4_unsigned, q8_signed, scale);
    }
#endif
    return q4_dot32_scaled(q4_unsigned, q8_signed, scale);
}

static inline __m256i mm256_set_m128i(__m128i high, __m128i low) {
    return _mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1);
}

static inline __m256i scale_shuffle_k4(int i) {
    // Equivalent to llama.cpp get_scale_shuffle_k4(i): repeat byte pair
    // [2*i, 2*i+1] across the full 256-bit shuffle mask.
    const int mask16 = 0x0100 + 0x0202 * i;
    return _mm256_set1_epi16(static_cast<short>(mask16));
}

// -----------------------------------------------------------------------------
// Standalone port of the current llama.cpp AVX2 Q4_K x Q8_K inner kernel shape.
// Source snapshot used while writing this benchmark:
// ggml-org/llama.cpp @ 930e2fa5995789efbf249a8bf61325bb626e417b
// ggml/src/ggml-cpu/arch/x86/quants.c::ggml_vec_dot_q4_K_q8_K
//
// The surrounding ggml dispatch/runtime is intentionally not copied here; this
// function exists so Transit variants are compared against the relevant AVX2
// arithmetic path rather than against our simpler readable AVX2 loop.
// -----------------------------------------------------------------------------
static void gemv_llama_style_avx2_raw_1row(
    const uint8_t * q4raw,
    const Q8K * q8,
    float * out,
    int rows,
    int cols) {

    const int nb = cols / QK_K;
    const size_t row_bytes = static_cast<size_t>(nb) * sizeof(Q4KRaw);
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    constexpr uint32_t kmask1 = 0x3f3f3f3fU;
    constexpr uint32_t kmask2 = 0x0f0f0f0fU;
    constexpr uint32_t kmask3 = 0x03030303U;

    for (int row = 0; row < rows; ++row) {
        const Q4KRaw * x = reinterpret_cast<const Q4KRaw *>(
            q4raw + static_cast<size_t>(row) * row_bytes);
        __m256 acc = _mm256_setzero_ps();
        __m128 acc_m = _mm_setzero_ps();
        uint32_t utmp[4] = {};

        for (int i = 0; i < nb; ++i) {
            const float d = q8[i].d * half_to_float(x[i].d);
            const float dmin = -q8[i].d * half_to_float(x[i].dmin);

            std::memcpy(utmp, x[i].scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            const __m128i packed_scales_mins = _mm_set_epi32(
                static_cast<int>(utmp[3]), static_cast<int>(utmp[2]),
                static_cast<int>(utmp[1]), static_cast<int>(utmp[0]));
            const __m256i mins_and_scales = _mm256_cvtepu8_epi16(packed_scales_mins);

            const __m256i q8sums = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(q8[i].bsums));
            const __m128i q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(
                _mm256_extracti128_si256(mins_and_scales, 1), q8s);
            acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

            const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            const __m256i scales = mm256_set_m128i(sc128, sc128);

            const uint8_t * q4 = x[i].qs;
            const int8_t * yq = q8[i].qs;
            __m256i sumi = _mm256_setzero_si256();

            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i scale_l = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j));
                const __m256i scale_h = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 1));
                const __m256i q4bits = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(q4));
                q4 += 32;
                const __m256i q4l = _mm256_and_si256(q4bits, m4);
                const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                const __m256i q8l = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(yq));
                yq += 32;
                __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
                p16l = _mm256_madd_epi16(scale_l, p16l);

                const __m256i q8h = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(yq));
                yq += 32;
                __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
                p16h = _mm256_madd_epi16(scale_h, p16h);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }

            acc = _mm256_fmadd_ps(
                _mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        }

        acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
        acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
        out[row] = hsum256_ps(acc) + _mm_cvtss_f32(acc_m);
    }
}

// -----------------------------------------------------------------------------
// AVX2 single-row baseline. It intentionally mirrors the current llama.cpp
// arithmetic shape (nibble unpack -> maddubs -> madd -> float block scale), while
// keeping scale/min unpacking explicit and readable for the experiment.
// -----------------------------------------------------------------------------
static void gemv_avx2_raw_1row(
    const uint8_t * q4raw,
    const Q8K * q8,
    float * out,
    int rows,
    int cols) {

    const int nb = cols / QK_K;
    const size_t row_bytes = static_cast<size_t>(nb) * sizeof(Q4KRaw);
    const __m256i mask4 = _mm256_set1_epi8(0x0f);

    for (int row = 0; row < rows; ++row) {
        const Q4KRaw * xb = reinterpret_cast<const Q4KRaw *>(q4raw + static_cast<size_t>(row) * row_bytes);
        __m256 accv = _mm256_setzero_ps();
        float accmin = 0.0f;

        for (int b = 0; b < nb; ++b) {
            uint8_t scales[8];
            uint8_t mins[8];
            unpack_scales_mins(xb[b], scales, mins);

            int32_t mincorr = 0;
            for (int g = 0; g < 8; ++g) {
                mincorr += static_cast<int32_t>(mins[g]) * q8_sum32(q8[b], g);
            }

            __m256i sumi = _mm256_setzero_si256();
            for (int chunk = 0; chunk < 4; ++chunk) {
                const __m256i packed = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(xb[b].qs + chunk * 32));
                const __m256i ql = _mm256_and_si256(packed, mask4);
                const __m256i qh = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4);
                const __m256i ylo = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(q8[b].qs + chunk * 64));
                const __m256i yhi = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(q8[b].qs + chunk * 64 + 32));

                sumi = _mm256_add_epi32(sumi, q4_dot32_scaled(ql, ylo, scales[2 * chunk]));
                sumi = _mm256_add_epi32(sumi, q4_dot32_scaled(qh, yhi, scales[2 * chunk + 1]));
            }

            const float d = q8[b].d * half_to_float(xb[b].d);
            const float dm = -q8[b].d * half_to_float(xb[b].dmin);
            accv = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accv);
            accmin += dm * static_cast<float>(mincorr);
        }

        out[row] = hsum256_ps(accv) + accmin;
    }
}

// -----------------------------------------------------------------------------
// AVX2 with model-load-time repacked scales/mins (148 bytes/block, +2.78%).
// -----------------------------------------------------------------------------
static void gemv_avx2_repacked_1row(
    const Q4KRepacked * q4,
    const Q8K * q8,
    float * out,
    int rows,
    int cols) {

    const int nb = cols / QK_K;
    const __m256i mask4 = _mm256_set1_epi8(0x0f);

    for (int row = 0; row < rows; ++row) {
        const Q4KRepacked * xr = q4 + static_cast<size_t>(row) * nb;
        __m256 accv = _mm256_setzero_ps();
        float accmin = 0.0f;

        for (int b = 0; b < nb; ++b) {
            int32_t mincorr = 0;
            for (int g = 0; g < 8; ++g) {
                mincorr += static_cast<int32_t>(xr[b].mins[g]) * q8_sum32(q8[b], g);
            }

            __m256i sumi = _mm256_setzero_si256();
            for (int chunk = 0; chunk < 4; ++chunk) {
                const __m256i packed = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(xr[b].qs + chunk * 32));
                const __m256i ql = _mm256_and_si256(packed, mask4);
                const __m256i qh = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4);
                const __m256i ylo = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(q8[b].qs + chunk * 64));
                const __m256i yhi = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(q8[b].qs + chunk * 64 + 32));

                sumi = _mm256_add_epi32(sumi, q4_dot32_scaled(ql, ylo, xr[b].scales[2 * chunk]));
                sumi = _mm256_add_epi32(sumi, q4_dot32_scaled(qh, yhi, xr[b].scales[2 * chunk + 1]));
            }

            const float d = q8[b].d * half_to_float(xr[b].d);
            const float dm = -q8[b].d * half_to_float(xr[b].dmin);
            accv = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accv);
            accmin += dm * static_cast<float>(mincorr);
        }
        out[row] = hsum256_ps(accv) + accmin;
    }
}

#if defined(__AVXVNNI__)
static void gemv_avxvnni_raw_1row(
    const uint8_t * q4raw,
    const Q8K * q8,
    float * out,
    int rows,
    int cols) {

    const int nb = cols / QK_K;
    const size_t row_bytes = static_cast<size_t>(nb) * sizeof(Q4KRaw);
    const __m256i mask4 = _mm256_set1_epi8(0x0f);

    for (int row = 0; row < rows; ++row) {
        const Q4KRaw * xb = reinterpret_cast<const Q4KRaw *>(q4raw + static_cast<size_t>(row) * row_bytes);
        __m256 accv = _mm256_setzero_ps();
        float accmin = 0.0f;
        for (int b = 0; b < nb; ++b) {
            uint8_t scales[8], mins[8];
            unpack_scales_mins(xb[b], scales, mins);
            int32_t mincorr = 0;
            for (int g = 0; g < 8; ++g) {
                mincorr += static_cast<int32_t>(mins[g]) * q8_sum32(q8[b], g);
            }
            __m256i sumi = _mm256_setzero_si256();
            for (int chunk = 0; chunk < 4; ++chunk) {
                const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xb[b].qs + chunk * 32));
                const __m256i ql = _mm256_and_si256(packed, mask4);
                const __m256i qh = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4);
                const __m256i ylo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8[b].qs + chunk * 64));
                const __m256i yhi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8[b].qs + chunk * 64 + 32));
                sumi = _mm256_add_epi32(sumi, q4_dot32_scaled_vnni(ql, ylo, scales[2 * chunk]));
                sumi = _mm256_add_epi32(sumi, q4_dot32_scaled_vnni(qh, yhi, scales[2 * chunk + 1]));
            }
            const float d = q8[b].d * half_to_float(xb[b].d);
            const float dm = -q8[b].d * half_to_float(xb[b].dmin);
            accv = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accv);
            accmin += dm * static_cast<float>(mincorr);
        }
        out[row] = hsum256_ps(accv) + accmin;
    }
}

static void gemv_avxvnni_repacked_1row(
    const Q4KRepacked * q4,
    const Q8K * q8,
    float * out,
    int rows,
    int cols) {

    const int nb = cols / QK_K;
    const __m256i mask4 = _mm256_set1_epi8(0x0f);
    for (int row = 0; row < rows; ++row) {
        const Q4KRepacked * xb = q4 + static_cast<size_t>(row) * nb;
        __m256 accv = _mm256_setzero_ps();
        float accmin = 0.0f;
        for (int b = 0; b < nb; ++b) {
            int32_t mincorr = 0;
            for (int g = 0; g < 8; ++g) {
                mincorr += static_cast<int32_t>(xb[b].mins[g]) * q8_sum32(q8[b], g);
            }
            __m256i sumi = _mm256_setzero_si256();
            for (int chunk = 0; chunk < 4; ++chunk) {
                const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xb[b].qs + chunk * 32));
                const __m256i ql = _mm256_and_si256(packed, mask4);
                const __m256i qh = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4);
                const __m256i ylo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8[b].qs + chunk * 64));
                const __m256i yhi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8[b].qs + chunk * 64 + 32));
                sumi = _mm256_add_epi32(sumi, q4_dot32_scaled_vnni(ql, ylo, xb[b].scales[2 * chunk]));
                sumi = _mm256_add_epi32(sumi, q4_dot32_scaled_vnni(qh, yhi, xb[b].scales[2 * chunk + 1]));
            }
            const float d = q8[b].d * half_to_float(xb[b].d);
            const float dm = -q8[b].d * half_to_float(xb[b].dmin);
            accv = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accv);
            accmin += dm * static_cast<float>(mincorr);
        }
        out[row] = hsum256_ps(accv) + accmin;
    }
}
#endif

// -----------------------------------------------------------------------------
// Multi-row batched AVX2. Q8 vectors are loaded once per chunk and reused across
// R independent Q4 rows. We reduce each 256-weight block to scalar before moving
// to the next block so R=8 does not spill 16+ YMM accumulators.
// -----------------------------------------------------------------------------
