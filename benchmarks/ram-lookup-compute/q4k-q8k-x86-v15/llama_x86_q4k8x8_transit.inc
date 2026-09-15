// Transit experimental x86 specialization for llama.cpp CPU repack Q4_Kx8 x Q8_K GEMV.
// Intended insertion into ggml/src/ggml-cpu/arch/x86/repack.cpp.
//
// Properties:
//   * consumes llama.cpp's existing block_q4_Kx8 runtime repack layout
//   * no change to GGUF weights
//   * no extra quantization / no codebook approximation
//   * reuses one Q8_K activation load across 8 output rows
//   * AVX2 implementation + SSSE3 implementation for Ivy Bridge-class CPUs
//   * falls back to ggml_gemv_q4_K_8x8_q8_K_generic otherwise
//
// Experimental until built and benchmarked in a full llama.cpp tree.

static inline void transit_q4kx8_unpack_scales(
        const uint8_t * p,
        uint8_t sc[8],
        uint8_t mn[8]) {
    for (int r = 0; r < 4; ++r) {
        sc[r]     = p[r] & 63;
        mn[r]     = p[r + 4] & 63;
        sc[r + 4] = (uint8_t) ((p[r + 8] & 15) | ((p[r] >> 6) << 4));
        mn[r + 4] = (uint8_t) ((p[r + 8] >> 4) | ((p[r + 4] >> 6) << 4));
    }
}

static inline uint64_t transit_rep16_4(uint8_t x) {
    return (uint64_t) x * UINT64_C(0x0001000100010001);
}

#if defined(__AVX2__)
static inline __m256i transit_scale4_epi16(const uint8_t * s) {
    return _mm256_set_epi64x(
        (long long) transit_rep16_4(s[3]),
        (long long) transit_rep16_4(s[2]),
        (long long) transit_rep16_4(s[1]),
        (long long) transit_rep16_4(s[0]));
}
#endif

#if defined(__SSSE3__)
static inline __m128i transit_scale2_epi16(const uint8_t * s) {
    return _mm_set_epi64x(
        (long long) transit_rep16_4(s[1]),
        (long long) transit_rep16_4(s[0]));
}
#endif

void ggml_gemv_q4_K_8x8_q8_K(
        int n,
        float * GGML_RESTRICT s,
        size_t bs,
        const void * GGML_RESTRICT vx,
        const void * GGML_RESTRICT vy,
        int nr,
        int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    constexpr int ncols_interleaved = 8;

    assert(n % qk == 0);
    assert(nr == 1);
    assert(nc % ncols_interleaved == 0);
    GGML_UNUSED(bs);

    const block_q4_Kx8 * GGML_RESTRICT x = (const block_q4_Kx8 *) vx;
    const block_q8_K   * GGML_RESTRICT y = (const block_q8_K   *) vy;

#if defined(__AVX2__)
    const __m256i mask4 = _mm256_set1_epi8(15);

    for (int rg = 0; rg < nc / 8; ++rg) {
        float total[8] = {};
        const block_q4_Kx8 * GGML_RESTRICT xb = x + (size_t) rg * nb;

        for (int ib = 0; ib < nb; ++ib) {
            __m256i acc03 = _mm256_setzero_si256();
            __m256i acc47 = _mm256_setzero_si256();
            int mincorr[8] = {};

            for (int pair = 0; pair < 4; ++pair) {
                const int g0 = pair * 2;
                const int g1 = g0 + 1;

                uint8_t sc0[8], mn0[8], sc1[8], mn1[8];
                transit_q4kx8_unpack_scales(xb[ib].scales + g0 * 12, sc0, mn0);
                transit_q4kx8_unpack_scales(xb[ib].scales + g1 * 12, sc1, mn1);

                const __m256i scale_lo_03 = transit_scale4_epi16(sc0 + 0);
                const __m256i scale_hi_03 = transit_scale4_epi16(sc1 + 0);
                const __m256i scale_lo_47 = transit_scale4_epi16(sc0 + 4);
                const __m256i scale_hi_47 = transit_scale4_epi16(sc1 + 4);

                const int sum0 = y[ib].bsums[g0 * 2] + y[ib].bsums[g0 * 2 + 1];
                const int sum1 = y[ib].bsums[g1 * 2] + y[ib].bsums[g1 * 2 + 1];
                for (int r = 0; r < 8; ++r) {
                    mincorr[r] += (int) mn0[r] * sum0 + (int) mn1[r] * sum1;
                }

                for (int kk = 0; kk < 4; ++kk) {
                    const int k   = pair * 4 + kk;
                    const int off = pair * 64 + kk * 8;

                    int64_t a_lo64, a_hi64;
                    memcpy(&a_lo64, y[ib].qs + off,      sizeof(a_lo64));
                    memcpy(&a_hi64, y[ib].qs + off + 32, sizeof(a_hi64));

                    const __m256i a_lo = _mm256_set1_epi64x(a_lo64);
                    const __m256i a_hi = _mm256_set1_epi64x(a_hi64);

                    const uint8_t * q = xb[ib].qs + k * 64;
                    const __m256i q03 = _mm256_loadu_si256((const __m256i *) (q +  0));
                    const __m256i q47 = _mm256_loadu_si256((const __m256i *) (q + 32));

                    const __m256i q03_lo = _mm256_and_si256(q03, mask4);
                    const __m256i q03_hi = _mm256_and_si256(_mm256_srli_epi16(q03, 4), mask4);
                    const __m256i q47_lo = _mm256_and_si256(q47, mask4);
                    const __m256i q47_hi = _mm256_and_si256(_mm256_srli_epi16(q47, 4), mask4);

                    const __m256i p03_lo = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(q03_lo, a_lo), scale_lo_03);
                    const __m256i p03_hi = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(q03_hi, a_hi), scale_hi_03);
                    const __m256i p47_lo = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(q47_lo, a_lo), scale_lo_47);
                    const __m256i p47_hi = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(q47_hi, a_hi), scale_hi_47);

                    acc03 = _mm256_add_epi32(acc03, _mm256_add_epi32(p03_lo, p03_hi));
                    acc47 = _mm256_add_epi32(acc47, _mm256_add_epi32(p47_lo, p47_hi));
                }
            }

            alignas(32) int32_t a03[8], a47[8];
            _mm256_store_si256((__m256i *) a03, _mm256_hadd_epi32(acc03, acc03));
            _mm256_store_si256((__m256i *) a47, _mm256_hadd_epi32(acc47, acc47));
            const int sumi[8] = {
                a03[0], a03[1], a03[4], a03[5],
                a47[0], a47[1], a47[4], a47[5]
            };

            const float yd = y[ib].d;
            for (int r = 0; r < 8; ++r) {
                const float d    = GGML_CPU_FP16_TO_FP32(xb[ib].d[r]);
                const float dmin = GGML_CPU_FP16_TO_FP32(xb[ib].dmin[r]);
                total[r] += yd * (d * (float) sumi[r] - dmin * (float) mincorr[r]);
            }
        }

        for (int r = 0; r < 8; ++r) {
            s[rg * 8 + r] = total[r];
        }
    }

#elif defined(__SSSE3__)
    const __m128i mask4 = _mm_set1_epi8(15);

    for (int rg = 0; rg < nc / 8; ++rg) {
        float total[8] = {};
        const block_q4_Kx8 * GGML_RESTRICT xb = x + (size_t) rg * nb;

        for (int ib = 0; ib < nb; ++ib) {
            __m128i acc[4] = {
                _mm_setzero_si128(), _mm_setzero_si128(),
                _mm_setzero_si128(), _mm_setzero_si128()
            };
            int mincorr[8] = {};

            for (int pair = 0; pair < 4; ++pair) {
                const int g0 = pair * 2;
                const int g1 = g0 + 1;

                uint8_t sc0[8], mn0[8], sc1[8], mn1[8];
                transit_q4kx8_unpack_scales(xb[ib].scales + g0 * 12, sc0, mn0);
                transit_q4kx8_unpack_scales(xb[ib].scales + g1 * 12, sc1, mn1);

                __m128i scale_lo[4], scale_hi[4];
                for (int p = 0; p < 4; ++p) {
                    scale_lo[p] = transit_scale2_epi16(sc0 + p * 2);
                    scale_hi[p] = transit_scale2_epi16(sc1 + p * 2);
                }

                const int sum0 = y[ib].bsums[g0 * 2] + y[ib].bsums[g0 * 2 + 1];
                const int sum1 = y[ib].bsums[g1 * 2] + y[ib].bsums[g1 * 2 + 1];
                for (int r = 0; r < 8; ++r) {
                    mincorr[r] += (int) mn0[r] * sum0 + (int) mn1[r] * sum1;
                }

                for (int kk = 0; kk < 4; ++kk) {
                    const int k   = pair * 4 + kk;
                    const int off = pair * 64 + kk * 8;

                    __m128i a_lo = _mm_loadl_epi64((const __m128i *) (y[ib].qs + off));
                    __m128i a_hi = _mm_loadl_epi64((const __m128i *) (y[ib].qs + off + 32));
                    a_lo = _mm_unpacklo_epi64(a_lo, a_lo);
                    a_hi = _mm_unpacklo_epi64(a_hi, a_hi);

                    const uint8_t * q = xb[ib].qs + k * 64;
                    for (int p = 0; p < 4; ++p) {
                        const __m128i qp = _mm_loadu_si128((const __m128i *) (q + p * 16));
                        const __m128i q_lo = _mm_and_si128(qp, mask4);
                        const __m128i q_hi = _mm_and_si128(_mm_srli_epi16(qp, 4), mask4);
                        const __m128i p_lo = _mm_madd_epi16(
                            _mm_maddubs_epi16(q_lo, a_lo), scale_lo[p]);
                        const __m128i p_hi = _mm_madd_epi16(
                            _mm_maddubs_epi16(q_hi, a_hi), scale_hi[p]);
                        acc[p] = _mm_add_epi32(acc[p], _mm_add_epi32(p_lo, p_hi));
                    }
                }
            }

            int sumi[8];
            for (int p = 0; p < 4; ++p) {
                alignas(16) int32_t tmp[4];
                _mm_store_si128((__m128i *) tmp, _mm_hadd_epi32(acc[p], acc[p]));
                sumi[p * 2 + 0] = tmp[0];
                sumi[p * 2 + 1] = tmp[1];
            }

            const float yd = y[ib].d;
            for (int r = 0; r < 8; ++r) {
                const float d    = GGML_CPU_FP16_TO_FP32(xb[ib].d[r]);
                const float dmin = GGML_CPU_FP16_TO_FP32(xb[ib].dmin[r]);
                total[r] += yd * (d * (float) sumi[r] - dmin * (float) mincorr[r]);
            }
        }

        for (int r = 0; r < 8; ++r) {
            s[rg * 8 + r] = total[r];
        }
    }

#else
    ggml_gemv_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
#endif
}
