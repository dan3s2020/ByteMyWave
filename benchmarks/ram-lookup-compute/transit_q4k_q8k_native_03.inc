template<int R, bool REPACKED, bool USE_VNNI = false>
static void gemv_avx2_batched_impl(
    const uint8_t * q4raw,
    const Q4KRepacked * q4rep,
    const Q8K * q8,
    float * out,
    int rows,
    int cols) {

    const int nb = cols / QK_K;
    const size_t row_bytes = static_cast<size_t>(nb) * sizeof(Q4KRaw);
    const __m256i mask4 = _mm256_set1_epi8(0x0f);

    int row = 0;
    for (; row + R <= rows; row += R) {
        float acc[R] = {};

        for (int b = 0; b < nb; ++b) {
            __m256i sumi[R];
            int32_t mincorr[R];
            float dscale[R];
            float dminscale[R];
            uint8_t sc[R][8];

            for (int r = 0; r < R; ++r) {
                sumi[r] = _mm256_setzero_si256();
                mincorr[r] = 0;

                if constexpr (REPACKED) {
                    const Q4KRepacked & xb = q4rep[static_cast<size_t>(row + r) * nb + b];
                    std::memcpy(sc[r], xb.scales, 8);
                    for (int g = 0; g < 8; ++g) {
                        mincorr[r] += static_cast<int32_t>(xb.mins[g]) * q8_sum32(q8[b], g);
                    }
                    dscale[r] = q8[b].d * half_to_float(xb.d);
                    dminscale[r] = -q8[b].d * half_to_float(xb.dmin);
                } else {
                    const Q4KRaw * xr = reinterpret_cast<const Q4KRaw *>(
                        q4raw + static_cast<size_t>(row + r) * row_bytes);
                    const Q4KRaw & xb = xr[b];
                    uint8_t mins[8];
                    unpack_scales_mins(xb, sc[r], mins);
                    for (int g = 0; g < 8; ++g) {
                        mincorr[r] += static_cast<int32_t>(mins[g]) * q8_sum32(q8[b], g);
                    }
                    dscale[r] = q8[b].d * half_to_float(xb.d);
                    dminscale[r] = -q8[b].d * half_to_float(xb.dmin);
                }
            }

            for (int chunk = 0; chunk < 4; ++chunk) {
                const __m256i ylo = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(q8[b].qs + chunk * 64));
                const __m256i yhi = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(q8[b].qs + chunk * 64 + 32));

                for (int r = 0; r < R; ++r) {
                    const uint8_t * qs;
                    if constexpr (REPACKED) {
                        qs = q4rep[static_cast<size_t>(row + r) * nb + b].qs;
                    } else {
                        const Q4KRaw * xr = reinterpret_cast<const Q4KRaw *>(
                            q4raw + static_cast<size_t>(row + r) * row_bytes);
                        qs = xr[b].qs;
                    }
                    const __m256i packed = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(qs + chunk * 32));
                    const __m256i ql = _mm256_and_si256(packed, mask4);
                    const __m256i qh = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4);
                    sumi[r] = _mm256_add_epi32(sumi[r], q4_dot32_dispatch<USE_VNNI>(ql, ylo, sc[r][2 * chunk]));
                    sumi[r] = _mm256_add_epi32(sumi[r], q4_dot32_dispatch<USE_VNNI>(qh, yhi, sc[r][2 * chunk + 1]));
                }
            }

            for (int r = 0; r < R; ++r) {
                const int32_t weighted = hsum256_epi32(sumi[r]);
                acc[r] += dscale[r] * static_cast<float>(weighted)
                        + dminscale[r] * static_cast<float>(mincorr[r]);
            }
        }

        for (int r = 0; r < R; ++r) {
            out[row + r] = acc[r];
        }
    }

    // Tail should not matter for the current 2560-row tensor, but keep it general.
    if (row < rows) {
        if constexpr (REPACKED) {
            gemv_avx2_repacked_1row(q4rep + static_cast<size_t>(row) * nb, q8, out + row, rows - row, cols);
        } else {
            gemv_avx2_raw_1row(q4raw + static_cast<size_t>(row) * row_bytes, q8, out + row, rows - row, cols);
        }
    }
}

static void gemv_avx2_raw_4row(const uint8_t * q4, const Q8K * q8, float * out, int rows, int cols) {
    gemv_avx2_batched_impl<4, false>(q4, nullptr, q8, out, rows, cols);
}
static void gemv_avx2_raw_8row(const uint8_t * q4, const Q8K * q8, float * out, int rows, int cols) {
    gemv_avx2_batched_impl<8, false>(q4, nullptr, q8, out, rows, cols);
}
static void gemv_avx2_rep_4row(const Q4KRepacked * q4, const Q8K * q8, float * out, int rows, int cols) {
    gemv_avx2_batched_impl<4, true>(nullptr, q4, q8, out, rows, cols);
}
static void gemv_avx2_rep_8row(const Q4KRepacked * q4, const Q8K * q8, float * out, int rows, int cols) {
    gemv_avx2_batched_impl<8, true>(nullptr, q4, q8, out, rows, cols);
}

#if defined(__AVXVNNI__)
static void gemv_avxvnni_raw_4row(const uint8_t * q4, const Q8K * q8, float * out, int rows, int cols) {
    gemv_avx2_batched_impl<4, false, true>(q4, nullptr, q8, out, rows, cols);
}
static void gemv_avxvnni_raw_8row(const uint8_t * q4, const Q8K * q8, float * out, int rows, int cols) {
    gemv_avx2_batched_impl<8, false, true>(q4, nullptr, q8, out, rows, cols);
}
static void gemv_avxvnni_rep_4row(const Q4KRepacked * q4, const Q8K * q8, float * out, int rows, int cols) {
    gemv_avx2_batched_impl<4, true, true>(nullptr, q4, q8, out, rows, cols);
}
static void gemv_avxvnni_rep_8row(const Q4KRepacked * q4, const Q8K * q8, float * out, int rows, int cols) {
    gemv_avx2_batched_impl<8, true, true>(nullptr, q4, q8, out, rows, cols);
}
#endif

// -----------------------------------------------------------------------------
// Exact activation-dependent nibble LUT for Q8_K integer activations.
// Table entry: LUT[position][q4] = q4 * q8_integer[position].
// 9216 positions x 16 int16 values = 288 KiB on the current tensor.
// -----------------------------------------------------------------------------
static void build_nibble_lut_q8(const Q8K * q8, int cols, int16_t * lut) {
    const int nb = cols / QK_K;
    for (int b = 0; b < nb; ++b) {
        for (int p = 0; p < QK_K; ++p) {
            const int pos = b * QK_K + p;
            const int v = static_cast<int>(q8[b].qs[p]);
            int16_t * dst = lut + static_cast<size_t>(pos) * 16;
            for (int q = 0; q < 16; ++q) {
                dst[q] = static_cast<int16_t>(q * v);
            }
        }
    }
}

static void gemv_lut_raw(
    const uint8_t * q4raw,
    const Q8K * q8,
    const int16_t * lut,
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
                const int base = b * QK_K + chunk * 64;
                int32_t lo = 0;
                int32_t hi = 0;
                for (int l = 0; l < 32; ++l) {
                    const uint8_t qb = q[l];
                    lo += lut[(static_cast<size_t>(base + l) * 16) + (qb & 15)];
                    hi += lut[(static_cast<size_t>(base + 32 + l) * 16) + (qb >> 4)];
                }
                weighted += static_cast<int32_t>(scales[2 * chunk]) * lo;
                weighted += static_cast<int32_t>(scales[2 * chunk + 1]) * hi;
            }
            for (int g = 0; g < 8; ++g) {
                mincorr += static_cast<int32_t>(mins[g]) * q8_sum32(q8[b], g);
            }
            const float d = half_to_float(xb[b].d);
            const float dm = half_to_float(xb[b].dmin);
            acc += q8[b].d * (d * static_cast<float>(weighted) - dm * static_cast<float>(mincorr));
        }
        out[row] = acc;
    }
}

static void gemv_lut_repacked(
    const Q4KRepacked * q4,
    const Q8K * q8,
    const int16_t * lut,
    float * out,
    int rows,
    int cols) {

    const int nb = cols / QK_K;
    for (int row = 0; row < rows; ++row) {
        const Q4KRepacked * xb = q4 + static_cast<size_t>(row) * nb;
        float acc = 0.0f;

        for (int b = 0; b < nb; ++b) {
            int32_t weighted = 0;
            int32_t mincorr = 0;
            for (int chunk = 0; chunk < 4; ++chunk) {
                const uint8_t * q = xb[b].qs + chunk * 32;
                const int base = b * QK_K + chunk * 64;
                int32_t lo = 0;
                int32_t hi = 0;
                for (int l = 0; l < 32; ++l) {
                    const uint8_t qb = q[l];
                    lo += lut[(static_cast<size_t>(base + l) * 16) + (qb & 15)];
                    hi += lut[(static_cast<size_t>(base + 32 + l) * 16) + (qb >> 4)];
                }
                weighted += static_cast<int32_t>(xb[b].scales[2 * chunk]) * lo;
                weighted += static_cast<int32_t>(xb[b].scales[2 * chunk + 1]) * hi;
            }
            for (int g = 0; g < 8; ++g) {
                mincorr += static_cast<int32_t>(xb[b].mins[g]) * q8_sum32(q8[b], g);
            }
            const float d = half_to_float(xb[b].d);
            const float dm = half_to_float(xb[b].dmin);
            acc += q8[b].d * (d * static_cast<float>(weighted) - dm * static_cast<float>(mincorr));
        }
        out[row] = acc;
    }
}

// -----------------------------------------------------------------------------
// Correctness and benchmark utilities.
// -----------------------------------------------------------------------------
struct Metrics {
    double max_abs = 0.0;
    double rel_l2 = 0.0;
    double cosine = 0.0;
};

static Metrics compare_vectors(const std::vector<float> & ref, const std::vector<float> & got) {
    if (ref.size() != got.size()) {
        throw std::runtime_error("vector size mismatch in compare");
    }
    long double se = 0.0L;
    long double rr = 0.0L;
    long double gg = 0.0L;
    long double rg = 0.0L;
    double max_abs = 0.0;

    for (size_t i = 0; i < ref.size(); ++i) {
        const long double a = ref[i];
        const long double b = got[i];
        const long double d = a - b;
        se += d * d;
        rr += a * a;
        gg += b * b;
        rg += a * b;
        max_abs = std::max(max_abs, std::fabs(static_cast<double>(d)));
    }

    Metrics m;
    m.max_abs = max_abs;
    m.rel_l2 = rr > 0 ? std::sqrt(static_cast<double>(se / rr)) : 0.0;
    m.cosine = (rr > 0 && gg > 0) ? static_cast<double>(rg / std::sqrt(rr * gg)) : 1.0;
    return m;
}

struct BenchStats {
    double best_ms = 0.0;
    double median_ms = 0.0;
    double mean_ms = 0.0;
};

template<typename Fn>
static BenchStats benchmark(int warmup, int iterations, int rows, Fn && fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
        g_sink += static_cast<double>(i + 1) * 1e-300;
    }

    std::vector<double> ms;
    ms.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ms.push_back(elapsed);
        g_sink += elapsed * 1e-300 + static_cast<double>((i * 131) % std::max(rows, 1)) * 1e-300;
    }

    std::sort(ms.begin(), ms.end());
    BenchStats s;
    s.best_ms = ms.front();
    s.median_ms = ms[ms.size() / 2];
    s.mean_ms = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
    return s;
}

static void print_metrics(const std::string & name, const Metrics & m) {
    std::cout << std::left << std::setw(30) << name
              << " max_abs=" << std::scientific << std::setprecision(6) << m.max_abs
              << " rel_L2=" << m.rel_l2
              << " cosine=" << std::fixed << std::setprecision(12) << m.cosine << "\n";
}

static void print_bench(const std::string & name, const BenchStats & s, double baseline_ms) {
    const double speed = s.median_ms > 0 ? baseline_ms / s.median_ms : 0.0;
    std::cout << std::left << std::setw(34) << name
              << " median=" << std::fixed << std::setprecision(3) << std::setw(9) << s.median_ms
              << " best=" << std::setw(9) << s.best_ms
              << " mean=" << std::setw(9) << s.mean_ms
              << " speed=" << std::setprecision(3) << speed << "x\n";
}

struct Args {
    std::string q4_path;
    std::string x_path;
    std::string fp32_ref_path;
    int rows = 0;
    int cols = 0;
    int iterations = 10;
    int warmup = 2;
    int cpu = -1;
};

static void usage(const char * argv0) {
    std::cerr
        << "Usage:\n  " << argv0
        << " --q4 tensor-q4k.bin --x activation-f32.bin --rows N --cols K"
        << " [--ref reference-f32.bin] [--iterations 10] [--warmup 2] [--cpu logical_index]\n";
}

