// TensorWave Phase 6: AVX-only Q4 expert microbenchmark for Xeon E7-4890 v2.
//
// Purpose:
//   Measure whether one R920 socket can process enough selected Kimi K2.5
//   expert weights per second to justify the heterogeneous CPU-expert path.
//
// This is deliberately AVX1-compatible (-mavx), not AVX2/AVX-512/AMX.
// It is a correctness-oriented baseline kernel, NOT the final optimized kernel.
// Run one process per NUMA socket under numactl/OS affinity.
//
// Q4 layout mirrors TensorWave Phase 3 density:
//   group = 32 signed int4 weights + float32 scale = 20 bytes = 0.625 B/w.
//
// One K2.5 routed SwiGLU expert:
//   gate: 2048 x 7168
//   up:   2048 x 7168
//   down: 7168 x 2048
//   total = 44,040,192 weights.
//
// Build Linux/GCC:
//   g++ -O3 -std=c++17 -mavx -fopenmp bench_cpu_expert_q4.cpp -o bench_cpu_expert_q4
//
// Example, socket/NUMA node 0:
//   OMP_NUM_THREADS=15 numactl --cpunodebind=0 --membind=0 ./bench_cpu_expert_q4 --iters 20
//
// Windows/MSVC (OpenMP enabled):
//   cl /O2 /std:c++17 /arch:AVX /openmp bench_cpu_expert_q4.cpp
//
// The benchmark reports Gweights/s and the corresponding K2.5 routed-expert
// tok/s ceiling for 4-way equal socket sharding. It does not include router,
// attention, GPU work, CPU<->GPU handoff, KV/state, or synchronization.

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int K2_HIDDEN = 7168;
constexpr int K2_EXPERT_FF = 2048;
constexpr int GROUP = 32;
constexpr int PACKED_BYTES = 16;
constexpr int GROUP_BYTES = 20;
constexpr double BYTES_PER_WEIGHT = 20.0 / 32.0;
constexpr double K25_ROUTED_ACTIVE_WEIGHTS_PER_TOKEN = 21.13929216e9;
constexpr int DEFAULT_SOCKETS = 4;

struct Args {
    int iters = 10;
    int warmup = 2;
    int threads = 15;
    int sockets = DEFAULT_SOCKETS;
};

int parse_int(const char* s, const char* name) {
    try {
        int v = std::stoi(s);
        if (v <= 0) throw std::runtime_error("non-positive");
        return v;
    } catch (...) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + s);
    }
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (k == "--iters") a.iters = parse_int(need("--iters"), "iters");
        else if (k == "--warmup") a.warmup = parse_int(need("--warmup"), "warmup");
        else if (k == "--threads") a.threads = parse_int(need("--threads"), "threads");
        else if (k == "--sockets") a.sockets = parse_int(need("--sockets"), "sockets");
        else if (k == "--help" || k == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " [--iters N] [--warmup N] [--threads N] [--sockets N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + k);
        }
    }
    return a;
}

inline int8_t decode_i4(uint8_t nibble) {
    nibble &= 0x0F;
    return static_cast<int8_t>(nibble < 8 ? nibble : static_cast<int>(nibble) - 16);
}

struct Q4Matrix {
    int rows = 0;
    int cols = 0;
    int groups_per_row = 0;
    std::vector<uint8_t> data;

    Q4Matrix(int r, int c) : rows(r), cols(c) {
        if (c % GROUP != 0) throw std::runtime_error("cols must be divisible by 32");
        groups_per_row = c / GROUP;
        data.resize(static_cast<size_t>(rows) * groups_per_row * GROUP_BYTES);
        init_deterministic();
    }

    size_t encoded_bytes() const { return data.size(); }
    uint64_t weights() const { return static_cast<uint64_t>(rows) * cols; }

    void init_deterministic() {
        for (int r = 0; r < rows; ++r) {
            for (int g = 0; g < groups_per_row; ++g) {
                uint8_t* p = data.data() +
                    (static_cast<size_t>(r) * groups_per_row + g) * GROUP_BYTES;

                for (int b = 0; b < PACKED_BYTES; ++b) {
                    // Two signed int4 values in [-7, 7], deterministic and non-zero.
                    int lo = ((r + g + 2 * b) % 15) - 7;
                    int hi = ((r + 3 * g + 2 * b + 1) % 15) - 7;
                    uint8_t ulo = static_cast<uint8_t>(lo) & 0x0F;
                    uint8_t uhi = static_cast<uint8_t>(hi) & 0x0F;
                    p[b] = static_cast<uint8_t>(ulo | (uhi << 4));
                }

                float scale = 0.00390625f * static_cast<float>(1 + ((r + g) & 7));
                std::memcpy(p + PACKED_BYTES, &scale, sizeof(scale));
            }
        }
    }
};

inline float horizontal_sum(__m256 v) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, v);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3] +
           tmp[4] + tmp[5] + tmp[6] + tmp[7];
}

// AVX1-compatible float MAC path. Int4 unpack is intentionally scalar because
// 256-bit integer unpack/shuffle facilities are AVX2, which E7-4890 v2 lacks.
inline float dot_q4_row_avx1(const uint8_t* row, const float* x, int groups_per_row) {
    __m256 acc = _mm256_setzero_ps();

    for (int g = 0; g < groups_per_row; ++g) {
        const uint8_t* p = row + static_cast<size_t>(g) * GROUP_BYTES;
        float scale;
        std::memcpy(&scale, p + PACKED_BYTES, sizeof(scale));
        const float* xg = x + g * GROUP;

        for (int j = 0; j < GROUP; j += 8) {
            float q[8];
            for (int k = 0; k < 8; ++k) {
                const int idx = j + k;
                const uint8_t packed = p[idx >> 1];
                const uint8_t nibble = (idx & 1) ? (packed >> 4) : packed;
                q[k] = static_cast<float>(decode_i4(nibble)) * scale;
            }
            const __m256 qv = _mm256_set_ps(
                q[7], q[6], q[5], q[4], q[3], q[2], q[1], q[0]);
            const __m256 xv = _mm256_loadu_ps(xg + j);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(qv, xv));
        }
    }

    return horizontal_sum(acc);
}

void gemv_q4(const Q4Matrix& w, const std::vector<float>& x, std::vector<float>& y) {
    if (static_cast<int>(x.size()) != w.cols) throw std::runtime_error("bad x size");
    y.resize(w.rows);
    const size_t row_bytes = static_cast<size_t>(w.groups_per_row) * GROUP_BYTES;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < w.rows; ++r) {
        const uint8_t* row = w.data.data() + static_cast<size_t>(r) * row_bytes;
        y[r] = dot_q4_row_avx1(row, x.data(), w.groups_per_row);
    }
}

void run_expert(const Q4Matrix& gate,
                const Q4Matrix& up,
                const Q4Matrix& down,
                const std::vector<float>& x,
                std::vector<float>& gate_y,
                std::vector<float>& up_y,
                std::vector<float>& hidden,
                std::vector<float>& out) {
    gemv_q4(gate, x, gate_y);
    gemv_q4(up, x, up_y);
    hidden.resize(gate_y.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(hidden.size()); ++i) {
        const float g = gate_y[i];
        const float silu = g / (1.0f + std::exp(-g));
        hidden[i] = silu * up_y[i];
    }

    gemv_q4(down, hidden, out);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args a = parse_args(argc, argv);
        omp_set_dynamic(0);
        omp_set_num_threads(a.threads);

        std::cout << "TensorWave Phase-6 K2.5 CPU expert Q4 benchmark\n";
        std::cout << "ISA target     : AVX1 (-mavx), no AVX2 assumption\n";
        std::cout << "OpenMP threads : " << a.threads << "\n";
        std::cout << "Socket shards  : " << a.sockets << "\n";
        std::cout << "Q4 density     : " << BYTES_PER_WEIGHT << " B/weight\n";

        Q4Matrix gate(K2_EXPERT_FF, K2_HIDDEN);
        Q4Matrix up(K2_EXPERT_FF, K2_HIDDEN);
        Q4Matrix down(K2_HIDDEN, K2_EXPERT_FF);

        const uint64_t weights_per_expert = gate.weights() + up.weights() + down.weights();
        const size_t encoded_bytes = gate.encoded_bytes() + up.encoded_bytes() + down.encoded_bytes();

        std::vector<float> x(K2_HIDDEN);
        for (int i = 0; i < K2_HIDDEN; ++i) x[i] = std::sin(0.001f * i);
        std::vector<float> gate_y, up_y, hidden, out;

        std::cout << "Weights/expert  : " << weights_per_expert << "\n";
        std::cout << "Encoded/expert  : " << std::fixed << std::setprecision(3)
                  << (encoded_bytes / 1e6) << " MB\n";

        for (int i = 0; i < a.warmup; ++i) {
            run_expert(gate, up, down, x, gate_y, up_y, hidden, out);
        }

        const auto t0 = std::chrono::steady_clock::now();
        double checksum = 0.0;
        for (int i = 0; i < a.iters; ++i) {
            run_expert(gate, up, down, x, gate_y, up_y, hidden, out);
            checksum += out[(i * 131) % out.size()];
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();

        const double total_weights = static_cast<double>(weights_per_expert) * a.iters;
        const double total_bytes = static_cast<double>(encoded_bytes) * a.iters;
        const double gweights_s = total_weights / seconds / 1e9;
        const double gb_s = total_bytes / seconds / 1e9;
        const double experts_s = a.iters / seconds;

        // With equal routed-expert sharding across N sockets, each socket is
        // responsible for routed_active/N selected weights per output token.
        const double weights_per_token_per_socket =
            K25_ROUTED_ACTIVE_WEIGHTS_PER_TOKEN / a.sockets;
        const double cpu_expert_tps_ceiling =
            (gweights_s * 1e9) / weights_per_token_per_socket;

        const double target5_gweights =
            (K25_ROUTED_ACTIVE_WEIGHTS_PER_TOKEN * 5.0 / a.sockets) / 1e9;
        const double target10_gweights =
            (K25_ROUTED_ACTIVE_WEIGHTS_PER_TOKEN * 10.0 / a.sockets) / 1e9;

        std::cout << "\n--- measured kernel result ---\n";
        std::cout << "Elapsed         : " << std::setprecision(6) << seconds << " s\n";
        std::cout << "Experts/s       : " << std::setprecision(3) << experts_s << "\n";
        std::cout << "Selected BW     : " << gb_s << " GB/s encoded Q4\n";
        std::cout << "Weight rate     : " << gweights_s << " Gweights/s\n";
        std::cout << "CPU-expert-only : " << cpu_expert_tps_ceiling
                  << " K2.5 tok/s ceiling for " << a.sockets << " equal socket shards\n";
        std::cout << "Checksum        : " << checksum << " (anti-DCE only)\n";

        std::cout << "\n--- exact Phase-6 gates ---\n";
        std::cout << "5 tok/s needs   : " << target5_gweights << " Gweights/s/socket -> "
                  << (gweights_s >= target5_gweights ? "PASS" : "FAIL") << "\n";
        std::cout << "10 tok/s needs  : " << target10_gweights << " Gweights/s/socket -> "
                  << (gweights_s >= target10_gweights ? "PASS" : "FAIL") << "\n";

        std::cout << "\nIMPORTANT: PASS here is necessary, not sufficient. Full-model speed also\n"
                     "includes 60 sequential MoE handoffs, GPU resident path, attention/KV,\n"
                     "router, reductions, NUMA effects, and synchronization.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
