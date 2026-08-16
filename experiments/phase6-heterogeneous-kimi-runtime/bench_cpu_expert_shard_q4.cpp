// TensorWave Phase 6: AVX1 Q4 benchmark for one distributed Kimi K2.5 expert shard.
//
// The 3xR920 design shards each selected routed expert along the SwiGLU
// intermediate dimension across all 12 CPU sockets. K2.5 FF=2048 and Q4 groups
// are 32 weights, so 64 FF groups are distributed as eight 160-row shards and
// four 192-row shards. This benchmark measures either shard size on one socket.
//
// Build:
//   g++ -O3 -std=c++17 -mavx -fopenmp bench_cpu_expert_shard_q4.cpp -o bench_cpu_expert_shard_q4
//
// Example on one E7-4890 v2 NUMA node:
//   OMP_NUM_THREADS=15 numactl --cpunodebind=0 --membind=0 \
//     ./bench_cpu_expert_shard_q4 --threads 15 --ff-rows 160 --iters 20
//
// Run both --ff-rows 160 and --ff-rows 192. No result from CI is a claim about
// R920 speed; physical NUMA-pinned measurements are required.

#include <immintrin.h>
#include <omp.h>

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
constexpr int HIDDEN = 7168;
constexpr int GROUP = 32;
constexpr int PACKED_BYTES = 16;
constexpr int GROUP_BYTES = 20;
constexpr double K25_ROUTED_ACTIVE_GWEIGHTS_PER_TOKEN = 21.13929216;
constexpr int THREE_R920_SOCKETS = 12;

struct Args {
    int ff_rows = 160;
    int iters = 10;
    int warmup = 2;
    int threads = 15;
};

int parse_positive(const char* text, const char* name) {
    try {
        const int v = std::stoi(text);
        if (v <= 0) throw std::runtime_error("non-positive");
        return v;
    } catch (...) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[i];
        };
        if (k == "--ff-rows") a.ff_rows = parse_positive(need("ff-rows"), "ff-rows");
        else if (k == "--iters") a.iters = parse_positive(need("iters"), "iters");
        else if (k == "--warmup") a.warmup = parse_positive(need("warmup"), "warmup");
        else if (k == "--threads") a.threads = parse_positive(need("threads"), "threads");
        else if (k == "--help" || k == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " [--ff-rows 160|192] [--threads N] [--warmup N] [--iters N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + k);
        }
    }
    if (a.ff_rows % GROUP != 0) {
        throw std::runtime_error("ff-rows must be divisible by 32 for Q4 group alignment");
    }
    return a;
}

inline int8_t decode_i4(uint8_t nibble) {
    nibble &= 0x0F;
    return static_cast<int8_t>(nibble < 8 ? nibble : static_cast<int>(nibble) - 16);
}

struct Q4Matrix {
    int rows;
    int cols;
    int groups_per_row;
    std::vector<uint8_t> data;

    Q4Matrix(int r, int c) : rows(r), cols(c), groups_per_row(c / GROUP) {
        if (c % GROUP != 0) throw std::runtime_error("matrix K dimension must be divisible by 32");
        data.resize(static_cast<size_t>(rows) * groups_per_row * GROUP_BYTES);
        init();
    }

    void init() {
        for (int r = 0; r < rows; ++r) {
            for (int g = 0; g < groups_per_row; ++g) {
                uint8_t* p = data.data() +
                    (static_cast<size_t>(r) * groups_per_row + g) * GROUP_BYTES;
                for (int b = 0; b < PACKED_BYTES; ++b) {
                    const int lo = ((r + g + 2 * b) % 15) - 7;
                    const int hi = ((r + 3 * g + 2 * b + 1) % 15) - 7;
                    p[b] = static_cast<uint8_t>((lo & 0x0F) | ((hi & 0x0F) << 4));
                }
                const float scale = 0.00390625f * static_cast<float>(1 + ((r + g) & 7));
                std::memcpy(p + PACKED_BYTES, &scale, sizeof(scale));
            }
        }
    }

    uint64_t weights() const { return static_cast<uint64_t>(rows) * cols; }
    size_t encoded_bytes() const { return data.size(); }
};

inline float hsum(__m256 v) {
    alignas(32) float x[8];
    _mm256_store_ps(x, v);
    return x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];
}

inline float dot_q4_avx1(const uint8_t* row, const float* x, int groups_per_row) {
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
            const __m256 qv = _mm256_set_ps(q[7], q[6], q[5], q[4], q[3], q[2], q[1], q[0]);
            const __m256 xv = _mm256_loadu_ps(xg + j);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(qv, xv));
        }
    }
    return hsum(acc);
}

void gemv(const Q4Matrix& w, const std::vector<float>& x, std::vector<float>& y) {
    if (static_cast<int>(x.size()) != w.cols) throw std::runtime_error("input size mismatch");
    y.resize(w.rows);
    const size_t row_bytes = static_cast<size_t>(w.groups_per_row) * GROUP_BYTES;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < w.rows; ++r) {
        y[r] = dot_q4_avx1(w.data.data() + static_cast<size_t>(r) * row_bytes,
                           x.data(), w.groups_per_row);
    }
}

void run_shard(const Q4Matrix& gate,
               const Q4Matrix& up,
               const Q4Matrix& down,
               const std::vector<float>& x,
               std::vector<float>& gate_y,
               std::vector<float>& up_y,
               std::vector<float>& intermediate,
               std::vector<float>& partial_out) {
    gemv(gate, x, gate_y);
    gemv(up, x, up_y);
    intermediate.resize(gate_y.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(intermediate.size()); ++i) {
        const float g = gate_y[i];
        intermediate[i] = (g / (1.0f + std::exp(-g))) * up_y[i];
    }
    gemv(down, intermediate, partial_out);
}
} // namespace

int main(int argc, char** argv) {
    try {
        const Args a = parse_args(argc, argv);
        omp_set_dynamic(0);
        omp_set_num_threads(a.threads);

        Q4Matrix gate(a.ff_rows, HIDDEN);
        Q4Matrix up(a.ff_rows, HIDDEN);
        Q4Matrix down(HIDDEN, a.ff_rows);

        const uint64_t weights = gate.weights() + up.weights() + down.weights();
        const size_t encoded = gate.encoded_bytes() + up.encoded_bytes() + down.encoded_bytes();

        std::vector<float> x(HIDDEN);
        for (int i = 0; i < HIDDEN; ++i) x[i] = std::sin(0.001f * i);
        std::vector<float> gate_y, up_y, intermediate, partial_out;

        std::cout << "TensorWave three-R920 K2.5 expert-shard benchmark\n"
                  << "ISA target       : AVX1 (-mavx)\n"
                  << "FF shard rows    : " << a.ff_rows << "\n"
                  << "OpenMP threads   : " << a.threads << "\n"
                  << "Weights/shard    : " << weights << "\n"
                  << "Encoded/shard MB : " << std::fixed << std::setprecision(3)
                  << encoded / 1e6 << "\n";

        for (int i = 0; i < a.warmup; ++i) {
            run_shard(gate, up, down, x, gate_y, up_y, intermediate, partial_out);
        }

        const auto t0 = std::chrono::steady_clock::now();
        double checksum = 0.0;
        for (int i = 0; i < a.iters; ++i) {
            run_shard(gate, up, down, x, gate_y, up_y, intermediate, partial_out);
            checksum += partial_out[(i * 131) % partial_out.size()];
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        const double gweights_s = static_cast<double>(weights) * a.iters / seconds / 1e9;
        const double encoded_gb_s = static_cast<double>(encoded) * a.iters / seconds / 1e9;
        const double average_socket_gweights_per_token =
            K25_ROUTED_ACTIVE_GWEIGHTS_PER_TOKEN / THREE_R920_SOCKETS;
        const double equal_weight_cluster_tps = gweights_s / average_socket_gweights_per_token;

        std::cout << "Seconds           : " << std::setprecision(6) << seconds << "\n"
                  << "Gweights/s        : " << std::setprecision(3) << gweights_s << "\n"
                  << "Encoded Q4 GB/s   : " << encoded_gb_s << "\n"
                  << "12-socket eq tps  : " << equal_weight_cluster_tps << "\n"
                  << "Checksum          : " << checksum << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }
}
