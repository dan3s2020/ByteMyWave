#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                                  \
    do {                                                                                  \
        const cudaError_t _err = (expr);                                                   \
        if (_err != cudaSuccess) {                                                         \
            std::ostringstream _oss;                                                       \
            _oss << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": "              \
                 << cudaGetErrorString(_err);                                              \
            throw std::runtime_error(_oss.str());                                          \
        }                                                                                 \
    } while (0)

#define CUBLAS_CHECK(expr)                                                                \
    do {                                                                                  \
        const cublasStatus_t _status = (expr);                                             \
        if (_status != CUBLAS_STATUS_SUCCESS) {                                            \
            std::ostringstream _oss;                                                       \
            _oss << "cuBLAS error at " << __FILE__ << ':' << __LINE__                    \
                 << ": status=" << static_cast<int>(_status);                             \
            throw std::runtime_error(_oss.str());                                          \
        }                                                                                 \
    } while (0)

constexpr int kGroupSize = 32;
constexpr int kScaleBytes = 4;
constexpr int kPackedValueBytes = 16;
constexpr int kGroupBytes = kScaleBytes + kPackedValueBytes;
constexpr double kSourceBytesPerGroup = kGroupSize * 2.0;
constexpr double kCompressionX = kSourceBytesPerGroup / kGroupBytes;

struct Config {
    int device = 0;
    int m = 512;
    int k = 0;
    int n = 0;
    int tiles = 0;
    int warmup = 3;
    std::string weights_file;
    std::string json_path;
};

struct Metrics {
    double wall_ms = 0.0;
    double copy_ms = 0.0;
    double compute_ms = 0.0;
    double dequant_ms = 0.0;
    double gemm_ms = 0.0;
    double starvation_ms = 0.0;
    double startup_ms = 0.0;
    double compressed_h2d_gbps = 0.0;
    double source_equivalent_h2d_gbps = 0.0;
    double steady_hidden_transfer_pct = 0.0;
    double steady_starvation_pct = 0.0;
};

struct RunResult {
    Metrics sequential;
    Metrics overlapped;
    double speedup = 0.0;
    double max_abs_error = 0.0;
    double rms_error = 0.0;
    bool correctness_ok = false;
    std::string verdict;
};

__global__ void dequant_q4_g32_f32s(
    const unsigned char* __restrict__ input,
    __half* __restrict__ output,
    std::size_t element_count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= element_count) {
        return;
    }

    const std::size_t group = index / kGroupSize;
    const int lane = static_cast<int>(index % kGroupSize);
    const unsigned char* record = input + group * kGroupBytes;

    // Every record starts at a 4-byte boundary because GROUP_BYTES=20.
    const float scale = *reinterpret_cast<const float*>(record);
    const unsigned char packed = record[kScaleBytes + lane / 2];
    const unsigned int nibble =
        (lane & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4u) & 0x0Fu);
    const int q = nibble < 8u ? static_cast<int>(nibble)
                              : static_cast<int>(nibble) - 16;
    output[index] = __float2half_rn(static_cast<float>(q) * scale);
}

void print_help() {
    std::cout
        << "TensorWave Phase-3 Q4 streaming proof\n\n"
        << "Usage:\n"
        << "  tensorwave_q4_stream_proof --weights-file PATH --k K --n N [options]\n\n"
        << "Q4 format:\n"
        << "  Q4_SYM_G32_F32S: 32 weights/group, float32 scale + 16 int4 bytes\n"
        << "  20 bytes per 32 weights = 5.0 effective bits/weight = 3.2x smaller than FP16\n\n"
        << "Required:\n"
        << "  --weights-file PATH  weights-q4.pack from tools/quantize_q4_pack.py\n"
        << "  --k N                Source weight K dimension\n"
        << "  --n N                Source weight tile row/output dimension\n\n"
        << "Options:\n"
        << "  --device N           CUDA device index (default 0)\n"
        << "  --m N                Activation rows (default 512)\n"
        << "  --tiles N            Q4 tiles; 0 = infer all from file\n"
        << "  --warmup N           Warmup dequant+GEMM iterations (default 3)\n"
        << "  --json PATH          Write result JSON\n"
        << "  --help               Show this message\n";
}

int parse_nonnegative(const char* value, const char* name, bool allow_zero) {
    try {
        const int parsed = std::stoi(value);
        if (parsed < 0 || (!allow_zero && parsed == 0)) {
            throw std::runtime_error("range");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid integer for ") + name + ": " + value);
    }
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value after ") + name);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_help();
            std::exit(0);
        } else if (arg == "--device") {
            cfg.device = parse_nonnegative(value("--device"), "--device", true);
        } else if (arg == "--m") {
            cfg.m = parse_nonnegative(value("--m"), "--m", false);
        } else if (arg == "--k") {
            cfg.k = parse_nonnegative(value("--k"), "--k", false);
        } else if (arg == "--n") {
            cfg.n = parse_nonnegative(value("--n"), "--n", false);
        } else if (arg == "--tiles") {
            cfg.tiles = parse_nonnegative(value("--tiles"), "--tiles", true);
        } else if (arg == "--warmup") {
            cfg.warmup = parse_nonnegative(value("--warmup"), "--warmup", true);
        } else if (arg == "--weights-file") {
            cfg.weights_file = value("--weights-file");
        } else if (arg == "--json") {
            cfg.json_path = value("--json");
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.weights_file.empty()) {
        throw std::runtime_error("--weights-file is required");
    }
    if (cfg.k <= 0 || cfg.n <= 0) {
        throw std::runtime_error("--k and --n are required and must be > 0");
    }
    const std::size_t elements = static_cast<std::size_t>(cfg.k) * cfg.n;
    if (elements % kGroupSize != 0) {
        throw std::runtime_error("K*N must be divisible by Q4 group size 32");
    }
    return cfg;
}

std::vector<cudaEvent_t> make_events(int count) {
    std::vector<cudaEvent_t> events(static_cast<std::size_t>(count), nullptr);
    for (auto& event : events) {
        CUDA_CHECK(cudaEventCreate(&event));
    }
    return events;
}

void destroy_events(std::vector<cudaEvent_t>& events) {
    for (auto event : events) {
        if (event != nullptr) {
            cudaEventDestroy(event);
        }
    }
    events.clear();
}

double event_ms(cudaEvent_t start, cudaEvent_t end) {
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    return static_cast<double>(milliseconds);
}

std::string json_escape(const std::string& input) {
    std::ostringstream out;
    for (const char c : input) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

void launch_dequant(
    cudaStream_t stream,
    const void* d_q4,
    __half* d_weight,
    std::size_t elements) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    dequant_q4_g32_f32s<<<blocks, threads, 0, stream>>>(
        static_cast<const unsigned char*>(d_q4), d_weight, elements);
    CUDA_CHECK(cudaGetLastError());
}

void gemm_accumulate(
    cublasHandle_t handle,
    const Config& cfg,
    const __half* d_x,
    const __half* d_weight,
    float* d_y,
    float beta) {
    const float alpha = 1.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        cfg.m,
        cfg.n,
        cfg.k,
        &alpha,
        d_x,
        CUDA_R_16F,
        cfg.m,
        d_weight,
        CUDA_R_16F,
        cfg.k,
        &beta,
        d_y,
        CUDA_R_32F,
        cfg.m,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

Metrics run_sequential(
    const Config& cfg,
    cublasHandle_t handle,
    cudaStream_t stream,
    const unsigned char* h_q4,
    std::size_t q4_tile_bytes,
    std::size_t tile_elements,
    std::size_t source_tile_bytes,
    const __half* d_x,
    void* d_q4,
    __half* d_weight,
    float* d_y) {
    auto copy_start = make_events(cfg.tiles);
    auto copy_end = make_events(cfg.tiles);
    auto compute_start = make_events(cfg.tiles);
    auto dequant_end = make_events(cfg.tiles);
    auto compute_end = make_events(cfg.tiles);

    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUDA_CHECK(cudaMemsetAsync(
        d_y, 0, static_cast<std::size_t>(cfg.m) * cfg.n * sizeof(float), stream));

    const auto wall_start = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.tiles; ++i) {
        const auto* src = h_q4 + static_cast<std::size_t>(i) * q4_tile_bytes;

        CUDA_CHECK(cudaEventRecord(copy_start[i], stream));
        CUDA_CHECK(cudaMemcpyAsync(
            d_q4, src, q4_tile_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(copy_end[i], stream));

        CUDA_CHECK(cudaEventRecord(compute_start[i], stream));
        launch_dequant(stream, d_q4, d_weight, tile_elements);
        CUDA_CHECK(cudaEventRecord(dequant_end[i], stream));
        gemm_accumulate(handle, cfg, d_x, d_weight, d_y, 1.0f);
        CUDA_CHECK(cudaEventRecord(compute_end[i], stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto wall_end = std::chrono::steady_clock::now();

    Metrics metrics;
    metrics.wall_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    for (int i = 0; i < cfg.tiles; ++i) {
        metrics.copy_ms += event_ms(copy_start[i], copy_end[i]);
        metrics.compute_ms += event_ms(compute_start[i], compute_end[i]);
        metrics.dequant_ms += event_ms(compute_start[i], dequant_end[i]);
        metrics.gemm_ms += event_ms(dequant_end[i], compute_end[i]);
    }

    const double q4_gb = static_cast<double>(q4_tile_bytes) * cfg.tiles / 1.0e9;
    const double source_gb = static_cast<double>(source_tile_bytes) * cfg.tiles / 1.0e9;
    if (metrics.copy_ms > 0.0) {
        const double seconds = metrics.copy_ms / 1000.0;
        metrics.compressed_h2d_gbps = q4_gb / seconds;
        metrics.source_equivalent_h2d_gbps = source_gb / seconds;
    }

    destroy_events(copy_start);
    destroy_events(copy_end);
    destroy_events(compute_start);
    destroy_events(dequant_end);
    destroy_events(compute_end);
    return metrics;
}

Metrics run_overlapped(
    const Config& cfg,
    cublasHandle_t handle,
    cudaStream_t copy_stream,
    cudaStream_t compute_stream,
    const unsigned char* h_q4,
    std::size_t q4_tile_bytes,
    std::size_t tile_elements,
    std::size_t source_tile_bytes,
    const __half* d_x,
    void* d_q40,
    void* d_q41,
    __half* d_weight,
    float* d_y) {
    auto copy_start = make_events(cfg.tiles);
    auto copy_end = make_events(cfg.tiles);
    auto compute_start = make_events(cfg.tiles);
    auto dequant_end = make_events(cfg.tiles);
    auto compute_end = make_events(cfg.tiles);

    void* slots[2] = {d_q40, d_q41};
    CUBLAS_CHECK(cublasSetStream(handle, compute_stream));
    CUDA_CHECK(cudaMemsetAsync(
        d_y, 0, static_cast<std::size_t>(cfg.m) * cfg.n * sizeof(float), compute_stream));

    const auto wall_start = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.tiles; ++i) {
        const int slot = i & 1;
        const auto* src = h_q4 + static_cast<std::size_t>(i) * q4_tile_bytes;

        // Same ownership contract as Phase 1/2, now for compressed slots.
        if (i >= 2) {
            CUDA_CHECK(cudaStreamWaitEvent(copy_stream, compute_end[i - 2], 0));
        }

        CUDA_CHECK(cudaEventRecord(copy_start[i], copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(
            slots[slot], src, q4_tile_bytes, cudaMemcpyHostToDevice, copy_stream));
        CUDA_CHECK(cudaEventRecord(copy_end[i], copy_stream));

        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, copy_end[i], 0));
        CUDA_CHECK(cudaEventRecord(compute_start[i], compute_stream));
        launch_dequant(compute_stream, slots[slot], d_weight, tile_elements);
        CUDA_CHECK(cudaEventRecord(dequant_end[i], compute_stream));
        gemm_accumulate(handle, cfg, d_x, d_weight, d_y, 1.0f);
        CUDA_CHECK(cudaEventRecord(compute_end[i], compute_stream));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto wall_end = std::chrono::steady_clock::now();

    Metrics metrics;
    metrics.wall_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    double steady_copy_ms = 0.0;
    for (int i = 0; i < cfg.tiles; ++i) {
        const double copy = event_ms(copy_start[i], copy_end[i]);
        metrics.copy_ms += copy;
        metrics.compute_ms += event_ms(compute_start[i], compute_end[i]);
        metrics.dequant_ms += event_ms(compute_start[i], dequant_end[i]);
        metrics.gemm_ms += event_ms(dequant_end[i], compute_end[i]);
        if (i > 0) {
            steady_copy_ms += copy;
            metrics.starvation_ms += event_ms(compute_end[i - 1], compute_start[i]);
        }
    }

    metrics.startup_ms = event_ms(copy_start[0], compute_start[0]);
    const double q4_gb = static_cast<double>(q4_tile_bytes) * cfg.tiles / 1.0e9;
    const double source_gb = static_cast<double>(source_tile_bytes) * cfg.tiles / 1.0e9;
    if (metrics.copy_ms > 0.0) {
        const double seconds = metrics.copy_ms / 1000.0;
        metrics.compressed_h2d_gbps = q4_gb / seconds;
        metrics.source_equivalent_h2d_gbps = source_gb / seconds;
    }
    if (steady_copy_ms > 0.0) {
        const double hidden = 1.0 - metrics.starvation_ms / steady_copy_ms;
        metrics.steady_hidden_transfer_pct =
            std::clamp(hidden * 100.0, 0.0, 100.0);
    }
    const double active = metrics.compute_ms + metrics.starvation_ms;
    if (active > 0.0) {
        metrics.steady_starvation_pct = 100.0 * metrics.starvation_ms / active;
    }

    destroy_events(copy_start);
    destroy_events(copy_end);
    destroy_events(compute_start);
    destroy_events(dequant_end);
    destroy_events(compute_end);
    return metrics;
}

void compare_outputs(
    const std::vector<float>& reference,
    const std::vector<float>& candidate,
    RunResult& result) {
    if (reference.size() != candidate.size()) {
        throw std::runtime_error("internal output-size mismatch");
    }

    long double squared = 0.0L;
    double max_abs = 0.0;
    bool finite = true;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(candidate[i])) {
            finite = false;
            continue;
        }
        const double diff =
            std::abs(static_cast<double>(reference[i]) - candidate[i]);
        max_abs = std::max(max_abs, diff);
        squared += static_cast<long double>(diff) * diff;
    }

    result.max_abs_error = max_abs;
    result.rms_error = reference.empty()
        ? 0.0
        : std::sqrt(static_cast<double>(squared / reference.size()));
    result.correctness_ok = finite && result.max_abs_error <= 1.0e-3;
}

void write_metrics(std::ofstream& out, const char* name, const Metrics& m, bool comma) {
    out << "  \"" << name << "\": {\n";
    out << "    \"wall_ms\": " << m.wall_ms << ",\n";
    out << "    \"copy_ms\": " << m.copy_ms << ",\n";
    out << "    \"compute_ms\": " << m.compute_ms << ",\n";
    out << "    \"dequant_ms\": " << m.dequant_ms << ",\n";
    out << "    \"gemm_ms\": " << m.gemm_ms << ",\n";
    out << "    \"starvation_ms\": " << m.starvation_ms << ",\n";
    out << "    \"startup_ms\": " << m.startup_ms << ",\n";
    out << "    \"compressed_h2d_gbps\": " << m.compressed_h2d_gbps << ",\n";
    out << "    \"source_equivalent_h2d_gbps\": " << m.source_equivalent_h2d_gbps << ",\n";
    out << "    \"steady_hidden_transfer_pct\": " << m.steady_hidden_transfer_pct << ",\n";
    out << "    \"steady_starvation_pct\": " << m.steady_starvation_pct << "\n";
    out << "  }" << (comma ? "," : "") << "\n";
}

void write_json(
    const std::string& path,
    const Config& cfg,
    const cudaDeviceProp& prop,
    std::size_t q4_pack_bytes,
    std::size_t q4_tile_bytes,
    std::size_t source_tile_bytes,
    std::size_t fixed_vram_bytes,
    const RunResult& result) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open JSON output: " + path);
    }
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"experiment\": \"phase3-q4-gpu-dequant-streaming\",\n";
    out << "  \"weights_file\": \"" << json_escape(cfg.weights_file) << "\",\n";
    out << "  \"quantization\": \"Q4_SYM_G32_F32S\",\n";
    out << "  \"compression_x_vs_16bit\": " << kCompressionX << ",\n";
    out << "  \"device\": {\n";
    out << "    \"name\": \"" << json_escape(prop.name) << "\",\n";
    out << "    \"compute_capability\": \"" << prop.major << '.' << prop.minor << "\",\n";
    out << "    \"total_vram_bytes\": "
        << static_cast<unsigned long long>(prop.totalGlobalMem) << ",\n";
    out << "    \"async_engine_count\": " << prop.asyncEngineCount << ",\n";
    out << "    \"device_overlap\": " << (prop.deviceOverlap ? "true" : "false") << "\n";
    out << "  },\n";
    out << "  \"geometry\": {\n";
    out << "    \"m\": " << cfg.m << ",\n";
    out << "    \"k\": " << cfg.k << ",\n";
    out << "    \"n\": " << cfg.n << ",\n";
    out << "    \"tiles\": " << cfg.tiles << ",\n";
    out << "    \"q4_tile_bytes\": " << static_cast<unsigned long long>(q4_tile_bytes) << ",\n";
    out << "    \"source_16bit_tile_bytes\": "
        << static_cast<unsigned long long>(source_tile_bytes) << ",\n";
    out << "    \"q4_pack_bytes_used\": "
        << static_cast<unsigned long long>(q4_pack_bytes) << ",\n";
    out << "    \"fixed_vram_working_set_bytes\": "
        << static_cast<unsigned long long>(fixed_vram_bytes) << "\n";
    out << "  },\n";
    write_metrics(out, "sequential", result.sequential, true);
    write_metrics(out, "overlapped", result.overlapped, true);
    out << "  \"speedup\": " << result.speedup << ",\n";
    out << "  \"max_abs_error\": " << result.max_abs_error << ",\n";
    out << "  \"rms_error\": " << result.rms_error << ",\n";
    out << "  \"correctness_ok\": " << (result.correctness_ok ? "true" : "false") << ",\n";
    out << "  \"verdict\": \"" << json_escape(result.verdict) << "\"\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);
        CUDA_CHECK(cudaSetDevice(cfg.device));

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));

        const std::size_t tile_elements = static_cast<std::size_t>(cfg.k) * cfg.n;
        const std::size_t source_tile_bytes = tile_elements * sizeof(__half);
        const std::size_t groups_per_tile = tile_elements / kGroupSize;
        const std::size_t q4_tile_bytes = groups_per_tile * kGroupBytes;

        const std::uintmax_t file_bytes_u = std::filesystem::file_size(cfg.weights_file);
        if (file_bytes_u > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("Q4 pack is too large for this process");
        }
        const std::size_t file_bytes = static_cast<std::size_t>(file_bytes_u);
        if (file_bytes == 0 || file_bytes % q4_tile_bytes != 0) {
            throw std::runtime_error("Q4 pack size is not an exact multiple of Q4 tile bytes");
        }
        const std::size_t available_tiles = file_bytes / q4_tile_bytes;
        if (cfg.tiles == 0) {
            if (available_tiles > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("too many Q4 tiles for current experiment counter");
            }
            cfg.tiles = static_cast<int>(available_tiles);
        }
        if (cfg.tiles < 2 || static_cast<std::size_t>(cfg.tiles) > available_tiles) {
            throw std::runtime_error("--tiles must be between 2 and Q4 packed tile count");
        }
        const std::size_t q4_pack_bytes = static_cast<std::size_t>(cfg.tiles) * q4_tile_bytes;

        unsigned char* h_q4 = nullptr;
        CUDA_CHECK(cudaHostAlloc(
            reinterpret_cast<void**>(&h_q4), q4_pack_bytes, cudaHostAllocDefault));
        {
            std::ifstream input(cfg.weights_file, std::ios::binary);
            if (!input) {
                cudaFreeHost(h_q4);
                throw std::runtime_error("could not open Q4 pack: " + cfg.weights_file);
            }
            input.read(reinterpret_cast<char*>(h_q4), static_cast<std::streamsize>(q4_pack_bytes));
            if (input.gcount() != static_cast<std::streamsize>(q4_pack_bytes)) {
                cudaFreeHost(h_q4);
                throw std::runtime_error("short read while loading Q4 pack into pinned RAM");
            }
        }

        const std::size_t x_elements = static_cast<std::size_t>(cfg.m) * cfg.k;
        const std::size_t x_bytes = x_elements * sizeof(__half);
        const std::size_t y_elements = static_cast<std::size_t>(cfg.m) * cfg.n;
        const std::size_t y_bytes = y_elements * sizeof(float);

        std::vector<__half> h_x(x_elements);
        for (std::size_t i = 0; i < h_x.size(); ++i) {
            const int centered = static_cast<int>((i * 17 + 13) % 257) - 128;
            h_x[i] = __float2half(static_cast<float>(centered) / 16384.0f);
        }

        __half* d_x = nullptr;
        void* d_q40 = nullptr;
        void* d_q41 = nullptr;
        __half* d_weight = nullptr;
        float* d_y = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_x), x_bytes));
        CUDA_CHECK(cudaMalloc(&d_q40, q4_tile_bytes));
        CUDA_CHECK(cudaMalloc(&d_q41, q4_tile_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_weight), source_tile_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_y), y_bytes));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_bytes, cudaMemcpyHostToDevice));

        cudaStream_t copy_stream = nullptr;
        cudaStream_t compute_stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));

        cublasHandle_t handle = nullptr;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
        CUBLAS_CHECK(cublasSetStream(handle, compute_stream));

        CUDA_CHECK(cudaMemcpyAsync(
            d_q40, h_q4, q4_tile_bytes, cudaMemcpyHostToDevice, compute_stream));
        for (int i = 0; i < cfg.warmup; ++i) {
            launch_dequant(compute_stream, d_q40, d_weight, tile_elements);
            gemm_accumulate(handle, cfg, d_x, d_weight, d_y, i == 0 ? 0.0f : 1.0f);
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        RunResult result;
        result.sequential = run_sequential(
            cfg,
            handle,
            compute_stream,
            h_q4,
            q4_tile_bytes,
            tile_elements,
            source_tile_bytes,
            d_x,
            d_q40,
            d_weight,
            d_y);

        std::vector<float> sequential_output(y_elements);
        CUDA_CHECK(cudaMemcpy(
            sequential_output.data(), d_y, y_bytes, cudaMemcpyDeviceToHost));

        result.overlapped = run_overlapped(
            cfg,
            handle,
            copy_stream,
            compute_stream,
            h_q4,
            q4_tile_bytes,
            tile_elements,
            source_tile_bytes,
            d_x,
            d_q40,
            d_q41,
            d_weight,
            d_y);

        std::vector<float> overlapped_output(y_elements);
        CUDA_CHECK(cudaMemcpy(
            overlapped_output.data(), d_y, y_bytes, cudaMemcpyDeviceToHost));
        compare_outputs(sequential_output, overlapped_output, result);

        if (result.overlapped.wall_ms > 0.0) {
            result.speedup = result.sequential.wall_ms / result.overlapped.wall_ms;
        }

        const bool strong_overlap =
            result.overlapped.steady_starvation_pct <= 10.0 &&
            result.overlapped.steady_hidden_transfer_pct >= 80.0;
        if (!result.correctness_ok) {
            result.verdict = "CORRECTNESS_FAILED";
        } else if (strong_overlap) {
            result.verdict = "Q4_STREAMING_HYPOTHESIS_SUPPORTED_FOR_SHAPE";
        } else {
            result.verdict = "Q4_OVERLAP_INSUFFICIENT_FOR_SHAPE";
        }

        const std::size_t fixed_vram_bytes =
            x_bytes + 2 * q4_tile_bytes + source_tile_bytes + y_bytes;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "TensorWave Phase-3 Q4 proof\n";
        std::cout << "  GPU:                       " << prop.name << "\n";
        std::cout << "  M/K/N:                     " << cfg.m << '/' << cfg.k << '/' << cfg.n << "\n";
        std::cout << "  tiles:                     " << cfg.tiles << "\n";
        std::cout << "  source tile MiB:           " << source_tile_bytes / (1024.0 * 1024.0) << "\n";
        std::cout << "  Q4 tile MiB:               " << q4_tile_bytes / (1024.0 * 1024.0) << "\n";
        std::cout << "  Q4 host pack MiB:          " << q4_pack_bytes / (1024.0 * 1024.0) << "\n";
        std::cout << "  compression vs 16-bit:     " << kCompressionX << "x\n";
        std::cout << "  fixed VRAM MiB:            " << fixed_vram_bytes / (1024.0 * 1024.0) << "\n\n";
        std::cout << "Sequential wall:             " << result.sequential.wall_ms << " ms\n";
        std::cout << "Overlapped wall:             " << result.overlapped.wall_ms << " ms\n";
        std::cout << "Compressed H2D:              " << result.overlapped.compressed_h2d_gbps << " GB/s\n";
        std::cout << "16-bit-equivalent feed rate: " << result.overlapped.source_equivalent_h2d_gbps << " GB/s\n";
        std::cout << "Dequant sum:                 " << result.overlapped.dequant_ms << " ms\n";
        std::cout << "GEMM sum:                    " << result.overlapped.gemm_ms << " ms\n";
        std::cout << "Compute sum:                 " << result.overlapped.compute_ms << " ms\n";
        std::cout << "GPU starvation sum:          " << result.overlapped.starvation_ms << " ms\n";
        std::cout << "Steady starvation:           " << result.overlapped.steady_starvation_pct << " %\n";
        std::cout << "Transfer hidden estimate:    " << result.overlapped.steady_hidden_transfer_pct << " %\n";
        std::cout << "Pipeline speedup:            " << result.speedup << "x\n";
        std::cout << "Sequential-vs-overlap max:   " << result.max_abs_error << "\n";
        std::cout << "Sequential-vs-overlap RMS:   " << result.rms_error << "\n";
        std::cout << "Verdict:                     " << result.verdict << "\n";

        if (!cfg.json_path.empty()) {
            write_json(
                cfg.json_path,
                cfg,
                prop,
                q4_pack_bytes,
                q4_tile_bytes,
                source_tile_bytes,
                fixed_vram_bytes,
                result);
            std::cout << "JSON:                        " << cfg.json_path << "\n";
        }

        cublasDestroy(handle);
        cudaStreamDestroy(copy_stream);
        cudaStreamDestroy(compute_stream);
        cudaFree(d_x);
        cudaFree(d_q40);
        cudaFree(d_q41);
        cudaFree(d_weight);
        cudaFree(d_y);
        cudaFreeHost(h_q4);

        return result.correctness_ok ? 0 : 3;
    } catch (const std::exception& exc) {
        std::cerr << "ERROR: " << exc.what() << '\n';
        return 1;
    }
}
