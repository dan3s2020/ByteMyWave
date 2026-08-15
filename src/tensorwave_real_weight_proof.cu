#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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

enum class DType {
    FP16,
    BF16,
};

struct Config {
    int device = 0;
    int m = 512;
    int k = 0;
    int n = 0;
    int tiles = 0;
    int warmup = 3;
    DType dtype = DType::BF16;
    std::string weights_file;
    std::string json_path;
};

struct Metrics {
    double wall_ms = 0.0;
    double copy_ms = 0.0;
    double compute_ms = 0.0;
    double starvation_ms = 0.0;
    double startup_ms = 0.0;
    double h2d_gbps = 0.0;
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

std::string dtype_name(DType dtype) {
    return dtype == DType::FP16 ? "F16" : "BF16";
}

cudaDataType_t cuda_dtype(DType dtype) {
    return dtype == DType::FP16 ? CUDA_R_16F : CUDA_R_16BF;
}

void print_help() {
    std::cout
        << "TensorWave Phase-2 real-weight streaming proof\n\n"
        << "Usage:\n"
        << "  tensorwave_real_weight_proof --weights-file PATH --dtype bf16|fp16 "
           "--k K --n N [options]\n\n"
        << "Required:\n"
        << "  --weights-file PATH  Flat tile pack produced by tools/pack_stream_tiles.py\n"
        << "  --dtype TYPE         bf16 or fp16; must match the execution plan\n"
        << "  --k N                GEMM K / source tensor second dimension\n"
        << "  --n N                Rows/output channels per packed tile\n\n"
        << "Options:\n"
        << "  --device N           CUDA device index (default 0)\n"
        << "  --m N                Activation rows (default 512)\n"
        << "  --tiles N            Number of packed tiles; 0 = infer/use all\n"
        << "  --warmup N           Warmup GEMMs (default 3)\n"
        << "  --json PATH          Write machine-readable result JSON\n"
        << "  --help               Show this message\n\n"
        << "The raw safetensors row layout [N,K] is byte-identical to the column-major\n"
        << "cuBLAS matrix [K,N], so the checkpoint bytes are used without transposition.\n";
}

int parse_positive(const char* value, const char* name, bool allow_zero = false) {
    try {
        const int parsed = std::stoi(value);
        if (parsed < 0 || (!allow_zero && parsed == 0)) {
            throw std::runtime_error("range");
        }
        return parsed;
    } catch (const std::exception&) {
        std::ostringstream oss;
        oss << "Invalid integer for " << name << ": " << value;
        throw std::runtime_error(oss.str());
    }
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value after ") + name);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_help();
            std::exit(0);
        } else if (arg == "--device") {
            cfg.device = parse_positive(require_value("--device"), "--device", true);
        } else if (arg == "--m") {
            cfg.m = parse_positive(require_value("--m"), "--m");
        } else if (arg == "--k") {
            cfg.k = parse_positive(require_value("--k"), "--k");
        } else if (arg == "--n") {
            cfg.n = parse_positive(require_value("--n"), "--n");
        } else if (arg == "--tiles") {
            cfg.tiles = parse_positive(require_value("--tiles"), "--tiles", true);
        } else if (arg == "--warmup") {
            cfg.warmup = parse_positive(require_value("--warmup"), "--warmup", true);
        } else if (arg == "--weights-file") {
            cfg.weights_file = require_value("--weights-file");
        } else if (arg == "--json") {
            cfg.json_path = require_value("--json");
        } else if (arg == "--dtype") {
            const std::string value = require_value("--dtype");
            if (value == "fp16" || value == "f16" || value == "F16") {
                cfg.dtype = DType::FP16;
            } else if (value == "bf16" || value == "BF16") {
                cfg.dtype = DType::BF16;
            } else {
                throw std::runtime_error("--dtype must be bf16 or fp16");
            }
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
    return cfg;
}

std::vector<cudaEvent_t> make_events(int count) {
    std::vector<cudaEvent_t> events(static_cast<size_t>(count), nullptr);
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
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    return static_cast<double>(ms);
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

template <typename T>
void fill_activation(std::vector<T>& values) {
    for (size_t i = 0; i < values.size(); ++i) {
        const int centered = static_cast<int>((i * 17 + 13) % 257) - 128;
        const float value = static_cast<float>(centered) / 16384.0f;
        if constexpr (std::is_same_v<T, __half>) {
            values[i] = __float2half(value);
        } else {
            values[i] = __float2bfloat16(value);
        }
    }
}

void gemm_accumulate(cublasHandle_t handle,
                     const Config& cfg,
                     const void* d_x,
                     const void* d_w,
                     float* d_y,
                     float beta) {
    const float alpha = 1.0f;
    const cudaDataType_t input_type = cuda_dtype(cfg.dtype);

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        cfg.m,
        cfg.n,
        cfg.k,
        &alpha,
        d_x,
        input_type,
        cfg.m,
        d_w,
        input_type,
        cfg.k,
        &beta,
        d_y,
        CUDA_R_32F,
        cfg.m,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

Metrics run_sequential(const Config& cfg,
                       cublasHandle_t handle,
                       cudaStream_t stream,
                       const std::uint8_t* h_weights,
                       size_t tile_bytes,
                       const void* d_x,
                       void* d_w,
                       float* d_y) {
    auto copy_start = make_events(cfg.tiles);
    auto copy_end = make_events(cfg.tiles);
    auto compute_start = make_events(cfg.tiles);
    auto compute_end = make_events(cfg.tiles);

    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUDA_CHECK(cudaMemsetAsync(
        d_y, 0, static_cast<size_t>(cfg.m) * cfg.n * sizeof(float), stream));

    const auto wall_start = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.tiles; ++i) {
        const auto* src = h_weights + static_cast<size_t>(i) * tile_bytes;
        CUDA_CHECK(cudaEventRecord(copy_start[static_cast<size_t>(i)], stream));
        CUDA_CHECK(cudaMemcpyAsync(d_w, src, tile_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(copy_end[static_cast<size_t>(i)], stream));

        CUDA_CHECK(cudaEventRecord(compute_start[static_cast<size_t>(i)], stream));
        gemm_accumulate(handle, cfg, d_x, d_w, d_y, 1.0f);
        CUDA_CHECK(cudaEventRecord(compute_end[static_cast<size_t>(i)], stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto wall_end = std::chrono::steady_clock::now();

    Metrics metrics;
    metrics.wall_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    for (int i = 0; i < cfg.tiles; ++i) {
        metrics.copy_ms += event_ms(copy_start[i], copy_end[i]);
        metrics.compute_ms += event_ms(compute_start[i], compute_end[i]);
    }
    const double gb = static_cast<double>(tile_bytes) * cfg.tiles / 1.0e9;
    if (metrics.copy_ms > 0.0) {
        metrics.h2d_gbps = gb / (metrics.copy_ms / 1000.0);
    }

    destroy_events(copy_start);
    destroy_events(copy_end);
    destroy_events(compute_start);
    destroy_events(compute_end);
    return metrics;
}

Metrics run_overlapped(const Config& cfg,
                       cublasHandle_t handle,
                       cudaStream_t copy_stream,
                       cudaStream_t compute_stream,
                       const std::uint8_t* h_weights,
                       size_t tile_bytes,
                       const void* d_x,
                       void* d_w0,
                       void* d_w1,
                       float* d_y) {
    auto copy_start = make_events(cfg.tiles);
    auto copy_end = make_events(cfg.tiles);
    auto compute_start = make_events(cfg.tiles);
    auto compute_end = make_events(cfg.tiles);

    void* slots[2] = {d_w0, d_w1};
    CUBLAS_CHECK(cublasSetStream(handle, compute_stream));
    CUDA_CHECK(cudaMemsetAsync(
        d_y, 0, static_cast<size_t>(cfg.m) * cfg.n * sizeof(float), compute_stream));

    const auto wall_start = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.tiles; ++i) {
        const int slot = i & 1;
        const auto* src = h_weights + static_cast<size_t>(i) * tile_bytes;

        if (i >= 2) {
            CUDA_CHECK(cudaStreamWaitEvent(copy_stream, compute_end[i - 2], 0));
        }

        CUDA_CHECK(cudaEventRecord(copy_start[i], copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(
            slots[slot], src, tile_bytes, cudaMemcpyHostToDevice, copy_stream));
        CUDA_CHECK(cudaEventRecord(copy_end[i], copy_stream));

        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, copy_end[i], 0));
        CUDA_CHECK(cudaEventRecord(compute_start[i], compute_stream));
        gemm_accumulate(handle, cfg, d_x, slots[slot], d_y, 1.0f);
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
        const double compute = event_ms(compute_start[i], compute_end[i]);
        metrics.copy_ms += copy;
        metrics.compute_ms += compute;
        if (i > 0) {
            steady_copy_ms += copy;
            metrics.starvation_ms += event_ms(compute_end[i - 1], compute_start[i]);
        }
    }

    metrics.startup_ms = event_ms(copy_start[0], compute_start[0]);
    const double gb = static_cast<double>(tile_bytes) * cfg.tiles / 1.0e9;
    if (metrics.copy_ms > 0.0) {
        metrics.h2d_gbps = gb / (metrics.copy_ms / 1000.0);
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
    destroy_events(compute_end);
    return metrics;
}

void compare_outputs(const std::vector<float>& reference,
                     const std::vector<float>& candidate,
                     RunResult& result) {
    if (reference.size() != candidate.size()) {
        throw std::runtime_error("internal output-size mismatch");
    }

    long double squared_sum = 0.0L;
    double max_abs = 0.0;
    bool finite = true;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(candidate[i])) {
            finite = false;
            continue;
        }
        const double diff =
            std::abs(static_cast<double>(reference[i]) - candidate[i]);
        max_abs = std::max(max_abs, diff);
        squared_sum += static_cast<long double>(diff) * diff;
    }

    result.max_abs_error = max_abs;
    result.rms_error = reference.empty()
        ? 0.0
        : std::sqrt(static_cast<double>(squared_sum / reference.size()));
    result.correctness_ok = finite && result.max_abs_error <= 1.0e-3;
}

void write_json(const std::string& path,
                const Config& cfg,
                const cudaDeviceProp& prop,
                size_t pack_bytes,
                size_t tile_bytes,
                size_t fixed_vram_bytes,
                const RunResult& result) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open JSON output: " + path);
    }

    auto emit_metrics = [&](const char* name, const Metrics& m, bool comma) {
        out << "  \"" << name << "\": {\n";
        out << "    \"wall_ms\": " << m.wall_ms << ",\n";
        out << "    \"copy_ms\": " << m.copy_ms << ",\n";
        out << "    \"compute_ms\": " << m.compute_ms << ",\n";
        out << "    \"starvation_ms\": " << m.starvation_ms << ",\n";
        out << "    \"startup_ms\": " << m.startup_ms << ",\n";
        out << "    \"h2d_gbps\": " << m.h2d_gbps << ",\n";
        out << "    \"steady_hidden_transfer_pct\": "
            << m.steady_hidden_transfer_pct << ",\n";
        out << "    \"steady_starvation_pct\": " << m.steady_starvation_pct << "\n";
        out << "  }" << (comma ? "," : "") << "\n";
    };

    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"experiment\": \"phase2-real-weight-streaming\",\n";
    out << "  \"weights_file\": \"" << json_escape(cfg.weights_file) << "\",\n";
    out << "  \"dtype\": \"" << dtype_name(cfg.dtype) << "\",\n";
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
    out << "    \"tile_bytes\": " << static_cast<unsigned long long>(tile_bytes) << ",\n";
    out << "    \"pack_bytes_used\": " << static_cast<unsigned long long>(pack_bytes) << ",\n";
    out << "    \"fixed_vram_working_set_bytes\": "
        << static_cast<unsigned long long>(fixed_vram_bytes) << "\n";
    out << "  },\n";
    emit_metrics("sequential", result.sequential, true);
    emit_metrics("overlapped", result.overlapped, true);
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

        const size_t element_bytes = 2;
        const size_t tile_elements = static_cast<size_t>(cfg.k) * cfg.n;
        const size_t tile_bytes = tile_elements * element_bytes;
        const std::uintmax_t file_bytes_u = std::filesystem::file_size(cfg.weights_file);
        if (file_bytes_u > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("weights pack is too large for this process");
        }
        const size_t file_bytes = static_cast<size_t>(file_bytes_u);
        if (file_bytes == 0 || file_bytes % tile_bytes != 0) {
            throw std::runtime_error(
                "weights pack size is not an exact multiple of K*N*2 bytes");
        }

        const size_t available_tiles = file_bytes / tile_bytes;
        if (cfg.tiles == 0) {
            if (available_tiles > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("too many tiles for current experiment counter");
            }
            cfg.tiles = static_cast<int>(available_tiles);
        }
        if (cfg.tiles < 2 || static_cast<size_t>(cfg.tiles) > available_tiles) {
            throw std::runtime_error("--tiles must be between 2 and the packed tile count");
        }
        const size_t pack_bytes = static_cast<size_t>(cfg.tiles) * tile_bytes;

        std::uint8_t* h_weights = nullptr;
        CUDA_CHECK(cudaHostAlloc(
            reinterpret_cast<void**>(&h_weights), pack_bytes, cudaHostAllocDefault));
        {
            std::ifstream input(cfg.weights_file, std::ios::binary);
            if (!input) {
                cudaFreeHost(h_weights);
                throw std::runtime_error("could not open weights pack: " + cfg.weights_file);
            }
            input.read(reinterpret_cast<char*>(h_weights), static_cast<std::streamsize>(pack_bytes));
            if (input.gcount() != static_cast<std::streamsize>(pack_bytes)) {
                cudaFreeHost(h_weights);
                throw std::runtime_error("short read while loading weights pack into pinned RAM");
            }
        }

        const size_t x_elements = static_cast<size_t>(cfg.m) * cfg.k;
        const size_t x_bytes = x_elements * element_bytes;
        const size_t y_elements = static_cast<size_t>(cfg.m) * cfg.n;
        const size_t y_bytes = y_elements * sizeof(float);

        void* d_x = nullptr;
        void* d_w0 = nullptr;
        void* d_w1 = nullptr;
        float* d_y = nullptr;
        CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
        CUDA_CHECK(cudaMalloc(&d_w0, tile_bytes));
        CUDA_CHECK(cudaMalloc(&d_w1, tile_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_y), y_bytes));

        if (cfg.dtype == DType::FP16) {
            std::vector<__half> h_x(x_elements);
            fill_activation(h_x);
            CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_bytes, cudaMemcpyHostToDevice));
        } else {
            std::vector<__nv_bfloat16> h_x(x_elements);
            fill_activation(h_x);
            CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_bytes, cudaMemcpyHostToDevice));
        }

        cudaStream_t copy_stream = nullptr;
        cudaStream_t compute_stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));

        cublasHandle_t handle = nullptr;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
        CUBLAS_CHECK(cublasSetStream(handle, compute_stream));

        CUDA_CHECK(cudaMemcpyAsync(
            d_w0, h_weights, tile_bytes, cudaMemcpyHostToDevice, compute_stream));
        for (int i = 0; i < cfg.warmup; ++i) {
            gemm_accumulate(handle, cfg, d_x, d_w0, d_y, i == 0 ? 0.0f : 1.0f);
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        RunResult result;
        result.sequential = run_sequential(
            cfg, handle, compute_stream, h_weights, tile_bytes, d_x, d_w0, d_y);

        std::vector<float> sequential_output(y_elements);
        CUDA_CHECK(cudaMemcpy(
            sequential_output.data(), d_y, y_bytes, cudaMemcpyDeviceToHost));

        result.overlapped = run_overlapped(
            cfg,
            handle,
            copy_stream,
            compute_stream,
            h_weights,
            tile_bytes,
            d_x,
            d_w0,
            d_w1,
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
            result.verdict = "HYPOTHESIS_SUPPORTED_FOR_REAL_WEIGHT_SHAPE";
        } else {
            result.verdict = "OVERLAP_INSUFFICIENT_FOR_THIS_REAL_WEIGHT_SHAPE";
        }

        const size_t fixed_vram_bytes = x_bytes + 2 * tile_bytes + y_bytes;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "TensorWave Phase-2 real-weight proof\n";
        std::cout << "  GPU:                    " << prop.name << "\n";
        std::cout << "  dtype:                  " << dtype_name(cfg.dtype) << "\n";
        std::cout << "  M/K/N:                  " << cfg.m << '/' << cfg.k << '/' << cfg.n << "\n";
        std::cout << "  real tiles:             " << cfg.tiles << "\n";
        std::cout << "  tile MiB:               " << tile_bytes / (1024.0 * 1024.0) << "\n";
        std::cout << "  pinned model MiB:       " << pack_bytes / (1024.0 * 1024.0) << "\n";
        std::cout << "  fixed VRAM MiB:         " << fixed_vram_bytes / (1024.0 * 1024.0) << "\n\n";
        std::cout << "Sequential wall:          " << result.sequential.wall_ms << " ms\n";
        std::cout << "Sequential H2D:           " << result.sequential.h2d_gbps << " GB/s\n";
        std::cout << "Overlapped wall:          " << result.overlapped.wall_ms << " ms\n";
        std::cout << "Overlapped H2D:           " << result.overlapped.h2d_gbps << " GB/s\n";
        std::cout << "Compute sum:              " << result.overlapped.compute_ms << " ms\n";
        std::cout << "GPU starvation sum:       " << result.overlapped.starvation_ms << " ms\n";
        std::cout << "Steady starvation:        "
                  << result.overlapped.steady_starvation_pct << " %\n";
        std::cout << "Transfer hidden estimate: "
                  << result.overlapped.steady_hidden_transfer_pct << " %\n";
        std::cout << "Pipeline speedup:         " << result.speedup << "x\n";
        std::cout << "Max abs error:            " << result.max_abs_error << "\n";
        std::cout << "RMS error:                " << result.rms_error << "\n";
        std::cout << "Verdict:                  " << result.verdict << "\n";

        if (!cfg.json_path.empty()) {
            write_json(
                cfg.json_path,
                cfg,
                prop,
                pack_bytes,
                tile_bytes,
                fixed_vram_bytes,
                result);
            std::cout << "JSON:                     " << cfg.json_path << "\n";
        }

        cublasDestroy(handle);
        cudaStreamDestroy(copy_stream);
        cudaStreamDestroy(compute_stream);
        cudaFree(d_x);
        cudaFree(d_w0);
        cudaFree(d_w1);
        cudaFree(d_y);
        cudaFreeHost(h_weights);

        return result.correctness_ok ? 0 : 3;
    } catch (const std::exception& exc) {
        std::cerr << "ERROR: " << exc.what() << '\n';
        return 1;
    }
}
