#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
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

struct Config {
    int device = 0;
    int m = 512;
    int k = 4096;
    int n = 2048;
    int tiles = 32;
    int warmup = 3;
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

void print_help() {
    std::cout
        << "TensorWave Phase-1 streaming proof\n\n"
        << "Usage:\n"
        << "  tensorwave_stream_proof [options]\n\n"
        << "Options:\n"
        << "  --device N      CUDA device index (default 0)\n"
        << "  --m N           GEMM M dimension / activation rows (default 512)\n"
        << "  --k N           GEMM K dimension (default 4096)\n"
        << "  --n N           GEMM N dimension / weight tile width (default 2048)\n"
        << "  --tiles N       Number of host-resident weight tiles (default 32)\n"
        << "  --warmup N      Warmup GEMMs before measurement (default 3)\n"
        << "  --json PATH     Write machine-readable result JSON\n"
        << "  --help           Show this message\n\n"
        << "The experiment compares:\n"
        << "  1) sequential H2D copy -> GEMM\n"
        << "  2) fixed two-slot VRAM ring with H2D(N+1) overlapped with GEMM(N)\n";
}

int parse_int(const char* value, const char* name) {
    try {
        const int parsed = std::stoi(value);
        if (parsed <= 0) {
            throw std::runtime_error(std::string(name) + " must be > 0");
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
            cfg.device = std::stoi(require_value("--device"));
        } else if (arg == "--m") {
            cfg.m = parse_int(require_value("--m"), "--m");
        } else if (arg == "--k") {
            cfg.k = parse_int(require_value("--k"), "--k");
        } else if (arg == "--n") {
            cfg.n = parse_int(require_value("--n"), "--n");
        } else if (arg == "--tiles") {
            cfg.tiles = parse_int(require_value("--tiles"), "--tiles");
        } else if (arg == "--warmup") {
            cfg.warmup = parse_int(require_value("--warmup"), "--warmup");
        } else if (arg == "--json") {
            cfg.json_path = require_value("--json");
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.device < 0) {
        throw std::runtime_error("--device must be >= 0");
    }
    if (cfg.tiles < 2) {
        throw std::runtime_error("--tiles must be >= 2 for double buffering");
    }
    return cfg;
}

std::vector<cudaEvent_t> make_events(int count) {
    std::vector<cudaEvent_t> events(static_cast<size_t>(count));
    for (auto& event : events) {
        CUDA_CHECK(cudaEventCreate(&event));
    }
    return events;
}

void destroy_events(std::vector<cudaEvent_t>& events) {
    for (auto event : events) {
        if (event) {
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

void gemm_tile(cublasHandle_t handle,
               const Config& cfg,
               const __half* d_x,
               const __half* d_w,
               __half* d_y) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

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
        d_w,
        CUDA_R_16F,
        cfg.k,
        &beta,
        d_y,
        CUDA_R_16F,
        cfg.m,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

Metrics run_sequential(const Config& cfg,
                       cublasHandle_t handle,
                       cudaStream_t compute_stream,
                       const __half* h_weights,
                       size_t tile_elems,
                       size_t tile_bytes,
                       const __half* d_x,
                       __half* d_w,
                       __half* d_y) {
    auto copy_start = make_events(cfg.tiles);
    auto copy_end = make_events(cfg.tiles);
    auto compute_start = make_events(cfg.tiles);
    auto compute_end = make_events(cfg.tiles);

    CUBLAS_CHECK(cublasSetStream(handle, compute_stream));

    const auto wall_start = std::chrono::steady_clock::now();

    for (int i = 0; i < cfg.tiles; ++i) {
        const __half* src = h_weights + static_cast<size_t>(i) * tile_elems;

        CUDA_CHECK(cudaEventRecord(copy_start[static_cast<size_t>(i)], compute_stream));
        CUDA_CHECK(cudaMemcpyAsync(
            d_w, src, tile_bytes, cudaMemcpyHostToDevice, compute_stream));
        CUDA_CHECK(cudaEventRecord(copy_end[static_cast<size_t>(i)], compute_stream));

        CUDA_CHECK(cudaEventRecord(compute_start[static_cast<size_t>(i)], compute_stream));
        gemm_tile(handle, cfg, d_x, d_w, d_y);
        CUDA_CHECK(cudaEventRecord(compute_end[static_cast<size_t>(i)], compute_stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    const auto wall_end = std::chrono::steady_clock::now();

    Metrics result;
    result.wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    for (int i = 0; i < cfg.tiles; ++i) {
        result.copy_ms += event_ms(copy_start[static_cast<size_t>(i)], copy_end[static_cast<size_t>(i)]);
        result.compute_ms += event_ms(compute_start[static_cast<size_t>(i)], compute_end[static_cast<size_t>(i)]);
    }

    const double total_gb = (static_cast<double>(tile_bytes) * cfg.tiles) / 1.0e9;
    if (result.copy_ms > 0.0) {
        result.h2d_gbps = total_gb / (result.copy_ms / 1000.0);
    }

    destroy_events(copy_start);
    destroy_events(copy_end);
    destroy_events(compute_start);
    destroy_events(compute_end);
    return result;
}

Metrics run_overlapped(const Config& cfg,
                       cublasHandle_t handle,
                       cudaStream_t copy_stream,
                       cudaStream_t compute_stream,
                       const __half* h_weights,
                       size_t tile_elems,
                       size_t tile_bytes,
                       const __half* d_x,
                       __half* d_w0,
                       __half* d_w1,
                       __half* d_y) {
    auto copy_start = make_events(cfg.tiles);
    auto copy_end = make_events(cfg.tiles);
    auto compute_start = make_events(cfg.tiles);
    auto compute_end = make_events(cfg.tiles);

    __half* slots[2] = {d_w0, d_w1};
    CUBLAS_CHECK(cublasSetStream(handle, compute_stream));

    const auto wall_start = std::chrono::steady_clock::now();

    for (int i = 0; i < cfg.tiles; ++i) {
        const int slot = i & 1;
        const __half* src = h_weights + static_cast<size_t>(i) * tile_elems;

        // Before overwriting a VRAM slot, wait until the GEMM that last used
        // that slot has completed. This is the fixed two-slot ring contract.
        if (i >= 2) {
            CUDA_CHECK(cudaStreamWaitEvent(
                copy_stream, compute_end[static_cast<size_t>(i - 2)], 0));
        }

        CUDA_CHECK(cudaEventRecord(copy_start[static_cast<size_t>(i)], copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(
            slots[slot], src, tile_bytes, cudaMemcpyHostToDevice, copy_stream));
        CUDA_CHECK(cudaEventRecord(copy_end[static_cast<size_t>(i)], copy_stream));

        // Compute is allowed to start only when the tile assigned to this
        // execution-plan entry is resident in its fixed VRAM slot.
        CUDA_CHECK(cudaStreamWaitEvent(
            compute_stream, copy_end[static_cast<size_t>(i)], 0));
        CUDA_CHECK(cudaEventRecord(compute_start[static_cast<size_t>(i)], compute_stream));
        gemm_tile(handle, cfg, d_x, slots[slot], d_y);
        CUDA_CHECK(cudaEventRecord(compute_end[static_cast<size_t>(i)], compute_stream));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto wall_end = std::chrono::steady_clock::now();

    Metrics result;
    result.wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    double steady_copy_ms = 0.0;
    for (int i = 0; i < cfg.tiles; ++i) {
        const double this_copy = event_ms(
            copy_start[static_cast<size_t>(i)], copy_end[static_cast<size_t>(i)]);
        const double this_compute = event_ms(
            compute_start[static_cast<size_t>(i)], compute_end[static_cast<size_t>(i)]);

        result.copy_ms += this_copy;
        result.compute_ms += this_compute;
        if (i > 0) {
            steady_copy_ms += this_copy;
            result.starvation_ms += event_ms(
                compute_end[static_cast<size_t>(i - 1)],
                compute_start[static_cast<size_t>(i)]);
        }
    }

    result.startup_ms = event_ms(copy_start[0], compute_start[0]);

    const double total_gb = (static_cast<double>(tile_bytes) * cfg.tiles) / 1.0e9;
    if (result.copy_ms > 0.0) {
        result.h2d_gbps = total_gb / (result.copy_ms / 1000.0);
    }

    if (steady_copy_ms > 0.0) {
        const double hidden = 1.0 - (result.starvation_ms / steady_copy_ms);
        result.steady_hidden_transfer_pct = std::clamp(hidden * 100.0, 0.0, 100.0);
    }

    const double steady_active_ms = result.compute_ms + result.starvation_ms;
    if (steady_active_ms > 0.0) {
        result.steady_starvation_pct = 100.0 * result.starvation_ms / steady_active_ms;
    }

    destroy_events(copy_start);
    destroy_events(copy_end);
    destroy_events(compute_start);
    destroy_events(compute_end);
    return result;
}

void write_json(const std::string& path,
                const Config& cfg,
                const cudaDeviceProp& prop,
                size_t host_model_bytes,
                size_t tile_bytes,
                size_t fixed_vram_bytes,
                const RunResult& result) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open JSON output path: " + path);
    }

    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"experiment\": \"phase1-h2d-overlap\",\n";
    out << "  \"device\": {\n";
    out << "    \"name\": \"" << json_escape(prop.name) << "\",\n";
    out << "    \"compute_capability\": \"" << prop.major << '.' << prop.minor << "\",\n";
    out << "    \"total_vram_bytes\": " << static_cast<unsigned long long>(prop.totalGlobalMem) << ",\n";
    out << "    \"async_engine_count\": " << prop.asyncEngineCount << ",\n";
    out << "    \"device_overlap\": " << (prop.deviceOverlap ? "true" : "false") << "\n";
    out << "  },\n";
    out << "  \"config\": {\n";
    out << "    \"m\": " << cfg.m << ",\n";
    out << "    \"k\": " << cfg.k << ",\n";
    out << "    \"n\": " << cfg.n << ",\n";
    out << "    \"tiles\": " << cfg.tiles << ",\n";
    out << "    \"warmup\": " << cfg.warmup << ",\n";
    out << "    \"host_model_bytes\": " << static_cast<unsigned long long>(host_model_bytes) << ",\n";
    out << "    \"tile_bytes\": " << static_cast<unsigned long long>(tile_bytes) << ",\n";
    out << "    \"fixed_vram_bytes\": " << static_cast<unsigned long long>(fixed_vram_bytes) << "\n";
    out << "  },\n";

    auto emit_metrics = [&](const char* name, const Metrics& m, bool comma) {
        out << "  \"" << name << "\": {\n";
        out << "    \"wall_ms\": " << m.wall_ms << ",\n";
        out << "    \"copy_ms_sum\": " << m.copy_ms << ",\n";
        out << "    \"compute_ms_sum\": " << m.compute_ms << ",\n";
        out << "    \"starvation_ms_sum\": " << m.starvation_ms << ",\n";
        out << "    \"startup_ms\": " << m.startup_ms << ",\n";
        out << "    \"h2d_gbps\": " << m.h2d_gbps << ",\n";
        out << "    \"steady_hidden_transfer_pct\": " << m.steady_hidden_transfer_pct << ",\n";
        out << "    \"steady_starvation_pct\": " << m.steady_starvation_pct << "\n";
        out << "  }" << (comma ? "," : "") << "\n";
    };

    emit_metrics("sequential", result.sequential, true);
    emit_metrics("overlapped", result.overlapped, true);

    out << "  \"comparison\": {\n";
    out << "    \"speedup\": " << result.speedup << ",\n";
    out << "    \"max_abs_error\": " << result.max_abs_error << ",\n";
    out << "    \"rms_error\": " << result.rms_error << ",\n";
    out << "    \"correctness_ok\": " << (result.correctness_ok ? "true" : "false") << ",\n";
    out << "    \"verdict\": \"" << json_escape(result.verdict) << "\"\n";
    out << "  }\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    __half* h_weights = nullptr;
    __half* h_x = nullptr;
    __half* d_x = nullptr;
    __half* d_y = nullptr;
    __half* d_w0 = nullptr;
    __half* d_w1 = nullptr;
    cudaStream_t copy_stream = nullptr;
    cudaStream_t compute_stream = nullptr;
    cublasHandle_t handle = nullptr;

    try {
        const Config cfg = parse_args(argc, argv);

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (cfg.device >= device_count) {
            throw std::runtime_error("Requested CUDA device does not exist");
        }
        CUDA_CHECK(cudaSetDevice(cfg.device));

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));

        const size_t tile_elems = static_cast<size_t>(cfg.k) * static_cast<size_t>(cfg.n);
        const size_t tile_bytes = tile_elems * sizeof(__half);
        const size_t host_model_bytes = tile_bytes * static_cast<size_t>(cfg.tiles);
        const size_t x_elems = static_cast<size_t>(cfg.m) * static_cast<size_t>(cfg.k);
        const size_t y_elems = static_cast<size_t>(cfg.m) * static_cast<size_t>(cfg.n);
        const size_t x_bytes = x_elems * sizeof(__half);
        const size_t y_bytes = y_elems * sizeof(__half);
        const size_t fixed_vram_bytes = (2 * tile_bytes) + x_bytes + y_bytes;

        size_t free_vram = 0;
        size_t total_vram = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_vram, &total_vram));
        if (fixed_vram_bytes > static_cast<size_t>(static_cast<double>(free_vram) * 0.90)) {
            std::ostringstream oss;
            oss << "Experiment needs about "
                << (fixed_vram_bytes / (1024.0 * 1024.0))
                << " MiB VRAM but only "
                << (free_vram / (1024.0 * 1024.0))
                << " MiB is free. Reduce M/K/N or close GPU workloads.";
            throw std::runtime_error(oss.str());
        }

        std::cout << "=== TensorWave Phase-1: fixed-VRAM streaming proof ===\n";
        std::cout << "GPU: " << prop.name << " | CC " << prop.major << '.' << prop.minor
                  << " | async engines=" << prop.asyncEngineCount
                  << " | overlap=" << (prop.deviceOverlap ? "yes" : "no") << "\n";
        std::cout << "GEMM shape: M=" << cfg.m << " K=" << cfg.k << " N=" << cfg.n
                  << " | tiles=" << cfg.tiles << "\n";
        std::cout << "Host model: " << std::fixed << std::setprecision(1)
                  << (host_model_bytes / (1024.0 * 1024.0)) << " MiB pinned RAM\n";
        std::cout << "One weight tile: " << (tile_bytes / (1024.0 * 1024.0)) << " MiB\n";
        std::cout << "Fixed VRAM working set: "
                  << (fixed_vram_bytes / (1024.0 * 1024.0))
                  << " MiB (2 weight slots + X + Y)\n\n";

        // The host-resident model is deliberately pinned in Phase 1 so the
        // experiment isolates the PCIe/DMA overlap question. Later phases can
        // introduce a smaller pinned staging window in front of normal RAM.
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_weights), host_model_bytes));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_x), x_bytes));

        std::vector<__half> tile_template(tile_elems);
        for (size_t i = 0; i < tile_elems; ++i) {
            const int centered = static_cast<int>(i % 257) - 128;
            tile_template[i] = __float2half(static_cast<float>(centered) / 4096.0f);
        }
        for (int t = 0; t < cfg.tiles; ++t) {
            __half* dst = h_weights + static_cast<size_t>(t) * tile_elems;
            std::memcpy(dst, tile_template.data(), tile_bytes);
            dst[0] = __float2half(static_cast<float>((t % 31) - 15) / 512.0f);
        }
        for (size_t i = 0; i < x_elems; ++i) {
            const int centered = static_cast<int>(i % 127) - 63;
            h_x[i] = __float2half(static_cast<float>(centered) / 1024.0f);
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_x), x_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_y), y_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_w0), tile_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_w1), tile_bytes));

        CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
        CUBLAS_CHECK(cublasSetStream(handle, compute_stream));

        CUDA_CHECK(cudaMemcpyAsync(d_x, h_x, x_bytes, cudaMemcpyHostToDevice, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        std::cout << "Warmup: " << cfg.warmup << " iterations...\n";
        for (int i = 0; i < cfg.warmup; ++i) {
            const int tile = i % cfg.tiles;
            const __half* src = h_weights + static_cast<size_t>(tile) * tile_elems;
            CUDA_CHECK(cudaMemcpyAsync(
                d_w0, src, tile_bytes, cudaMemcpyHostToDevice, compute_stream));
            gemm_tile(handle, cfg, d_x, d_w0, d_y);
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        std::cout << "Running sequential baseline...\n";
        const Metrics sequential = run_sequential(
            cfg, handle, compute_stream, h_weights, tile_elems, tile_bytes,
            d_x, d_w0, d_y);

        const size_t sample_elems = std::min<size_t>(y_elems, 4096);
        std::vector<__half> sequential_sample(sample_elems);
        CUDA_CHECK(cudaMemcpy(
            sequential_sample.data(), d_y, sample_elems * sizeof(__half), cudaMemcpyDeviceToHost));

        std::cout << "Running overlapped two-slot pipeline...\n";
        const Metrics overlapped = run_overlapped(
            cfg, handle, copy_stream, compute_stream, h_weights, tile_elems, tile_bytes,
            d_x, d_w0, d_w1, d_y);

        std::vector<__half> overlapped_sample(sample_elems);
        CUDA_CHECK(cudaMemcpy(
            overlapped_sample.data(), d_y, sample_elems * sizeof(__half), cudaMemcpyDeviceToHost));

        double max_abs_error = 0.0;
        double sum_sq_error = 0.0;
        for (size_t i = 0; i < sample_elems; ++i) {
            const double a = static_cast<double>(__half2float(sequential_sample[i]));
            const double b = static_cast<double>(__half2float(overlapped_sample[i]));
            const double err = std::abs(a - b);
            max_abs_error = std::max(max_abs_error, err);
            sum_sq_error += err * err;
        }
        const double rms_error = sample_elems > 0
            ? std::sqrt(sum_sq_error / static_cast<double>(sample_elems))
            : 0.0;
        const bool correctness_ok = max_abs_error <= 1.0e-2;

        RunResult result;
        result.sequential = sequential;
        result.overlapped = overlapped;
        result.speedup = overlapped.wall_ms > 0.0
            ? sequential.wall_ms / overlapped.wall_ms
            : 0.0;
        result.max_abs_error = max_abs_error;
        result.rms_error = rms_error;
        result.correctness_ok = correctness_ok;

        if (!correctness_ok) {
            result.verdict = "FAIL_CORRECTNESS";
        } else if (overlapped.steady_starvation_pct <= 10.0 &&
                   overlapped.steady_hidden_transfer_pct >= 80.0) {
            result.verdict = "HYPOTHESIS_SUPPORTED_FOR_THIS_SHAPE";
        } else if (result.speedup >= 1.05 || overlapped.steady_hidden_transfer_pct >= 30.0) {
            result.verdict = "PARTIAL_OVERLAP_FOR_THIS_SHAPE";
        } else {
            result.verdict = "TRANSFER_NOT_HIDDEN_FOR_THIS_SHAPE";
        }

        std::cout << std::setprecision(3);
        std::cout << "\n--- Results ---\n";
        std::cout << "Sequential wall:       " << sequential.wall_ms << " ms\n";
        std::cout << "Sequential H2D:        " << sequential.h2d_gbps << " GB/s\n";
        std::cout << "Overlapped wall:       " << overlapped.wall_ms << " ms\n";
        std::cout << "Overlapped H2D:        " << overlapped.h2d_gbps << " GB/s\n";
        std::cout << "Compute sum:           " << overlapped.compute_ms << " ms\n";
        std::cout << "GPU starvation sum:    " << overlapped.starvation_ms << " ms\n";
        std::cout << "Steady starvation:     " << overlapped.steady_starvation_pct << " %\n";
        std::cout << "Transfer hidden est.:  " << overlapped.steady_hidden_transfer_pct << " %\n";
        std::cout << "Pipeline speedup:      " << result.speedup << "x\n";
        std::cout << "Max abs error:         " << result.max_abs_error << "\n";
        std::cout << "RMS error:             " << result.rms_error << "\n";
        std::cout << "Verdict:               " << result.verdict << "\n";

        if (!cfg.json_path.empty()) {
            write_json(cfg.json_path, cfg, prop, host_model_bytes, tile_bytes,
                       fixed_vram_bytes, result);
            std::cout << "JSON:                  " << cfg.json_path << "\n";
        }

        CUBLAS_CHECK(cublasDestroy(handle));
        handle = nullptr;
        CUDA_CHECK(cudaStreamDestroy(copy_stream));
        copy_stream = nullptr;
        CUDA_CHECK(cudaStreamDestroy(compute_stream));
        compute_stream = nullptr;
        CUDA_CHECK(cudaFree(d_w1)); d_w1 = nullptr;
        CUDA_CHECK(cudaFree(d_w0)); d_w0 = nullptr;
        CUDA_CHECK(cudaFree(d_y)); d_y = nullptr;
        CUDA_CHECK(cudaFree(d_x)); d_x = nullptr;
        CUDA_CHECK(cudaFreeHost(h_x)); h_x = nullptr;
        CUDA_CHECK(cudaFreeHost(h_weights)); h_weights = nullptr;

        return correctness_ok ? 0 : 2;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << '\n';

        if (handle) cublasDestroy(handle);
        if (copy_stream) cudaStreamDestroy(copy_stream);
        if (compute_stream) cudaStreamDestroy(compute_stream);
        if (d_w1) cudaFree(d_w1);
        if (d_w0) cudaFree(d_w0);
        if (d_y) cudaFree(d_y);
        if (d_x) cudaFree(d_x);
        if (h_x) cudaFreeHost(h_x);
        if (h_weights) cudaFreeHost(h_weights);
        return 1;
    }
}
