static Args parse_args(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](const char * what) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + what);
            return argv[++i];
        };
        if (k == "--q4") a.q4_path = need("--q4");
        else if (k == "--x") a.x_path = need("--x");
        else if (k == "--ref") a.fp32_ref_path = need("--ref");
        else if (k == "--rows") a.rows = std::stoi(need("--rows"));
        else if (k == "--cols") a.cols = std::stoi(need("--cols"));
        else if (k == "--iterations") a.iterations = std::stoi(need("--iterations"));
        else if (k == "--warmup") a.warmup = std::stoi(need("--warmup"));
        else if (k == "--cpu") a.cpu = std::stoi(need("--cpu"));
        else if (k == "--help" || k == "-h") { usage(argv[0]); std::exit(0); }
        else throw std::runtime_error("unknown argument: " + k);
    }

    if (a.q4_path.empty() || a.x_path.empty() || a.rows <= 0 || a.cols <= 0) {
        usage(argv[0]);
        throw std::runtime_error("required arguments are missing");
    }
    if (a.cols % QK_K != 0) throw std::runtime_error("cols must be divisible by 256");
    if (a.iterations <= 0 || a.warmup < 0) throw std::runtime_error("invalid iteration counts");
    return a;
}

static void pin_current_thread(int cpu) {
    if (cpu < 0) return;
#ifdef _WIN32
    if (cpu >= static_cast<int>(8 * sizeof(DWORD_PTR))) {
        throw std::runtime_error("--cpu is outside current Windows processor group mask width");
    }
    const DWORD_PTR mask = (static_cast<DWORD_PTR>(1) << cpu);
    if (SetThreadAffinityMask(GetCurrentThread(), mask) == 0) {
        throw std::runtime_error("SetThreadAffinityMask failed");
    }
#else
    std::cerr << "warning: --cpu affinity is implemented only for Windows in this benchmark\n";
#endif
}

static bool runtime_avx2() {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#else
    return true;
#endif
}

static bool runtime_fma() {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("fma");
#else
    return true;
#endif
}

} // namespace transit

int main(int argc, char ** argv) {
    using namespace transit;

    try {
        const Args args = parse_args(argc, argv);
        pin_current_thread(args.cpu);

        if (!runtime_avx2() || !runtime_fma()) {
            throw std::runtime_error("this executable requires AVX2 + FMA at runtime");
        }

        const auto raw = read_bytes(args.q4_path);
        const auto x = read_floats(args.x_path);
        if (static_cast<int>(x.size()) != args.cols) {
            throw std::runtime_error("activation length does not match --cols");
        }
        const int nb = args.cols / QK_K;
        const size_t expected_q4 = static_cast<size_t>(args.rows) * nb * sizeof(Q4KRaw);
        if (raw.size() != expected_q4) {
            std::ostringstream oss;
            oss << "Q4_K size mismatch: got " << raw.size() << ", expected " << expected_q4;
            throw std::runtime_error(oss.str());
        }

        std::cout << "\n======================================================================\n";
        std::cout << " Transit native Q4_K x Q8_K benchmark\n";
        std::cout << "======================================================================\n\n";
        std::cout << "Rows                       : " << args.rows << "\n";
        std::cout << "Columns                    : " << args.cols << "\n";
        std::cout << "Logical weights/GEMV       : " << static_cast<uint64_t>(args.rows) * args.cols << "\n";
        std::cout << "Q4_K blocks                : " << static_cast<uint64_t>(args.rows) * nb << "\n";
        std::cout << "Q4_K bytes                 : " << raw.size() << " (" << std::fixed << std::setprecision(3)
                  << raw.size() / 1048576.0 << " MiB)\n";
        std::cout << "Q8_K activation bytes      : " << static_cast<size_t>(nb) * sizeof(Q8K)
                  << " (" << static_cast<size_t>(nb) * sizeof(Q8K) / 1024.0 << " KiB)\n";
        std::cout << "Compiled __AVX2__          : "
#ifdef __AVX2__
                  << "yes\n";
#else
                  << "no\n";
#endif
        std::cout << "Compiled __FMA__           : "
#ifdef __FMA__
                  << "yes\n";
#else
                  << "no\n";
#endif
        std::cout << "Compiled __AVXVNNI__       : "
#ifdef __AVXVNNI__
                  << "yes\n";
#else
                  << "no\n";
#endif
        std::cout << "Runtime AVX2/FMA           : yes/yes\n";
        std::cout << "Pinned logical CPU         : " << args.cpu << " (-1 means scheduler)\n";
        std::cout << "Warmup / timed iterations  : " << args.warmup << " / " << args.iterations << "\n";

        const auto t_q8_0 = std::chrono::steady_clock::now();
        auto q8 = build_q8(x);
        const auto t_q8_1 = std::chrono::steady_clock::now();
        const double q8_once_ms = std::chrono::duration<double, std::milli>(t_q8_1 - t_q8_0).count();

        const auto t_rep_0 = std::chrono::steady_clock::now();
        auto rep = repack_q4(raw, args.rows, args.cols);
        const auto t_rep_1 = std::chrono::steady_clock::now();
        const double repack_once_ms = std::chrono::duration<double, std::milli>(t_rep_1 - t_rep_0).count();

        std::vector<int16_t> lut(static_cast<size_t>(args.cols) * 16);
        const auto t_lut_0 = std::chrono::steady_clock::now();
        build_nibble_lut_q8(q8.data(), args.cols, lut.data());
        const auto t_lut_1 = std::chrono::steady_clock::now();
        const double lut_once_ms = std::chrono::duration<double, std::milli>(t_lut_1 - t_lut_0).count();

        std::cout << "\nOne-time / per-activation preparation\n";
        std::cout << "  Q8_K quantize once        : " << std::fixed << std::setprecision(3) << q8_once_ms << " ms\n";
        std::cout << "  model repack once         : " << repack_once_ms << " ms\n";
        std::cout << "  repacked Q4_K bytes       : " << rep.size() * sizeof(Q4KRepacked)
                  << " (" << rep.size() * sizeof(Q4KRepacked) / 1048576.0 << " MiB, "
                  << (100.0 * (sizeof(Q4KRepacked) - sizeof(Q4KRaw)) / sizeof(Q4KRaw)) << "% block overhead)\n";
        std::cout << "  Q8 nibble LUT build once  : " << lut_once_ms << " ms\n";
        std::cout << "  Q8 nibble LUT bytes       : " << lut.size() * sizeof(int16_t)
                  << " (" << lut.size() * sizeof(int16_t) / 1048576.0 << " MiB)\n";

        std::vector<float> ref(args.rows), tmp(args.rows);
        gemv_scalar_raw(raw.data(), q8.data(), ref.data(), args.rows, args.cols);

        std::cout << "\nCorrectness vs scalar Q4_K x Q8_K reference\n";
        gemv_llama_style_avx2_raw_1row(raw.data(), q8.data(), tmp.data(), args.rows, args.cols);
        print_metrics("llama-style AVX2 baseline", compare_vectors(ref, tmp));
        gemv_avx2_raw_1row(raw.data(), q8.data(), tmp.data(), args.rows, args.cols);
        print_metrics("Transit AVX2 raw 1-row", compare_vectors(ref, tmp));
        gemv_avx2_raw_4row(raw.data(), q8.data(), tmp.data(), args.rows, args.cols);
        print_metrics("AVX2 raw 4-row", compare_vectors(ref, tmp));
        gemv_avx2_raw_8row(raw.data(), q8.data(), tmp.data(), args.rows, args.cols);
        print_metrics("AVX2 raw 8-row", compare_vectors(ref, tmp));
        gemv_avx2_repacked_1row(rep.data(), q8.data(), tmp.data(), args.rows, args.cols);
        print_metrics("AVX2 repacked 1-row", compare_vectors(ref, tmp));
        gemv_avx2_rep_4row(rep.data(), q8.data(), tmp.data(), args.rows, args.cols);
        print_metrics("AVX2 repacked 4-row", compare_vectors(ref, tmp));
        gemv_avx2_rep_8row(rep.data(), q8.data(), tmp.data(), args.rows, args.cols);
        print_metrics("AVX2 repacked 8-row", compare_vectors(ref, tmp));
#if defined(__AVXVNNI__)
        gemv_avxvnni_raw_1row(raw.data(), q8.data(), tmp.data(), args.rows, args.cols);
        print_metrics("AVX-VNNI raw 1-row", compare_vectors(ref, tmp));
        gemv_avxvnni_repacked_1row(rep.data(), q8.data(), tmp.data(), args.rows, args.cols);
        print_metrics("AVX-VNNI repacked 1-row", compare_vectors(ref, tmp));
        gemv_avxvnni_raw_4row(raw.data(), q8.data(), tmp.data(), args.rows, args.cols);
        print_metrics("AVX-VNNI raw 4-row", compare_vectors(ref, tmp));
        gemv_avxvnni_raw_8row(raw.data(), q8.data(), tmp.data(), args.rows, args.cols);
        print_metrics("AVX-VNNI raw 8-row", compare_vectors(ref, tmp));
        gemv_avxvnni_rep_4row(rep.data(), q8.data(), tmp.data(), args.rows, args.cols);
        print_metrics("AVX-VNNI repacked 4-row", compare_vectors(ref, tmp));
        gemv_avxvnni_rep_8row(rep.data(), q8.data(), tmp.data(), args.rows, args.cols);
        print_metrics("AVX-VNNI repacked 8-row", compare_vectors(ref, tmp));
#endif
        gemv_lut_raw(raw.data(), q8.data(), lut.data(), tmp.data(), args.rows, args.cols);
        print_metrics("Nibble LUT raw exact", compare_vectors(ref, tmp));
        gemv_lut_repacked(rep.data(), q8.data(), lut.data(), tmp.data(), args.rows, args.cols);
        print_metrics("Nibble LUT repacked exact", compare_vectors(ref, tmp));

        if (!args.fp32_ref_path.empty()) {
            const auto fp32ref = read_floats(args.fp32_ref_path);
            if (static_cast<int>(fp32ref.size()) != args.rows) {
                throw std::runtime_error("--ref length does not match rows");
            }
            std::cout << "\nQ8_K activation quantization effect vs prior FP32-activation Python reference\n";
            print_metrics("scalar Q4_K x Q8_K", compare_vectors(fp32ref, ref));
        }

        std::cout << "\nKernel-only benchmark (Q8 and model repack already prepared)\n";

        // Scalar is reported but AVX2 raw 1-row is the optimization baseline.
        const auto b_scalar = benchmark(args.warmup, args.iterations, args.rows, [&] {
            gemv_scalar_raw(raw.data(), q8.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[0 % args.rows] * 1e-300;
        });
        const auto b_llama = benchmark(args.warmup, args.iterations, args.rows, [&] {
            gemv_llama_style_avx2_raw_1row(raw.data(), q8.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[1 % args.rows] * 1e-300;
        });
        const auto b_avx1 = benchmark(args.warmup, args.iterations, args.rows, [&] {
            gemv_avx2_raw_1row(raw.data(), q8.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[1 % args.rows] * 1e-300;
        });
        const double baseline = b_llama.median_ms;

        print_bench("scalar reference", b_scalar, baseline);
        print_bench("llama-style AVX2 BASELINE", b_llama, baseline);
        print_bench("Transit AVX2 raw 1-row", b_avx1, baseline);

        const auto b_avx4 = benchmark(args.warmup, args.iterations, args.rows, [&] {
            gemv_avx2_raw_4row(raw.data(), q8.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[2 % args.rows] * 1e-300;
        });
        const auto b_avx8 = benchmark(args.warmup, args.iterations, args.rows, [&] {
            gemv_avx2_raw_8row(raw.data(), q8.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[3 % args.rows] * 1e-300;
        });
        const auto b_rep1 = benchmark(args.warmup, args.iterations, args.rows, [&] {
            gemv_avx2_repacked_1row(rep.data(), q8.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[4 % args.rows] * 1e-300;
        });
        const auto b_rep4 = benchmark(args.warmup, args.iterations, args.rows, [&] {
            gemv_avx2_rep_4row(rep.data(), q8.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[5 % args.rows] * 1e-300;
        });
        const auto b_rep8 = benchmark(args.warmup, args.iterations, args.rows, [&] {
            gemv_avx2_rep_8row(rep.data(), q8.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[6 % args.rows] * 1e-300;
        });
        const auto b_lut_raw = benchmark(args.warmup, args.iterations, args.rows, [&] {
            gemv_lut_raw(raw.data(), q8.data(), lut.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[7 % args.rows] * 1e-300;
        });
        const auto b_lut_rep = benchmark(args.warmup, args.iterations, args.rows, [&] {
            gemv_lut_repacked(rep.data(), q8.data(), lut.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[8 % args.rows] * 1e-300;
        });

        print_bench("AVX2 raw 4-row", b_avx4, baseline);
        print_bench("AVX2 raw 8-row", b_avx8, baseline);
        print_bench("AVX2 repacked 1-row", b_rep1, baseline);
        print_bench("AVX2 repacked 4-row", b_rep4, baseline);
        print_bench("AVX2 repacked 8-row", b_rep8, baseline);
#if defined(__AVXVNNI__)
        const auto b_vnni1 = benchmark(args.warmup, args.iterations, args.rows, [&] {
            gemv_avxvnni_raw_1row(raw.data(), q8.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[12 % args.rows] * 1e-300;
        });
        const auto b_vnnir1 = benchmark(args.warmup, args.iterations, args.rows, [&] {
            gemv_avxvnni_repacked_1row(rep.data(), q8.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[13 % args.rows] * 1e-300;
        });
        const auto b_vnni4 = benchmark(args.warmup, args.iterations, args.rows, [&] {
            gemv_avxvnni_raw_4row(raw.data(), q8.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[14 % args.rows] * 1e-300;
        });
        const auto b_vnni8 = benchmark(args.warmup, args.iterations, args.rows, [&] {
            gemv_avxvnni_raw_8row(raw.data(), q8.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[15 % args.rows] * 1e-300;
        });
        const auto b_vnnir4 = benchmark(args.warmup, args.iterations, args.rows, [&] {
            gemv_avxvnni_rep_4row(rep.data(), q8.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[16 % args.rows] * 1e-300;
        });
        const auto b_vnnir8 = benchmark(args.warmup, args.iterations, args.rows, [&] {
            gemv_avxvnni_rep_8row(rep.data(), q8.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[17 % args.rows] * 1e-300;
        });
        print_bench("AVX-VNNI raw 1-row", b_vnni1, baseline);
        print_bench("AVX-VNNI repacked 1-row", b_vnnir1, baseline);
        print_bench("AVX-VNNI raw 4-row", b_vnni4, baseline);
        print_bench("AVX-VNNI raw 8-row", b_vnni8, baseline);
        print_bench("AVX-VNNI repacked 4-row", b_vnnir4, baseline);
        print_bench("AVX-VNNI repacked 8-row", b_vnnir8, baseline);
#endif
        print_bench("Nibble LUT raw compute", b_lut_raw, baseline);
        print_bench("Nibble LUT repacked compute", b_lut_rep, baseline);

        // Measure dynamic costs in the exact place they belong.
        std::vector<Q8K> q8_work(q8.size());
        std::vector<int16_t> lut_work(lut.size());
        const auto b_q8 = benchmark(args.warmup, args.iterations, args.rows, [&] {
            quantize_q8k_ref(x.data(), q8_work.data(), args.cols);
            g_sink += q8_work[0].d * 1e-300;
        });
        const auto b_lut_build = benchmark(args.warmup, args.iterations, args.rows, [&] {
            build_nibble_lut_q8(q8.data(), args.cols, lut_work.data());
            g_sink += lut_work[0] * 1e-300;
        });

        std::cout << "\nDynamic preparation benchmark\n";
        print_bench("Q8_K quantize", b_q8, baseline);
        print_bench("Nibble LUT build", b_lut_build, baseline);

        const auto b_e2e_direct = benchmark(args.warmup, args.iterations, args.rows, [&] {
            quantize_q8k_ref(x.data(), q8_work.data(), args.cols);
            gemv_llama_style_avx2_raw_1row(raw.data(), q8_work.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[9 % args.rows] * 1e-300;
        });
        const auto b_e2e_bestrep4 = benchmark(args.warmup, args.iterations, args.rows, [&] {
            quantize_q8k_ref(x.data(), q8_work.data(), args.cols);
            gemv_avx2_rep_4row(rep.data(), q8_work.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[10 % args.rows] * 1e-300;
        });
        const auto b_e2e_lut = benchmark(args.warmup, args.iterations, args.rows, [&] {
            quantize_q8k_ref(x.data(), q8_work.data(), args.cols);
            build_nibble_lut_q8(q8_work.data(), args.cols, lut_work.data());
            gemv_lut_repacked(rep.data(), q8_work.data(), lut_work.data(), tmp.data(), args.rows, args.cols);
            g_sink += tmp[11 % args.rows] * 1e-300;
        });

        std::cout << "\nActivation-to-output benchmark (dynamic Q8/LUT costs included)\n";
        print_bench("Q8 + llama-style AVX2", b_e2e_direct, b_e2e_direct.median_ms);
        print_bench("Q8 + AVX2 repacked 4-row", b_e2e_bestrep4, b_e2e_direct.median_ms);
        print_bench("Q8 + LUT build + repacked LUT", b_e2e_lut, b_e2e_direct.median_ms);

        std::cout << "\nInterpretation guardrails\n";
        std::cout << "  * llama-style AVX2 ports the current ggml Q4_K x Q8_K arithmetic shape, but not full ggml runtime overhead.\n";
        std::cout << "  * Repack is model-load-time and preserves Q4 nibbles + FP16 d/dmin exactly.\n";
        std::cout << "  * Nibble LUT is exact relative to this Q4_K x Q8_K arithmetic; it changes no weights.\n";
        std::cout << "  * The decisive comparison is against AVX2, not against the older managed C# direct loop.\n";
        std::cout << "  * End-to-end model tok/s is not measured here.\n";
        std::cout << "\nSink: " << std::setprecision(17) << g_sink << "\n";
        std::cout << "======================================================================\n";

        return 0;
    } catch (const std::exception & e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
