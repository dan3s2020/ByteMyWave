#!/usr/bin/env python3
"""TensorWave Phase-6 heterogeneous MoE feasibility calculator.

Analytical requirements/ceilings only; this is not a hardware benchmark.
It separates:
- host->GPU streaming,
- deterministic non-routed VRAM residency,
- CPU NUMA routed-expert requirements,
- ideal multi-GPU independent-link sharding,
- compression-only limits,
- full-residency VRAM-bandwidth limits,
- CPU<->GPU per-layer handoff sensitivity.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

GB = 1e9
GIB = 1024.0 ** 3


@dataclass(frozen=True)
class QuantFormat:
    name: str
    bytes_per_weight: float
    note: str


FORMATS = {
    "tw_q4_g32_f32s": QuantFormat(
        "tw_q4_g32_f32s", 20.0 / 32.0,
        "Current TensorWave Phase-3: 32 int4 weights + FP32 scale.",
    ),
    "q3_g64_f16s": QuantFormat(
        "q3_g64_f16s", 26.0 / 64.0,
        "Candidate: 64x3-bit + FP16 scale.",
    ),
    "q2_g32_f32s": QuantFormat(
        "q2_g32_f32s", 12.0 / 32.0,
        "Candidate: 32x2-bit + FP32 scale.",
    ),
    "q2_g64_f16s": QuantFormat(
        "q2_g64_f16s", 18.0 / 64.0,
        "Candidate: 64x2-bit + FP16 scale.",
    ),
    "ideal_1bit": QuantFormat(
        "ideal_1bit", 1.0 / 8.0,
        "Stress test only; not a claimed quality-preserving PTQ format.",
    ),
}


@dataclass(frozen=True)
class Model:
    name: str
    total_b: float
    active_b: float
    layers: int
    dense_layers: int
    hidden: int
    moe_intermediate: int
    experts: int
    selected: int
    shared: int

    @property
    def moe_layers(self) -> int:
        return self.layers - self.dense_layers

    @property
    def params_per_routed_expert(self) -> int:
        # SwiGLU gate + up + down.
        return 3 * self.hidden * self.moe_intermediate

    @property
    def routed_active_b(self) -> float:
        n = self.moe_layers * self.selected * self.params_per_routed_expert
        return n / 1e9

    @property
    def non_routed_active_b(self) -> float:
        # Planning remainder: official active count is rounded.
        return max(0.0, self.active_b - self.routed_active_b)


MODELS = {
    "k2.5": Model(
        "Kimi K2.5", 1000.0, 32.0, 61, 1,
        7168, 2048, 384, 8, 1,
    ),
    "k3": Model(
        "Kimi K3", 2800.0, 104.0, 93, 1,
        7168, 3072, 896, 16, 2,
    ),
}


@dataclass(frozen=True)
class Hardware:
    host_ram_gib: float = 1024.0
    cpu_sockets: int = 4
    cores_per_socket: int = 15
    cpu_base_ghz: float = 2.8
    cpu_mem_bw_gbps_published: float = 85.0
    cpu_instruction_set: str = "Intel AVX (E7-4890 v2; no AVX2/FMA)"
    gpu_vram_gib: float = 12.0
    reserve_vram_gib: float = 4.0
    h2d_gbps_per_gpu: float = 12.0
    effective_gpu_tflops: float = 10.0
    gpu_vram_bw_gbps: float = 360.0


def model_storage_gb(model: Model, q: QuantFormat) -> float:
    return model.total_b * q.bytes_per_weight


def resident_weight_budget_gb(hw: Hardware) -> float:
    return max(0.0, (hw.gpu_vram_gib - hw.reserve_vram_gib) * GIB / GB)


def stream_case(model: Model, q: QuantFormat, hw: Hardware,
                static_non_routed: bool) -> dict:
    if static_non_routed:
        resident_gb = model.non_routed_active_b * q.bytes_per_weight
        budget = resident_weight_budget_gb(hw)
        if resident_gb > budget:
            return {
                "strategy": "static-residency",
                "format": q.name,
                "error": f"resident set {resident_gb:.3f} GB > weight budget {budget:.3f} GB",
            }
        streamed_b = model.routed_active_b
        strategy = "static-residency"
    else:
        streamed_b = model.active_b
        strategy = "stream-all"

    stream_gb = streamed_b * q.bytes_per_weight
    transfer_s = stream_gb / hw.h2d_gbps_per_gpu
    compute_s = (2.0 * model.active_b * 1e9) / (hw.effective_gpu_tflops * 1e12)
    step_s = max(transfer_s, compute_s)
    return {
        "strategy": strategy,
        "format": q.name,
        "bytes_per_weight": q.bytes_per_weight,
        "streamed_params_b": streamed_b,
        "stream_gb_per_token": stream_gb,
        "transfer_ms": transfer_s * 1000.0,
        "gpu_compute_ms": compute_s * 1000.0,
        "bandwidth_ceiling_tps": 1.0 / transfer_s,
        "compute_ceiling_tps": 1.0 / compute_s,
        "combined_roofline_tps": 1.0 / step_s,
    }


def cpu_expert_requirement(model: Model, q: QuantFormat, hw: Hardware,
                           target_tps: float) -> dict:
    routed_gb = model.routed_active_b * q.bytes_per_weight
    per_socket_bw = routed_gb * target_tps / hw.cpu_sockets
    per_socket_gflops = 2.0 * model.routed_active_b * target_tps / hw.cpu_sockets
    per_socket_gweights = model.routed_active_b * target_tps / hw.cpu_sockets

    aggregate_cycles_s = hw.cores_per_socket * hw.cpu_base_ghz * 1e9
    cycles_per_weight = aggregate_cycles_s / (per_socket_gweights * 1e9)

    # Conservative volume bound: one BF16 hidden vector sent to every socket
    # and one BF16 partial vector returned from every socket for every MoE layer.
    hidden_bytes = model.hidden * 2
    activation_bytes_token = 2 * hw.cpu_sockets * hidden_bytes * model.moe_layers

    return {
        "target_tps": target_tps,
        "sockets": hw.cpu_sockets,
        "format": q.name,
        "routed_active_params_b": model.routed_active_b,
        "routed_gb_per_token": routed_gb,
        "required_mem_bw_gbps_per_socket": per_socket_bw,
        "fraction_of_published_85gbps": per_socket_bw / hw.cpu_mem_bw_gbps_published,
        "required_gflops_per_socket": per_socket_gflops,
        "required_gweights_per_s_per_socket": per_socket_gweights,
        "aggregate_core_cycles_per_weight_budget": cycles_per_weight,
        "activation_roundtrip_mb_per_token_upper_bound": activation_bytes_token / 1e6,
        "activation_pcie_gbps_at_target": activation_bytes_token * target_tps / 1e9,
    }


def compression_only_required_bits(model: Model, hw: Hardware,
                                   target_tps: float) -> float:
    max_gb_token = hw.h2d_gbps_per_gpu / target_tps
    return max_gb_token * 8.0 / model.active_b


def ideal_multi_gpu_routed(model: Model, q: QuantFormat, hw: Hardware,
                           gpus: int, accepted_tokens_per_pass: float) -> dict:
    routed_gb = model.routed_active_b * q.bytes_per_weight
    aggregate_h2d = gpus * hw.h2d_gbps_per_gpu
    passes_s = aggregate_h2d / routed_gb
    return {
        "gpus": gpus,
        "format": q.name,
        "routed_gb_per_pass": routed_gb,
        "aggregate_h2d_gbps": aggregate_h2d,
        "accepted_tokens_per_pass": accepted_tokens_per_pass,
        "ideal_target_passes_per_s": passes_s,
        "ideal_output_tps": passes_s * accepted_tokens_per_pass,
        "warning": "Ideal independent-link routed-shard ceiling only; excludes attention, reductions, routing, handoff and synchronization.",
    }


def hypothetical_full_vram(model: Model, q: QuantFormat, hw: Hardware) -> dict:
    active_gb = model.active_b * q.bytes_per_weight
    vram_bw_tps = hw.gpu_vram_bw_gbps / active_gb
    compute_tps = hw.effective_gpu_tflops * 1e12 / (2.0 * model.active_b * 1e9)
    return {
        "format": q.name,
        "active_gb_read_per_token": active_gb,
        "vram_bandwidth_ceiling_tps": vram_bw_tps,
        "compute_only_ceiling_tps": compute_tps,
        "combined_roofline_tps": min(vram_bw_tps, compute_tps),
        "note": "Hypothetical: full model resident, same GPU bandwidth/compute profile.",
    }


def handoff_sensitivity(model: Model) -> list[dict]:
    rows = []
    for us in (10, 50, 100, 250, 500, 1000):
        ms = model.moe_layers * us / 1000.0
        rows.append({
            "roundtrip_us_per_moe_layer": us,
            "serial_handoff_ms_per_token": ms,
            "handoff_only_ceiling_tps": 1000.0 / ms,
        })
    return rows


def build(model: Model, hw: Hardware) -> dict:
    stream = []
    for q in FORMATS.values():
        stream.append(stream_case(model, q, hw, False))
        stream.append(stream_case(model, q, hw, True))

    cpu = []
    for sockets in (2, 4):
        h = Hardware(**{**asdict(hw), "cpu_sockets": sockets})
        for fmt in ("tw_q4_g32_f32s", "q2_g64_f16s"):
            for target in (5.0, 10.0):
                cpu.append(cpu_expert_requirement(model, FORMATS[fmt], h, target))

    multi = []
    for fmt in ("tw_q4_g32_f32s", "q3_g64_f16s", "q2_g64_f16s"):
        for gpus in (1, 2, 3, 4, 6):
            for accepted in (1.0, 2.61, 4.73):
                multi.append(ideal_multi_gpu_routed(
                    model, FORMATS[fmt], hw, gpus, accepted
                ))

    return {
        "schema": "tensorwave.heterogeneous-moe-feasibility.v1",
        "model": {
            **asdict(model),
            "moe_layers": model.moe_layers,
            "params_per_routed_expert": model.params_per_routed_expert,
            "routed_active_b": model.routed_active_b,
            "derived_non_routed_active_b": model.non_routed_active_b,
            "note": "Non-routed active is derived from a rounded official active count; replace with checkpoint census before implementation.",
        },
        "hardware": asdict(hw),
        "resident_weight_budget_gb": resident_weight_budget_gb(hw),
        "storage": {
            name: {
                "bytes_per_weight": q.bytes_per_weight,
                "model_gb": model_storage_gb(model, q),
                "fits_host_ram_capacity_only": model_storage_gb(model, q) * GB <= hw.host_ram_gib * GIB,
            }
            for name, q in FORMATS.items()
        },
        "streaming": stream,
        "cpu_expert_requirements": cpu,
        "compression_only_required_bits_per_active_weight": {
            "5_tps": compression_only_required_bits(model, hw, 5.0),
            "10_tps": compression_only_required_bits(model, hw, 10.0),
        },
        "ideal_multi_gpu_routed": multi,
        "hypothetical_full_vram": {
            "tw_q4": hypothetical_full_vram(model, FORMATS["tw_q4_g32_f32s"], hw),
            "ideal_1bit": hypothetical_full_vram(model, FORMATS["ideal_1bit"], hw),
        },
        "cpu_gpu_handoff_sensitivity": handoff_sensitivity(model),
    }


def summary(report: dict) -> str:
    m = report["model"]
    h = report["hardware"]
    out = [
        f"# {m['name']} — heterogeneous feasibility report",
        "",
        "> Analytical requirements/ceilings, not measured hardware performance.",
        "",
        f"- total: {m['total_b']:.1f}B",
        f"- active: {m['active_b']:.1f}B",
        f"- routed active derived: {m['routed_active_b']:.3f}B",
        f"- non-routed planning remainder: {m['derived_non_routed_active_b']:.3f}B",
        f"- host RAM: {h['host_ram_gib']:.0f} GiB",
        f"- H2D/GPU: {h['h2d_gbps_per_gpu']:.1f} GB/s",
        f"- CPU sockets: {h['cpu_sockets']}",
        f"- CPU ISA: {h['cpu_instruction_set']}",
        "",
        "## Streaming",
        "",
        "| strategy | format | GB/token | ms | tok/s ceiling |",
        "|---|---|---:|---:|---:|",
    ]
    wanted = {"tw_q4_g32_f32s", "q2_g64_f16s", "ideal_1bit"}
    for r in report["streaming"]:
        if "error" in r or r["format"] not in wanted:
            continue
        out.append(
            f"| {r['strategy']} | {r['format']} | {r['stream_gb_per_token']:.3f} | "
            f"{r['transfer_ms']:.1f} | {r['combined_roofline_tps']:.3f} |"
        )

    out += [
        "",
        "## CPU expert thresholds",
        "",
        "| sockets | format | target | GB/s/socket | GFLOP/s/socket | Gweights/s/socket | cycles/weight |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in report["cpu_expert_requirements"]:
        out.append(
            f"| {r['sockets']} | {r['format']} | {r['target_tps']:.0f} | "
            f"{r['required_mem_bw_gbps_per_socket']:.3f} | "
            f"{r['required_gflops_per_socket']:.3f} | "
            f"{r['required_gweights_per_s_per_socket']:.3f} | "
            f"{r['aggregate_core_cycles_per_weight_budget']:.3f} |"
        )

    bits = report["compression_only_required_bits_per_active_weight"]
    out += [
        "",
        "## Compression-only requirement if every active weight crosses one PCIe feed",
        "",
        f"- 5 tok/s: {bits['5_tps']:.3f} bits/active-weight",
        f"- 10 tok/s: {bits['10_tps']:.3f} bits/active-weight",
        "",
        "The CPU-expert path is not claimed to achieve 5/10 tok/s until the real low-bit R920 kernel meets the reported thresholds.",
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, default="k2.5")
    p.add_argument("--host-ram-gib", type=float, default=1024.0)
    p.add_argument("--cpu-sockets", type=int, default=4)
    p.add_argument("--h2d-gbps", type=float, default=12.0)
    p.add_argument("--effective-gpu-tflops", type=float, default=10.0)
    p.add_argument("--gpu-vram-gib", type=float, default=12.0)
    p.add_argument("--reserve-vram-gib", type=float, default=4.0)
    p.add_argument("--gpu-vram-bw-gbps", type=float, default=360.0)
    p.add_argument("--output-dir", type=Path, default=Path("out/heterogeneous-kimi"))
    a = p.parse_args()

    hw = Hardware(
        host_ram_gib=a.host_ram_gib,
        cpu_sockets=a.cpu_sockets,
        h2d_gbps_per_gpu=a.h2d_gbps,
        effective_gpu_tflops=a.effective_gpu_tflops,
        gpu_vram_gib=a.gpu_vram_gib,
        reserve_vram_gib=a.reserve_vram_gib,
        gpu_vram_bw_gbps=a.gpu_vram_bw_gbps,
    )
    report = build(MODELS[a.model], hw)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "simulation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (a.output_dir / "REPORT.md").write_text(summary(report), encoding="utf-8")
    print(summary(report))


if __name__ == "__main__":
    main()
