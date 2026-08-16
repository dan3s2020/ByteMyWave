#!/usr/bin/env python3
"""TensorWave Phase 6: three-R920 distributed MoE control/data-plane prototype.

This module is intentionally standard-library only. It provides:
- deterministic Kimi K2.5 FF-dimension expert sharding across 12 sockets;
- a tiny framed TCP protocol for activation -> partial-result exchanges;
- a transport/barrier benchmark that measures p50/p95/p99 round-trip latency;
- a worker server with an echo-sized placeholder compute hook.

The worker server does NOT implement model inference yet. Its purpose is to prove
and measure the distributed transport/barrier contract before wiring the AVX Q4
expert-shard kernel and Weight Atlas-backed model bytes into the data plane.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import socketserver
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

MAGIC = b"TW3R"
PROTOCOL_VERSION = 1
KIND_ACTIVATION = 1
KIND_PARTIAL = 2
KIND_HELLO = 3
KIND_ERROR = 255
_HEADER = struct.Struct("<4sBBHQHBBI")
_U16 = struct.Struct("<H")


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    sockets: int
    role: str = "worker"
    gpu: str | None = None


@dataclass(frozen=True)
class ClusterSpec:
    cluster_id: str
    model: str
    hidden_size: int
    expert_ff_size: int
    routed_experts: int
    selected_experts: int
    moe_layers: int
    q4_bytes_per_weight: float
    activation_dtype_bytes: int
    nodes: tuple[NodeSpec, ...]

    @property
    def socket_count(self) -> int:
        return sum(n.sockets for n in self.nodes)

    @property
    def activation_bytes(self) -> int:
        return self.hidden_size * self.activation_dtype_bytes


@dataclass(frozen=True)
class SocketTarget:
    ordinal: int
    node_id: str
    socket_index: int


@dataclass(frozen=True)
class ExpertShard:
    expert_id: int
    target: SocketTarget
    ff_start: int
    ff_end: int
    group_size: int
    hidden_size: int
    q4_bytes_per_weight: float

    @property
    def ff_rows(self) -> int:
        return self.ff_end - self.ff_start

    @property
    def weights(self) -> int:
        # gate + up + down, sharded on the SwiGLU intermediate dimension.
        return 3 * self.hidden_size * self.ff_rows

    @property
    def q4_bytes(self) -> float:
        return self.weights * self.q4_bytes_per_weight


@dataclass(frozen=True)
class Frame:
    kind: int
    request_id: int
    layer_id: int
    expert_ids: tuple[int, ...]
    payload: bytes
    flags: int = 0


def load_cluster_spec(path: str | Path) -> ClusterSpec:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = tuple(
        NodeSpec(
            node_id=n["node_id"],
            host=n["host"],
            port=int(n["port"]),
            sockets=int(n["sockets"]),
            role=n.get("role", "worker"),
            gpu=n.get("gpu"),
        )
        for n in obj["nodes"]
    )
    transport = obj.get("transport", {})
    spec = ClusterSpec(
        cluster_id=obj["cluster_id"],
        model=obj["model"],
        hidden_size=int(obj["hidden_size"]),
        expert_ff_size=int(obj["expert_ff_size"]),
        routed_experts=int(obj["routed_experts"]),
        selected_experts=int(obj["selected_experts"]),
        moe_layers=int(obj["moe_layers"]),
        q4_bytes_per_weight=float(obj["q4_bytes_per_weight"]),
        activation_dtype_bytes=int(transport.get("activation_dtype_bytes", 2)),
        nodes=nodes,
    )
    validate_cluster_spec(spec)
    return spec


def validate_cluster_spec(spec: ClusterSpec, group_size: int = 32) -> None:
    if spec.socket_count <= 0:
        raise ValueError("cluster must expose at least one CPU socket")
    if spec.expert_ff_size % group_size:
        raise ValueError("expert FF size must be divisible by Q4 group size")
    if spec.routed_experts <= 0 or spec.selected_experts <= 0:
        raise ValueError("expert counts must be positive")
    if spec.selected_experts > spec.routed_experts:
        raise ValueError("selected experts cannot exceed routed experts")
    ids = [n.node_id for n in spec.nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("node_id values must be unique")
    if any(n.sockets <= 0 for n in spec.nodes):
        raise ValueError("each node must expose at least one socket")


def socket_targets(spec: ClusterSpec) -> tuple[SocketTarget, ...]:
    targets: list[SocketTarget] = []
    ordinal = 0
    for node in spec.nodes:
        for socket_index in range(node.sockets):
            targets.append(SocketTarget(ordinal, node.node_id, socket_index))
            ordinal += 1
    return tuple(targets)


def build_expert_shards(
    spec: ClusterSpec, expert_id: int, group_size: int = 32
) -> tuple[ExpertShard, ...]:
    """Shard one expert's intermediate dimension across every CPU socket.

    K2.5 has FF=2048 => 64 Q4 groups of 32. Across 12 sockets this becomes
    eight 160-row shards and four 192-row shards. The four heavier shards are
    rotated by expert_id so no fixed socket permanently receives the extra work.
    """
    if not 0 <= expert_id < spec.routed_experts:
        raise ValueError(f"expert_id out of range: {expert_id}")
    targets = socket_targets(spec)
    total_groups = spec.expert_ff_size // group_size
    base, extra = divmod(total_groups, len(targets))
    heavy = {(expert_id + i) % len(targets) for i in range(extra)}

    shards: list[ExpertShard] = []
    cursor = 0
    for target in targets:
        groups = base + (1 if target.ordinal in heavy else 0)
        start = cursor * group_size
        cursor += groups
        end = cursor * group_size
        shards.append(
            ExpertShard(
                expert_id=expert_id,
                target=target,
                ff_start=start,
                ff_end=end,
                group_size=group_size,
                hidden_size=spec.hidden_size,
                q4_bytes_per_weight=spec.q4_bytes_per_weight,
            )
        )
    if cursor != total_groups:
        raise AssertionError("shard planner failed exact FF coverage")
    return tuple(shards)


def planning_summary(spec: ClusterSpec) -> dict:
    shards = build_expert_shards(spec, 0)
    per_expert_weights = sum(s.weights for s in shards)
    expected_expert_weights = 3 * spec.hidden_size * spec.expert_ff_size
    if per_expert_weights != expected_expert_weights:
        raise AssertionError("expert shard weight census mismatch")

    routed_weights_per_token = (
        spec.moe_layers * spec.selected_experts * expected_expert_weights
    )
    routed_q4_bytes_per_token = routed_weights_per_token * spec.q4_bytes_per_weight
    per_socket_avg_weights = routed_weights_per_token / spec.socket_count
    return {
        "cluster_id": spec.cluster_id,
        "model": spec.model,
        "nodes": len(spec.nodes),
        "sockets": spec.socket_count,
        "activation_bytes": spec.activation_bytes,
        "expert_weights": expected_expert_weights,
        "expert_q4_bytes": expected_expert_weights * spec.q4_bytes_per_weight,
        "routed_weights_per_token": routed_weights_per_token,
        "routed_q4_bytes_per_token": routed_q4_bytes_per_token,
        "average_routed_weights_per_token_per_socket": per_socket_avg_weights,
        "socket_gweights_s_required": {
            str(tps): per_socket_avg_weights * tps / 1e9
            for tps in (5, 6, 10, 12, 15, 20)
        },
        "expert0_shards": [
            {
                "node_id": s.target.node_id,
                "socket_index": s.target.socket_index,
                "ff_start": s.ff_start,
                "ff_end": s.ff_end,
                "ff_rows": s.ff_rows,
                "weights": s.weights,
                "q4_bytes": s.q4_bytes,
            }
            for s in shards
        ],
    }


def pack_frame(frame: Frame) -> bytes:
    if len(frame.expert_ids) > 255:
        raise ValueError("protocol supports at most 255 expert IDs per frame")
    if frame.layer_id < 0 or frame.layer_id > 0xFFFF:
        raise ValueError("layer_id must fit uint16")
    if any(e < 0 or e > 0xFFFF for e in frame.expert_ids):
        raise ValueError("expert IDs must fit uint16")
    header = _HEADER.pack(
        MAGIC,
        PROTOCOL_VERSION,
        frame.kind,
        frame.flags,
        frame.request_id,
        frame.layer_id,
        len(frame.expert_ids),
        0,
        len(frame.payload),
    )
    experts = b"".join(_U16.pack(e) for e in frame.expert_ids)
    return header + experts + frame.payload


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed while receiving TensorWave frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(sock: socket.socket) -> Frame:
    raw = _recv_exact(sock, _HEADER.size)
    magic, version, kind, flags, request_id, layer_id, n_experts, _, payload_len = _HEADER.unpack(raw)
    if magic != MAGIC:
        raise ValueError("bad TensorWave frame magic")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported TensorWave protocol version {version}")
    expert_raw = _recv_exact(sock, n_experts * _U16.size)
    expert_ids = tuple(
        _U16.unpack_from(expert_raw, i * _U16.size)[0] for i in range(n_experts)
    )
    payload = _recv_exact(sock, payload_len)
    return Frame(kind, request_id, layer_id, expert_ids, payload, flags)


def send_frame(sock: socket.socket, frame: Frame) -> None:
    sock.sendall(pack_frame(frame))


class _WorkerHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server: "WorkerServer" = self.server  # type: ignore[assignment]
        while True:
            try:
                frame = recv_frame(self.request)
            except EOFError:
                return
            if frame.kind == KIND_HELLO:
                send_frame(
                    self.request,
                    Frame(KIND_HELLO, frame.request_id, frame.layer_id, (), server.node_id.encode()),
                )
                continue
            if frame.kind != KIND_ACTIVATION:
                send_frame(
                    self.request,
                    Frame(KIND_ERROR, frame.request_id, frame.layer_id, (), b"unexpected frame kind"),
                )
                continue
            # Transport prototype only: preserve the output vector size and return
            # a deterministic bytewise transform. The real implementation will
            # replace this hook with the Q4 expert-shard kernel + local reduction.
            payload = bytes((b ^ server.transform_mask) for b in frame.payload)
            send_frame(
                self.request,
                Frame(
                    KIND_PARTIAL,
                    frame.request_id,
                    frame.layer_id,
                    frame.expert_ids,
                    payload,
                ),
            )


class WorkerServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], node_id: str, transform_mask: int = 0x5A):
        self.node_id = node_id
        self.transform_mask = transform_mask & 0xFF
        super().__init__(address, _WorkerHandler)


class WorkerConnection:
    def __init__(self, host: str, port: int, timeout_s: float = 2.0):
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.sock.settimeout(timeout_s)

    def close(self) -> None:
        self.sock.close()

    def request(self, frame: Frame) -> Frame:
        send_frame(self.sock, frame)
        return recv_frame(self.sock)


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def benchmark_barrier(
    worker_endpoints: Sequence[tuple[str, int]],
    rounds: int,
    hidden_size: int = 7168,
    dtype_bytes: int = 2,
    selected_experts: Sequence[int] = tuple(range(8)),
    timeout_s: float = 2.0,
) -> dict:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if not worker_endpoints:
        raise ValueError("at least one remote worker endpoint is required")
    payload = bytes((i * 17 + 3) & 0xFF for i in range(hidden_size * dtype_bytes))
    conns = [WorkerConnection(h, p, timeout_s) for h, p in worker_endpoints]
    latencies_us: list[float] = []
    try:
        for i, conn in enumerate(conns):
            reply = conn.request(Frame(KIND_HELLO, i, 0, (), b"head"))
            if reply.kind != KIND_HELLO:
                raise RuntimeError("worker HELLO failed")

        for request_id in range(rounds):
            frame = Frame(
                KIND_ACTIVATION,
                request_id,
                request_id % 60,
                tuple(selected_experts),
                payload,
            )
            t0 = time.perf_counter_ns()
            # Send to all nodes first, then wait for every partial: this models
            # the layer barrier rather than serial RPC latency.
            for conn in conns:
                send_frame(conn.sock, frame)
            replies = [recv_frame(conn.sock) for conn in conns]
            t1 = time.perf_counter_ns()
            if any(r.kind != KIND_PARTIAL or len(r.payload) != len(payload) for r in replies):
                raise RuntimeError("invalid partial response")
            latencies_us.append((t1 - t0) / 1000.0)
    finally:
        for conn in conns:
            conn.close()

    return {
        "workers": len(worker_endpoints),
        "rounds": rounds,
        "activation_bytes": len(payload),
        "p50_us": percentile(latencies_us, 0.50),
        "p95_us": percentile(latencies_us, 0.95),
        "p99_us": percentile(latencies_us, 0.99),
        "min_us": min(latencies_us),
        "max_us": max(latencies_us),
        "mean_us": sum(latencies_us) / len(latencies_us),
        "raw_us": latencies_us,
    }


def _parse_endpoint(text: str) -> tuple[str, int]:
    host, sep, port = text.rpartition(":")
    if not sep or not host:
        raise argparse.ArgumentTypeError("endpoint must be HOST:PORT")
    try:
        return host, int(port)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid endpoint port") from exc


def _cmd_plan(args: argparse.Namespace) -> int:
    spec = load_cluster_spec(args.config)
    print(json.dumps(planning_summary(spec), indent=2, sort_keys=True))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    with WorkerServer((args.listen, args.port), args.node_id, args.transform_mask) as server:
        host, port = server.server_address
        print(f"TensorWave worker {args.node_id} listening on {host}:{port}", flush=True)
        server.serve_forever(poll_interval=0.1)
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    result = benchmark_barrier(
        args.worker,
        rounds=args.rounds,
        hidden_size=args.hidden_size,
        dtype_bytes=args.dtype_bytes,
        timeout_s=args.timeout,
    )
    raw = result.pop("raw_us")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.raw_output:
        Path(args.raw_output).write_text("\n".join(f"{x:.3f}" for x in raw) + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="validate config and print deterministic 12-socket plan")
    plan.add_argument("--config", required=True)
    plan.set_defaults(func=_cmd_plan)

    serve = sub.add_parser("serve", help="run a transport-prototype worker")
    serve.add_argument("--listen", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=29510)
    serve.add_argument("--node-id", required=True)
    serve.add_argument("--transform-mask", type=lambda x: int(x, 0), default=0x5A)
    serve.set_defaults(func=_cmd_serve)

    bench = sub.add_parser("bench", help="measure activation barrier RTT against remote workers")
    bench.add_argument("--worker", action="append", type=_parse_endpoint, required=True)
    bench.add_argument("--rounds", type=int, default=200)
    bench.add_argument("--hidden-size", type=int, default=7168)
    bench.add_argument("--dtype-bytes", type=int, default=2)
    bench.add_argument("--timeout", type=float, default=2.0)
    bench.add_argument("--raw-output")
    bench.set_defaults(func=_cmd_bench)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
