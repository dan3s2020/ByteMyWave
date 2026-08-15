#!/usr/bin/env python3
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

import distributed_r920_runtime as dr


def make_spec() -> dr.ClusterSpec:
    return dr.ClusterSpec(
        cluster_id="test-three-r920",
        model="k2.5",
        hidden_size=7168,
        expert_ff_size=2048,
        routed_experts=384,
        selected_experts=8,
        moe_layers=60,
        q4_bytes_per_weight=0.625,
        activation_dtype_bytes=2,
        nodes=(
            dr.NodeSpec("r920-0", "127.0.0.1", 29510, 4, "head", "RTX 3060 12GB"),
            dr.NodeSpec("r920-1", "127.0.0.1", 29511, 4),
            dr.NodeSpec("r920-2", "127.0.0.1", 29512, 4),
        ),
    )


class PlannerTests(unittest.TestCase):
    def test_exact_k25_census(self):
        spec = make_spec()
        dr.validate_cluster_spec(spec)
        summary = dr.planning_summary(spec)
        self.assertEqual(summary["sockets"], 12)
        self.assertEqual(summary["activation_bytes"], 14336)
        self.assertEqual(summary["expert_weights"], 44_040_192)
        self.assertEqual(summary["routed_weights_per_token"], 21_139_292_160)
        self.assertAlmostEqual(summary["routed_q4_bytes_per_token"], 13_212_057_600.0)
        self.assertAlmostEqual(summary["socket_gweights_s_required"]["10"], 17.6160768)

    def test_shards_cover_ff_dimension_without_overlap(self):
        spec = make_spec()
        for expert_id in (0, 1, 11, 12, 383):
            shards = dr.build_expert_shards(spec, expert_id)
            self.assertEqual(len(shards), 12)
            self.assertEqual(shards[0].ff_start, 0)
            self.assertEqual(shards[-1].ff_end, 2048)
            for a, b in zip(shards, shards[1:]):
                self.assertEqual(a.ff_end, b.ff_start)
            self.assertEqual(sorted(s.ff_rows for s in shards), [160] * 8 + [192] * 4)
            self.assertTrue(all(s.ff_start % 32 == 0 and s.ff_end % 32 == 0 for s in shards))
            self.assertEqual(sum(s.weights for s in shards), 44_040_192)

    def test_heavy_shards_rotate(self):
        spec = make_spec()
        heavy0 = {s.target.ordinal for s in dr.build_expert_shards(spec, 0) if s.ff_rows == 192}
        heavy1 = {s.target.ordinal for s in dr.build_expert_shards(spec, 1) if s.ff_rows == 192}
        self.assertNotEqual(heavy0, heavy1)
        self.assertEqual(heavy0, {0, 1, 2, 3})
        self.assertEqual(heavy1, {1, 2, 3, 4})


class ProtocolTests(unittest.TestCase):
    def test_frame_socket_roundtrip(self):
        a, b = socket.socketpair()
        try:
            frame = dr.Frame(dr.KIND_ACTIVATION, 123, 59, (1, 7, 383), b"abc" * 17)
            dr.send_frame(a, frame)
            got = dr.recv_frame(b)
            self.assertEqual(got, frame)
        finally:
            a.close()
            b.close()

    def test_worker_barrier_loopback(self):
        servers = [dr.WorkerServer(("127.0.0.1", 0), f"w{i}") for i in range(2)]
        threads = []
        try:
            for server in servers:
                t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
                t.start()
                threads.append(t)
            endpoints = [("127.0.0.1", server.server_address[1]) for server in servers]
            result = dr.benchmark_barrier(endpoints, rounds=5)
            self.assertEqual(result["workers"], 2)
            self.assertEqual(result["rounds"], 5)
            self.assertEqual(result["activation_bytes"], 14336)
            self.assertGreater(result["p50_us"], 0)
            self.assertGreaterEqual(result["p99_us"], result["p50_us"])
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for t in threads:
                t.join(timeout=1)


class ConfigTests(unittest.TestCase):
    def test_load_config(self):
        obj = {
            "cluster_id": "x",
            "model": "k2.5",
            "hidden_size": 7168,
            "expert_ff_size": 2048,
            "routed_experts": 384,
            "selected_experts": 8,
            "moe_layers": 60,
            "q4_bytes_per_weight": 0.625,
            "nodes": [
                {"node_id": "a", "host": "10.0.0.1", "port": 29510, "sockets": 4},
                {"node_id": "b", "host": "10.0.0.2", "port": 29510, "sockets": 4},
                {"node_id": "c", "host": "10.0.0.3", "port": 29510, "sockets": 4},
            ],
            "transport": {"activation_dtype_bytes": 2},
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cluster.json"
            path.write_text(json.dumps(obj), encoding="utf-8")
            spec = dr.load_cluster_spec(path)
        self.assertEqual(spec.socket_count, 12)
        self.assertEqual(spec.activation_bytes, 14336)


if __name__ == "__main__":
    unittest.main()
