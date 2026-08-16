import numpy as np

from host.protocol import Command, CommandFlags, Completion, CompletionStatus, Opcode
from host.reference_bitplane import (
    assert_exact,
    dot_bitplanes,
    dot_scalar,
    int4_to_bitplanes,
    int8_to_bitplanes,
    pack_planes_little,
    reconstruct_int4,
    reconstruct_int8,
    unpack_planes_little,
)


def test_edge_values_are_exact():
    w = np.asarray([-8, -7, -1, 0, 1, 6, 7, -8], dtype=np.int8)
    a = np.asarray([-128, -127, -1, 0, 1, 63, 127, 42], dtype=np.int8)
    assert_exact(w, a)
    assert dot_bitplanes(w, a).value == dot_scalar(w, a)


def test_random_vectors_are_exact():
    rng = np.random.default_rng(12345)
    for length in (8, 64, 256, 2048):
        for _ in range(10):
            w = rng.integers(-8, 8, size=length, dtype=np.int8)
            a = rng.integers(-128, 128, size=length, dtype=np.int16).astype(np.int8)
            assert_exact(w, a)


def test_bitplane_pack_round_trip():
    rng = np.random.default_rng(7)
    w = rng.integers(-8, 8, size=2048, dtype=np.int8)
    a = rng.integers(-128, 128, size=2048, dtype=np.int16).astype(np.int8)

    wp = int4_to_bitplanes(w)
    ap = int8_to_bitplanes(a)

    wp2 = unpack_planes_little(pack_planes_little(wp), 4, w.size)
    ap2 = unpack_planes_little(pack_planes_little(ap), 8, a.size)

    np.testing.assert_array_equal(reconstruct_int4(wp2), w)
    np.testing.assert_array_equal(reconstruct_int8(ap2), a)


def test_command_is_64_bytes_and_round_trips():
    cmd = Command(
        opcode=Opcode.RUN_EXPERT,
        sequence_id=0x1122334455667788,
        flags=CommandFlags.ACTIVATION_PRETRANSPOSED | CommandFlags.FENCE_AFTER,
        layer_id=17,
        expert_id=313,
        shard_id=4,
        format_id=1,
        activation_offset=0x100000,
        activation_bytes=7168,
        result_offset=0x200000,
        result_bytes=14336,
        scale_id=9,
    )
    blob = cmd.pack()
    assert len(blob) == 64
    assert Command.unpack(blob) == cmd


def test_completion_is_64_bytes_and_round_trips():
    completion = Completion(
        sequence_id=99,
        status=CompletionStatus.OK,
        cycles_total=1000,
        cycles_compute_active=900,
        ddr_bytes_read=16_777_216,
        pcie_bytes_rx=7168,
        pcie_bytes_tx=14336,
        weight_elements=33_030_000,
    )
    blob = completion.pack()
    assert len(blob) == 64
    assert Completion.unpack(blob) == completion
