"""Binary protocol reference for a Transit tile.

The transport/backend is intentionally not defined here. The same 64-byte
command can be placed in a PCIe DMA ring, BAR-backed queue, simulation FIFO or
unit-test byte array.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
import struct

TRANSIT_MAGIC = 0x54524E53  # ASCII-ish "TRNS" when treated as an integer tag.
PROTOCOL_VERSION = 1


class Opcode(IntEnum):
    IDENTIFY = 0x0001
    RESET = 0x0002
    LOAD_WEIGHT_BLOCK = 0x0010
    VERIFY_WEIGHT_BLOCK = 0x0011
    RUN_DOT = 0x0020
    RUN_MVM = 0x0021
    RUN_EXPERT = 0x0022
    READ_COUNTERS = 0x0030
    BARRIER = 0x0040
    ABORT = 0x00FF


class CommandFlags(IntFlag):
    NONE = 0
    ACTIVATION_PRETRANSPOSED = 1 << 0
    RESULT_INT64 = 1 << 1
    VERIFY_INPUT_CHECKSUM = 1 << 2
    FENCE_BEFORE = 1 << 3
    FENCE_AFTER = 1 << 4


# Exactly 64 bytes, little-endian, no native padding.
#
# I    magic
# H    protocol version
# H    opcode
# I    flags
# Q    sequence_id
# I    layer_id
# I    expert_id
# I    shard_id
# I    format_id
# Q    activation_offset
# I    activation_bytes
# Q    result_offset
# I    result_bytes
# I    scale_id
_COMMAND = struct.Struct("<IHHIQIIIIQIQII")
assert _COMMAND.size == 64


@dataclass(frozen=True)
class Command:
    opcode: Opcode
    sequence_id: int
    flags: CommandFlags = CommandFlags.NONE
    layer_id: int = 0
    expert_id: int = 0
    shard_id: int = 0
    format_id: int = 0
    activation_offset: int = 0
    activation_bytes: int = 0
    result_offset: int = 0
    result_bytes: int = 0
    scale_id: int = 0
    version: int = PROTOCOL_VERSION

    def pack(self) -> bytes:
        return _COMMAND.pack(
            TRANSIT_MAGIC,
            self.version,
            int(self.opcode),
            int(self.flags),
            self.sequence_id,
            self.layer_id,
            self.expert_id,
            self.shard_id,
            self.format_id,
            self.activation_offset,
            self.activation_bytes,
            self.result_offset,
            self.result_bytes,
            self.scale_id,
        )

    @classmethod
    def unpack(cls, blob: bytes) -> "Command":
        if len(blob) != _COMMAND.size:
            raise ValueError(f"command must be exactly {_COMMAND.size} bytes")
        (
            magic,
            version,
            opcode,
            flags,
            sequence_id,
            layer_id,
            expert_id,
            shard_id,
            format_id,
            activation_offset,
            activation_bytes,
            result_offset,
            result_bytes,
            scale_id,
        ) = _COMMAND.unpack(blob)
        if magic != TRANSIT_MAGIC:
            raise ValueError(f"bad Transit command magic 0x{magic:08x}")
        return cls(
            opcode=Opcode(opcode),
            sequence_id=sequence_id,
            flags=CommandFlags(flags),
            layer_id=layer_id,
            expert_id=expert_id,
            shard_id=shard_id,
            format_id=format_id,
            activation_offset=activation_offset,
            activation_bytes=activation_bytes,
            result_offset=result_offset,
            result_bytes=result_bytes,
            scale_id=scale_id,
            version=version,
        )


class CompletionStatus(IntEnum):
    OK = 0
    BAD_COMMAND = 1
    BAD_FORMAT = 2
    BAD_ADDRESS = 3
    CHECKSUM_ERROR = 4
    DDR_ERROR = 5
    INTERNAL_ERROR = 6
    ABORTED = 7


# Exactly 64 bytes.
# Q seq, I status, I error_flags, then six Q counters.
_COMPLETION = struct.Struct("<QIIQQQQQQ")
assert _COMPLETION.size == 64


@dataclass(frozen=True)
class Completion:
    sequence_id: int
    status: CompletionStatus = CompletionStatus.OK
    error_flags: int = 0
    cycles_total: int = 0
    cycles_compute_active: int = 0
    ddr_bytes_read: int = 0
    pcie_bytes_rx: int = 0
    pcie_bytes_tx: int = 0
    weight_elements: int = 0

    def pack(self) -> bytes:
        return _COMPLETION.pack(
            self.sequence_id,
            int(self.status),
            self.error_flags,
            self.cycles_total,
            self.cycles_compute_active,
            self.ddr_bytes_read,
            self.pcie_bytes_rx,
            self.pcie_bytes_tx,
            self.weight_elements,
        )

    @classmethod
    def unpack(cls, blob: bytes) -> "Completion":
        if len(blob) != _COMPLETION.size:
            raise ValueError(f"completion must be exactly {_COMPLETION.size} bytes")
        values = _COMPLETION.unpack(blob)
        return cls(
            sequence_id=values[0],
            status=CompletionStatus(values[1]),
            error_flags=values[2],
            cycles_total=values[3],
            cycles_compute_active=values[4],
            ddr_bytes_read=values[5],
            pcie_bytes_rx=values[6],
            pcie_bytes_tx=values[7],
            weight_elements=values[8],
        )


def describe_command_layout() -> str:
    return (
        "Transit command: 64 bytes little-endian; "
        "magic/version/opcode/flags/sequence + layer/expert/shard/format + "
        "activation DMA region + result DMA region + scale_id"
    )
