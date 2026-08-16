"""Transit Weight Atlas core data model.

This is intentionally model-agnostic. A future K3-specific checkpoint parser should
emit these records rather than baking guessed K3 tensor-name conventions into the
runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class WeightShard:
    tensor: str
    shard_id: int
    byte_length: int
    format_id: int
    checksum_sha256: str

    layer_id: int | None = None
    expert_id: int | None = None
    shape: tuple[int, ...] = ()
    scale_metadata: str | None = None

    # Placement is nullable while building/optimizing an atlas.
    tile_id: int | None = None
    channel_id: int | None = None
    local_address: int | None = None
    replica_group: int | None = None

    def validate(self, channels_per_tile: int | None = None) -> None:
        if not self.tensor:
            raise ValueError("tensor name must not be empty")
        if self.shard_id < 0:
            raise ValueError("shard_id must be >= 0")
        if self.byte_length <= 0:
            raise ValueError("byte_length must be > 0")
        if self.format_id < 0:
            raise ValueError("format_id must be >= 0")
        if len(self.checksum_sha256) != 64:
            raise ValueError("checksum_sha256 must contain 64 hex characters")
        try:
            int(self.checksum_sha256, 16)
        except ValueError as exc:
            raise ValueError("checksum_sha256 is not hexadecimal") from exc

        placed = (self.tile_id, self.channel_id, self.local_address)
        if any(v is None for v in placed) and not all(v is None for v in placed):
            raise ValueError("tile_id/channel_id/local_address must be all set or all unset")

        if self.tile_id is not None and self.tile_id < 0:
            raise ValueError("tile_id must be >= 0")
        if self.channel_id is not None:
            if self.channel_id < 0:
                raise ValueError("channel_id must be >= 0")
            if channels_per_tile is not None and self.channel_id >= channels_per_tile:
                raise ValueError(
                    f"channel_id={self.channel_id} exceeds channels_per_tile={channels_per_tile}"
                )
        if self.local_address is not None and self.local_address < 0:
            raise ValueError("local_address must be >= 0")

    @property
    def resident(self) -> bool:
        return self.tile_id is not None

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d

    @classmethod
    def from_json_dict(cls, d: dict) -> "WeightShard":
        d = dict(d)
        d["shape"] = tuple(d.get("shape", ()))
        return cls(**d)


class WeightAtlas:
    def __init__(self, shards: Iterable[WeightShard] = ()) -> None:
        self._shards = list(shards)

    def __iter__(self) -> Iterator[WeightShard]:
        return iter(self._shards)

    def __len__(self) -> int:
        return len(self._shards)

    def append(self, shard: WeightShard) -> None:
        self._shards.append(shard)

    def validate(self, channels_per_tile: int | None = None) -> None:
        seen = set()
        ranges: dict[tuple[int, int], list[tuple[int, int, str, int]]] = {}

        for shard in self._shards:
            shard.validate(channels_per_tile=channels_per_tile)
            key = (shard.tensor, shard.shard_id, shard.tile_id, shard.local_address)
            if key in seen:
                raise ValueError(f"duplicate atlas record: {key}")
            seen.add(key)

            if shard.resident:
                loc = (int(shard.tile_id), int(shard.channel_id))
                start = int(shard.local_address)
                end = start + shard.byte_length
                ranges.setdefault(loc, []).append((start, end, shard.tensor, shard.shard_id))

        # Prevent accidental overlapping weight allocations on one physical channel.
        for loc, entries in ranges.items():
            entries.sort()
            previous = None
            for entry in entries:
                if previous is not None and entry[0] < previous[1]:
                    raise ValueError(
                        f"overlap on tile/channel {loc}: "
                        f"{previous[2]}#{previous[3]} [{previous[0]}, {previous[1]}) vs "
                        f"{entry[2]}#{entry[3]} [{entry[0]}, {entry[1]})"
                    )
                previous = entry

    def total_bytes(self) -> int:
        return sum(s.byte_length for s in self._shards)

    def resident_bytes(self) -> int:
        return sum(s.byte_length for s in self._shards if s.resident)

    def for_expert(self, layer_id: int, expert_id: int) -> list[WeightShard]:
        return [
            s for s in self._shards
            if s.layer_id == layer_id and s.expert_id == expert_id
        ]

    def on_tile(self, tile_id: int) -> list[WeightShard]:
        return [s for s in self._shards if s.tile_id == tile_id]

    def placements_for_expert(self, layer_id: int, expert_id: int) -> dict[int, list[WeightShard]]:
        """Return resident expert shards grouped by tile ID."""
        grouped: dict[int, list[WeightShard]] = {}
        for shard in self.for_expert(layer_id, expert_id):
            if shard.tile_id is None:
                continue
            grouped.setdefault(shard.tile_id, []).append(shard)
        return grouped

    def save_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for shard in self._shards:
                f.write(json.dumps(shard.to_json_dict(), sort_keys=True, separators=(",", ":")))
                f.write("\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "WeightAtlas":
        shards = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    shards.append(WeightShard.from_json_dict(json.loads(line)))
                except Exception as exc:
                    raise ValueError(f"invalid atlas line {line_number}: {exc}") from exc
        return cls(shards)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
