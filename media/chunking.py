"""Chunking + XOR parity (one parity chunk per group of 4 data chunks).

A part's byte stream is  u16 length || data , split into <=200-byte chunks.
The length prefix lets the receiver trim padding from a parity-rebuilt last
chunk.  Any ONE missing chunk per group can be rebuilt without a resend.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from mesh.packet import MAX_PAYLOAD

GROUP = 4


def _xor(blocks: list[bytes]) -> bytes:
    n = max(len(b) for b in blocks)
    out = bytearray(n)
    for b in blocks:
        for i, v in enumerate(b):
            out[i] ^= v
    return bytes(out)


def split(data: bytes, chunk_size: int = MAX_PAYLOAD) -> list[bytes]:
    if len(data) > 0xFFFF:
        raise ValueError("part too large")
    stream = struct.pack(">H", len(data)) + data
    return [stream[i:i + chunk_size] for i in range(0, len(stream), chunk_size)]


def parity_chunks(chunks: list[bytes], group: int = GROUP) -> list[bytes]:
    """One XOR parity chunk per group of `group` data chunks."""
    return [_xor(chunks[g:g + group]) for g in range(0, len(chunks), group)]


def n_groups(total: int, group: int = GROUP) -> int:
    return (total + group - 1) // group


class ReassemblyError(Exception):
    pass


@dataclass
class PartAssembler:
    """Collects data + parity chunks for one part; tolerant of reorder and dupes."""
    total: int
    data: dict[int, bytes] = field(default_factory=dict)
    parity: dict[int, bytes] = field(default_factory=dict)
    recovered: int = 0  # chunks rebuilt from parity

    def add_data(self, index: int, payload: bytes) -> bool:
        """Returns True if this chunk was new."""
        if index >= self.total:
            raise ReassemblyError(f"index {index} >= total {self.total}")
        if index in self.data:
            return False
        self.data[index] = payload
        return True

    def add_parity(self, group: int, payload: bytes) -> bool:
        if group in self.parity:
            return False
        self.parity[group] = payload
        return True

    def try_recover(self) -> list[int]:
        """Rebuild any group missing exactly one chunk; returns rebuilt indexes."""
        rebuilt = []
        for g in range(n_groups(self.total)):
            idxs = list(range(g * GROUP, min((g + 1) * GROUP, self.total)))
            missing = [i for i in idxs if i not in self.data]
            if len(missing) == 1 and g in self.parity:
                have = [self.data[i] for i in idxs if i in self.data]
                self.data[missing[0]] = _xor(have + [self.parity[g]])
                self.recovered += 1
                rebuilt.append(missing[0])
        return rebuilt

    @property
    def received(self) -> int:
        return len(self.data)

    @property
    def complete(self) -> bool:
        return len(self.data) == self.total

    def missing(self) -> list[int]:
        return [i for i in range(self.total) if i not in self.data]

    def unrecoverable_groups(self) -> list[int]:
        """Groups that cannot be rebuilt from what is here now."""
        bad = []
        for g in range(n_groups(self.total)):
            idxs = range(g * GROUP, min((g + 1) * GROUP, self.total))
            miss = sum(1 for i in idxs if i not in self.data)
            if miss >= 2 or (miss == 1 and g not in self.parity):
                bad.append(g)
        return bad

    def assemble(self) -> bytes:
        if not self.complete:
            raise ReassemblyError(f"missing chunks {self.missing()}")
        stream = b"".join(self.data[i] for i in range(self.total))
        (n,) = struct.unpack(">H", stream[:2])
        body = stream[2:2 + n]
        if len(body) != n:
            raise ReassemblyError("length prefix larger than data")
        return body
