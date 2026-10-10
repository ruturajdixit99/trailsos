"""Over-the-air packet format for Trail SOS.

Header (19 bytes, big-endian):
    version     u8
    msg_id      u32
    index       u16   chunk index within its part (parity: group number)
    total       u16   number of DATA chunks in the part
    ctype       u8    ContentType; high bit (0x80) marks a parity chunk
    parts_mask  u8    bitmask of parts in the whole message (receiver learns
                      about parts it has never seen a packet for)
    src         u8    original source node id
    dst         u8    final destination node id
    hop         u8    hop count, incremented by each relay
    gen         u8    retransmission generation (0 = first send; bumped by the
                      source on NACK/probe resends so relays don't dedup them)
    crc32       u32   CRC32 over header fields + payload

Payload: at most 200 bytes.  Total packet at most 219 bytes, below the 230
byte LoRa budget.  encode() enforces both limits.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, replace
from enum import IntEnum

MAX_PACKET = 230
MAX_PAYLOAD = 200
VERSION = 1
PARITY_FLAG = 0x80

_HDR_NO_CRC = struct.Struct(">BIHHBBBBBB")
_CRC = struct.Struct(">I")
HEADER_SIZE = _HDR_NO_CRC.size + _CRC.size  # 19


class ContentType(IntEnum):
    TRANSCRIPT = 1
    CAPTION = 2
    AUDIO = 3
    THUMBNAIL = 4
    ACK = 5        # hop-by-hop ack
    E2E_ACK = 6    # end-to-end ack: whole message verified at destination
    NACK = 7       # destination asks the source to resend specific chunks
    VIDEO = 8      # keyframes (animated WebP)


MEDIA_TYPES = (ContentType.TRANSCRIPT, ContentType.CAPTION, ContentType.AUDIO, ContentType.THUMBNAIL, ContentType.VIDEO)
PART_NAMES = {
    ContentType.TRANSCRIPT: "transcript",
    ContentType.CAPTION: "caption",
    ContentType.AUDIO: "audio",
    ContentType.THUMBNAIL: "thumbnail",
    ContentType.VIDEO: "video",
}


def part_bit(ct: ContentType) -> int:
    return 1 << MEDIA_TYPES.index(ct)  # media parts only: fits the u8 parts mask


class PacketError(ValueError):
    pass


class ChecksumError(PacketError):
    pass


@dataclass(frozen=True)
class Packet:
    msg_id: int
    index: int
    total: int
    ctype: int          # ContentType value, possibly | PARITY_FLAG
    src: int
    dst: int
    payload: bytes = b""
    hop: int = 0
    parts_mask: int = 0
    gen: int = 0

    @property
    def is_parity(self) -> bool:
        return bool(self.ctype & PARITY_FLAG)

    @property
    def base_type(self) -> ContentType:
        return ContentType(self.ctype & ~PARITY_FLAG)

    @property
    def key(self) -> tuple[int, int, int]:
        """Identity of a chunk, independent of hop count."""
        return (self.msg_id, self.ctype, self.index)

    @property
    def size(self) -> int:
        return HEADER_SIZE + len(self.payload)

    def with_hop(self, hop: int) -> "Packet":
        return replace(self, hop=hop)

    def encode(self) -> bytes:
        if len(self.payload) > MAX_PAYLOAD:
            raise PacketError(f"payload {len(self.payload)}B exceeds {MAX_PAYLOAD}B")
        head = _HDR_NO_CRC.pack(VERSION, self.msg_id, self.index, self.total, self.ctype,
                                self.parts_mask, self.src, self.dst, self.hop, self.gen)
        crc = zlib.crc32(head + self.payload) & 0xFFFFFFFF
        raw = head + _CRC.pack(crc) + self.payload
        if len(raw) > MAX_PACKET:  # cannot happen with the limits above; belt and braces
            raise PacketError(f"packet {len(raw)}B exceeds {MAX_PACKET}B")
        return raw

    @classmethod
    def decode(cls, raw: bytes) -> "Packet":
        if len(raw) < HEADER_SIZE:
            raise PacketError("packet shorter than header")
        if len(raw) > MAX_PACKET:
            raise PacketError(f"packet {len(raw)}B exceeds {MAX_PACKET}B")
        head = raw[:_HDR_NO_CRC.size]
        (crc,) = _CRC.unpack(raw[_HDR_NO_CRC.size:HEADER_SIZE])
        payload = raw[HEADER_SIZE:]
        if zlib.crc32(head + payload) & 0xFFFFFFFF != crc:
            raise ChecksumError("CRC32 mismatch")
        ver, msg_id, index, total, ctype, mask, src, dst, hop, gen = _HDR_NO_CRC.unpack(head)
        if ver != VERSION:
            raise PacketError(f"unknown version {ver}")
        return cls(msg_id=msg_id, index=index, total=total, ctype=ctype, src=src, dst=dst,
                   payload=payload, hop=hop, parts_mask=mask, gen=gen)


def airtime_s(nbytes: int, rate_bps: float) -> float:
    """Seconds on air for one transmission of nbytes at rate_bps."""
    return nbytes * 8.0 / rate_bps
