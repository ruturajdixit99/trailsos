"""Text parts (transcript, caption) carry their AI confidence in-band:
    u8 confidence (0-100) || UTF-8 text
so the receiver can label low-confidence results "unverified" itself."""
from __future__ import annotations

CAPTION_BUDGET = 150
TRANSCRIPT_BUDGET = 197  # one packet payload (200) minus the chunker's 2-byte length prefix


def truncate_utf8(text: str, max_bytes: int) -> bytes:
    b = text.encode("utf-8")[:max_bytes]
    while b:
        try:
            b.decode("utf-8")
            return b
        except UnicodeDecodeError:
            b = b[:-1]
    return b


def pack(text: str, confidence: float, max_bytes: int) -> bytes:
    conf = max(0, min(100, round(confidence * 100)))
    return bytes([conf]) + truncate_utf8(text, max_bytes - 1)


def unpack(data: bytes) -> tuple[str, float]:
    return data[1:].decode("utf-8", errors="replace"), data[0] / 100.0
