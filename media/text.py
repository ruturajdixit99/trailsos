"""Text parts (transcript, caption) carry their AI confidence in-band:
    u8 confidence (0-100) || UTF-8 text
so the receiver can label low-confidence results "unverified" itself."""
from __future__ import annotations

import re

CAPTION_BUDGET = 150
TRANSCRIPT_BUDGET = 197  # one packet payload (200) minus the chunker's 2-byte length prefix
ELLIPSIS = "…"      # visible marker that text was shortened to fit its packet


def truncate_utf8(text: str, max_bytes: int) -> bytes:
    b = text.encode("utf-8")[:max_bytes]
    while b:
        try:
            b.decode("utf-8")
            return b
        except UnicodeDecodeError:
            b = b[:-1]
    return b


def fit_text(text: str, max_bytes: int) -> tuple[str, bool]:
    """Shorten text to max_bytes UTF-8 without cutting a word in half.
    Prefers ending at a sentence, else at a word; a shortened text always ends with an ellipsis so the
    ranger can see it was cut. Returns (text, shortened)."""
    text = " ".join(text.split())
    if len(text.encode("utf-8")) <= max_bytes:
        return text, False
    room = max_bytes - len((" " + ELLIPSIS).encode("utf-8"))
    head = truncate_utf8(text, room).decode("utf-8")
    sentences = [m.end() for m in re.finditer(r"[.!?](?=\s)", head + " ")]
    if sentences and sentences[-1] >= 0.6 * len(head):  # a whole sentence keeps most of the content
        return head[:sentences[-1]].strip() + " " + ELLIPSIS, True
    cut = head.rsplit(" ", 1)[0] if " " in head else head
    return cut.rstrip(" ,;:-") + ELLIPSIS, True  # always marked: the ranger must know text was cut


def pack(text: str, confidence: float, max_bytes: int) -> bytes:
    conf = max(0, min(100, round(confidence * 100)))
    fitted, _ = fit_text(text, max_bytes - 1)
    return bytes([conf]) + fitted.encode("utf-8")


def unpack(data: bytes) -> tuple[str, float]:
    return data[1:].decode("utf-8", errors="replace"), data[0] / 100.0
