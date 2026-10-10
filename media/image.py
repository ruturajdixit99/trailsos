"""Layered image: caption (1 packet) -> 48x48 grayscale thumbnail (<=800 B)."""
from __future__ import annotations

import io

from PIL import Image

THUMB_SIZE = (48, 48)
THUMB_BUDGET = 800


def make_thumbnail(img: Image.Image, budget: int = THUMB_BUDGET) -> tuple[bytes, str]:
    """Highest-quality 48x48 grayscale thumbnail that fits in `budget` bytes."""
    g = img.convert("L")
    w, h = g.size
    s = min(w, h)  # center crop to square
    g = g.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize(THUMB_SIZE, Image.LANCZOS)
    for fmt in ("WEBP", "JPEG"):
        for q in range(90, 4, -5):
            buf = io.BytesIO()
            kw = {"quality": q, "optimize": True} if fmt == "JPEG" else {"quality": q, "method": 6}
            g.save(buf, fmt, **kw)
            if buf.tell() <= budget:
                return buf.getvalue(), fmt.lower()
    raise ValueError("thumbnail does not fit budget")


def thumbnail_png(data: bytes, scale: int = 5) -> bytes:
    """Decode a received thumbnail and upscale it for display (still only 48x48 of real data)."""
    im = Image.open(io.BytesIO(data))
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()
