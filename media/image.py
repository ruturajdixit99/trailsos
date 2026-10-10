"""Layered image: caption (1 packet) -> 48x48 grayscale thumbnail (<=800 B).

Also: sent-vs-received image comparison (pixel-exact) and quality vs the original photo.
"""
from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image

THUMB_SIZE = (48, 48)
THUMB_BUDGET = 800


def reference_thumbnail(img: Image.Image) -> Image.Image:
    """The original photo reduced to the thumbnail's geometry, before any compression."""
    g = img.convert("L")
    w, h = g.size
    s = min(w, h)  # center crop to square
    return g.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize(THUMB_SIZE, Image.LANCZOS)


def make_thumbnail(img: Image.Image, budget: int = THUMB_BUDGET) -> tuple[bytes, str]:
    """Highest-quality 48x48 grayscale thumbnail that fits in `budget` bytes."""
    g = reference_thumbnail(img)
    for fmt in ("WEBP", "JPEG"):
        for q in range(90, 4, -5):
            buf = io.BytesIO()
            kw = {"quality": q, "optimize": True} if fmt == "JPEG" else {"quality": q, "method": 6}
            g.save(buf, fmt, **kw)
            if buf.tell() <= budget:
                return buf.getvalue(), fmt.lower()
    raise ValueError("thumbnail does not fit budget")


def _png(im: Image.Image, scale: int = 5) -> bytes:
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def thumbnail_png(data: bytes, scale: int = 5) -> bytes:
    """Decode a received thumbnail and upscale it for display (still only 48x48 of real data)."""
    return _png(Image.open(io.BytesIO(data)), scale)


def _gray(data_or_img) -> np.ndarray:
    im = data_or_img if isinstance(data_or_img, Image.Image) else Image.open(io.BytesIO(data_or_img))
    return np.asarray(im.convert("L"), dtype=np.int16)


def _heatmap(diff: np.ndarray, gain: float) -> Image.Image:
    """|difference| as a white->red map (white = identical)."""
    d = np.clip(np.abs(diff) * gain, 0, 255).astype(np.uint8)
    rgb = np.stack([np.full_like(d, 255), 255 - d, 255 - d], axis=-1)
    return Image.fromarray(rgb, "RGB")


def psnr(a: np.ndarray, b: np.ndarray) -> float | None:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return None if mse == 0 else round(10 * math.log10(255 ** 2 / mse), 1)


def compare_sent_received(sent: bytes, received: bytes) -> dict:
    """Pixel-exact comparison of what the hiker sent vs what the ranger decoded."""
    a, b = _gray(sent), _gray(received)
    if a.shape != b.shape:
        return {"identical": False, "same_pixels_pct": 0.0, "max_diff": 255, "psnr_db": None,
                "diff_png": None, "note": f"different sizes {a.shape} vs {b.shape}"}
    diff = b - a
    same = float(np.mean(diff == 0)) * 100
    return {"identical": bool(np.all(diff == 0)), "same_pixels_pct": round(same, 2),
            "max_diff": int(np.max(np.abs(diff))), "psnr_db": psnr(a, b),
            "diff_png": _png(_heatmap(diff, 8.0))}


def quality_vs_original(original: Image.Image, received: bytes) -> dict:
    """How much the 48x48 compression (not the network) changed the picture."""
    ref, rec = _gray(reference_thumbnail(original)), _gray(received)
    diff = rec - ref
    return {"psnr_db": psnr(ref, rec), "mean_abs_diff": round(float(np.mean(np.abs(diff))), 2),
            "reference_png": _png(Image.fromarray(ref.astype(np.uint8), "L")),
            "diff_png": _png(_heatmap(diff, 4.0))}
