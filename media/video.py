"""Video squeezing: keyframes -> tiny animated WebP (+ an AI description sent as text).

1. Decode the clip (bundled ffmpeg) and downscale every frame to 48x48 grayscale.
2. Keyframes: the first frame, then each frame that differs enough from the last kept one
   (mean absolute pixel change), capped at `max_frames`; the last frame is always kept.
3. Encode the keyframes as one animated WebP, keeping each frame's real timing, and
   lower the quality (then the frame count) until it fits the byte budget.
The receiver gets a flipbook: the real (tiny) frames, at the right times.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from media.image import THUMB_SIZE, reference_thumbnail

VIDEO_BUDGET = 1600  # bytes: 8 packets of data


def read_frames(path: str | Path) -> tuple[list[Image.Image], float]:
    """All frames of a video file as RGB PIL images, plus fps."""
    import imageio_ffmpeg
    gen = imageio_ffmpeg.read_frames(str(path))
    meta = gen.__next__()
    w, h = meta["size"]
    frames = [Image.frombytes("RGB", (w, h), f) for f in gen]
    return frames, float(meta.get("fps") or 10.0)


def pick_keyframes(frames: list[Image.Image], fps: float, max_frames: int = 6,
                   threshold: float = 6.0) -> list[tuple[float, Image.Image]]:
    """(time_s, 48x48 grayscale frame) for frames that change the picture."""
    small = [reference_thumbnail(f) for f in frames]
    arrs = [np.asarray(s, dtype=np.int16) for s in small]
    keep = [0]
    for i in range(1, len(arrs)):
        if np.mean(np.abs(arrs[i] - arrs[keep[-1]])) >= threshold:
            keep.append(i)
    if keep[-1] != len(arrs) - 1:
        keep.append(len(arrs) - 1)
    if len(keep) > max_frames:  # evenly thin, always keep first and last
        idx = np.linspace(0, len(keep) - 1, max_frames).round().astype(int)
        keep = [keep[i] for i in sorted(set(idx))]
    return [(i / fps, small[i]) for i in keep]


def encode(keyframes: list[tuple[float, Image.Image]], budget: int = VIDEO_BUDGET,
           total_s: float | None = None) -> tuple[bytes, int]:
    """Animated WebP of the keyframes that fits `budget`. Returns (bytes, frames used)."""
    kf = list(keyframes)
    while kf:
        times = [t for t, _ in kf] + [total_s if total_s is not None else kf[-1][0] + 1.0]
        durations = [max(int((times[i + 1] - times[i]) * 1000), 100) for i in range(len(kf))]
        for q in range(80, 4, -5):
            buf = io.BytesIO()
            kf[0][1].save(buf, "WEBP", save_all=True, append_images=[f for _, f in kf[1:]], duration=durations,
                          loop=0, quality=q, method=6)
            if buf.tell() <= budget:
                return buf.getvalue(), len(kf)
        # too big even at low quality: drop a middle frame and retry
        if len(kf) <= 2:
            break
        kf.pop(len(kf) // 2)
    raise ValueError("video does not fit budget")


def squeeze(path: str | Path, budget: int = VIDEO_BUDGET, max_frames: int = 6) -> dict:
    frames, fps = read_frames(path)
    kf = pick_keyframes(frames, fps, max_frames=max_frames)
    data, n = encode(kf, budget, total_s=len(frames) / fps)
    mid = frames[len(frames) // 2]
    return {"bytes": data, "frames_sent": n, "frames_total": len(frames), "fps": fps,
            "seconds": round(len(frames) / fps, 2), "keyframe_times": [round(t, 2) for t, _ in kf][:n],
            "describe_frame": mid}


def decode_frames(data: bytes) -> list[tuple[Image.Image, int]]:
    """(frame, duration_ms) pairs from the received animated WebP."""
    im = Image.open(io.BytesIO(data))
    out = []
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        out.append((im.convert("L").copy(), int(im.info.get("duration", 500))))
    return out


def flipbook_gif(data: bytes, scale: int = 5) -> bytes:
    """Upscaled animated GIF of the received keyframes for display (same real frames, same timing)."""
    frames = decode_frames(data)
    big = [f.resize((THUMB_SIZE[0] * scale, THUMB_SIZE[1] * scale), Image.NEAREST) for f, _ in frames]
    buf = io.BytesIO()
    big[0].save(buf, "GIF", save_all=True, append_images=big[1:], duration=[d for _, d in frames], loop=0)
    return buf.getvalue()


def frames_png(data: bytes, scale: int = 3) -> list[bytes]:
    out = []
    for f, _ in decode_frames(data):
        b = io.BytesIO()
        f.resize((f.width * scale, f.height * scale), Image.NEAREST).save(b, "PNG")
        out.append(b.getvalue())
    return out
