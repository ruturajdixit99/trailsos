"""Voice squeezing: Codec2 @ 1200 bps (pycodec2), Opus fallback via bundled ffmpeg."""
from __future__ import annotations

import io
import logging
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger("trail_sos.voice")
SAMPLE_RATE = 8000
CODEC2_MODE = 1200

try:
    import pycodec2  # type: ignore
    HAVE_CODEC2 = True
except Exception:  # pragma: no cover - depends on the machine
    HAVE_CODEC2 = False
    log.warning("pycodec2 unavailable - voice will FALL BACK to Opus via ffmpeg")


def load_wav(path: str | Path, seconds: float | None = 5.0) -> np.ndarray:
    """Load a WAV as 8 kHz mono int16, trimmed/padded to `seconds`."""
    with wave.open(str(path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError("expected 16-bit WAV")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(len(x) * SAMPLE_RATE / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    if seconds is not None:
        n = int(seconds * SAMPLE_RATE)
        x = np.pad(x, (0, max(0, n - len(x))))[:n]
    return np.clip(x, -32768, 32767).astype(np.int16)


def wav_bytes(pcm: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


def encode(pcm: np.ndarray) -> tuple[bytes, str]:
    """Returns (bitstream, codec_name)."""
    if HAVE_CODEC2:
        c2 = pycodec2.Codec2(CODEC2_MODE)
        spf = c2.samples_per_frame()
        n = len(pcm) - len(pcm) % spf
        out = b"".join(c2.encode(pcm[i:i + spf]) for i in range(0, n, spf))
        return out, f"codec2-{CODEC2_MODE}"
    log.warning("FALLBACK: encoding voice with Opus 6 kbps instead of Codec2")
    return _opus_encode(pcm), "opus-6k"


def decode(bits: bytes, codec: str) -> np.ndarray:
    if codec.startswith("codec2"):
        c2 = pycodec2.Codec2(CODEC2_MODE)
        bpf = (c2.bits_per_frame() + 7) // 8
        frames = [c2.decode(bits[i:i + bpf]) for i in range(0, len(bits) - len(bits) % bpf, bpf)]
        return np.concatenate(frames).astype(np.int16) if frames else np.zeros(0, np.int16)
    return _opus_decode(bits)


def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _opus_encode(pcm: np.ndarray) -> bytes:  # pragma: no cover - fallback path
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in.wav", Path(d) / "out.ogg"
        src.write_bytes(wav_bytes(pcm))
        subprocess.run([_ffmpeg(), "-loglevel", "error", "-y", "-i", str(src), "-c:a", "libopus",
                        "-b:a", "6k", "-application", "voip", str(dst)], check=True)
        return dst.read_bytes()


def _opus_decode(bits: bytes) -> np.ndarray:  # pragma: no cover - fallback path
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in.ogg", Path(d) / "out.wav"
        src.write_bytes(bits)
        subprocess.run([_ffmpeg(), "-loglevel", "error", "-y", "-i", str(src), "-ar", str(SAMPLE_RATE),
                        "-ac", "1", str(dst)], check=True)
        return load_wav(dst, seconds=None)
