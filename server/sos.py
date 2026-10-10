"""Hiker side (build the SOS) and ranger side (decode what actually arrived)."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ai.provider import AIProvider, AIResult, apply_labels
from media import image as mimage
from media import text as mtext
from media import voice
from mesh.packet import ContentType

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_VOICE = ROOT / "samples" / "sos_voice.wav"
SAMPLE_PHOTO = ROOT / "samples" / "trail_photo.jpg"
CODEC_IDS = {"codec2-1200": 1, "opus-6k": 2}
CODEC_NAMES = {v: k for k, v in CODEC_IDS.items()}


@dataclass
class SosPayload:
    parts: dict                      # ContentType -> bytes
    codec: str
    thumb_format: str
    voice_seconds: float
    ai_results: list = field(default_factory=list)

    def sizes(self) -> dict:
        return {ContentType(k).name.lower(): len(v) for k, v in self.parts.items()}


def build_sos(provider: AIProvider, voice_path: Path = SAMPLE_VOICE, photo_path: Path = SAMPLE_PHOTO) -> SosPayload:
    pcm = voice.load_wav(voice_path, seconds=5.0)
    photo = Image.open(photo_path)
    tr = provider.run_transcribe(voice.wav_bytes(pcm))
    cap = provider.run_caption(photo)
    bits, codec = voice.encode(pcm)
    thumb, fmt = mimage.make_thumbnail(photo)
    parts = {
        ContentType.TRANSCRIPT: mtext.pack(tr.text(), tr.confidence, mtext.TRANSCRIPT_BUDGET),
        ContentType.CAPTION: mtext.pack(cap.text(), cap.confidence, mtext.CAPTION_BUDGET),
        ContentType.AUDIO: bytes([CODEC_IDS[codec]]) + bits,
        ContentType.THUMBNAIL: thumb,
    }
    return SosPayload(parts, codec, fmt, len(pcm) / voice.SAMPLE_RATE, [tr, cap])


@dataclass
class RangerView:
    """What the ranger's phone can show, built only from reassembled bytes."""
    provider: AIProvider
    media: dict = field(default_factory=dict)   # name -> (bytes, mime)
    view: dict = field(default_factory=dict)
    rebuild: AIResult | None = None

    def update(self, part: str, data: bytes) -> AIResult | None:
        """Decode a freshly reassembled part. Returns an AIResult if an AI call was made."""
        new_ai = None
        if part in ("transcript", "caption"):
            text, conf = mtext.unpack(data)
            self.view[part] = {"text": text, "confidence": conf, "labels": apply_labels(part, conf)}
            if part == "caption" and self.rebuild is None:
                self.rebuild = new_ai = self.provider.run_rebuild(text, conf)
                self.media["rebuild.png"] = (self.rebuild.output, "image/png")
                self.view["rebuild"] = {"url": "rebuild.png", "labels": self.rebuild.labels,
                                        "confidence": conf}
        elif part == "audio":
            codec = CODEC_NAMES.get(data[0], "unknown")
            pcm = voice.decode(data[1:], codec)
            self.media["audio.wav"] = (voice.wav_bytes(pcm), "audio/wav")
            self.view["audio"] = {"url": "audio.wav", "codec": codec, "seconds": round(len(pcm) / voice.SAMPLE_RATE, 2),
                                  "bytes": len(data)}
        elif part == "thumbnail":
            self.media["thumbnail.png"] = (mimage.thumbnail_png(data), "image/png")
            self.view["thumbnail"] = {"url": "thumbnail.png", "bytes": len(data), "size": "48x48 grayscale"}
        return new_ai


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()
