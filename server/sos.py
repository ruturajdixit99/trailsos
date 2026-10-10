"""Hiker side (build the SOS) and ranger side (decode what actually arrived)."""
from __future__ import annotations

import base64
import io
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ai.provider import AIProvider, AIResult, apply_labels
from media import image as mimage
from media import text as mtext
from media import voice
from media.verify import compare
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
    sent_text: dict = field(default_factory=dict)  # kind -> {"text", "shortened", "original_chars"} actually sent

    def sizes(self) -> dict:
        return {ContentType(k).name.lower(): len(v) for k, v in self.parts.items()}


def build_sos(provider: AIProvider, voice_path: Path = SAMPLE_VOICE, photo_path: Path = SAMPLE_PHOTO) -> SosPayload:
    pcm = voice.load_wav(voice_path, seconds=5.0)
    photo = Image.open(photo_path)
    with ThreadPoolExecutor(2) as pool:  # both AI calls at once: the hiker waits for the slower one only
        # 1 s of trailing silence for the speech model only (Whisper drops words that end right at the
        # edge of a clip); the voice note sent over the mesh is still exactly the 5 s clip
        stt_pcm = np.concatenate([pcm, np.zeros(voice.SAMPLE_RATE, dtype=pcm.dtype)])
        f_tr = pool.submit(provider.run_transcribe, voice.wav_bytes(stt_pcm))
        f_cap = pool.submit(provider.run_caption, photo)
        tr, cap = f_tr.result(), f_cap.result()
    bits, codec = voice.encode(pcm)
    thumb, fmt = mimage.make_thumbnail(photo)
    parts = {
        ContentType.TRANSCRIPT: mtext.pack(tr.text(), tr.confidence, mtext.TRANSCRIPT_BUDGET),
        ContentType.CAPTION: mtext.pack(cap.text(), cap.confidence, mtext.CAPTION_BUDGET),
        ContentType.AUDIO: bytes([CODEC_IDS[codec]]) + bits,
        ContentType.THUMBNAIL: thumb,
    }
    sent = {}
    for r, part in ((tr, ContentType.TRANSCRIPT), (cap, ContentType.CAPTION)):
        text, _ = mtext.unpack(parts[part])
        sent[r.kind] = {"text": text, "shortened": text != " ".join(r.text().split()),
                        "original_chars": len(r.text())}
    return SosPayload(parts, codec, fmt, len(pcm) / voice.SAMPLE_RATE, [tr, cap], sent)


@dataclass
class RangerView:
    """What the ranger's phone can show, built only from reassembled bytes."""
    provider: AIProvider
    media: dict = field(default_factory=dict)   # name -> (bytes, mime)
    view: dict = field(default_factory=dict)
    rebuild: AIResult | None = None
    raw: dict = field(default_factory=dict)     # part -> reassembled bytes exactly as received
    checked: set = field(default_factory=set)   # verifications already run
    cache: dict = field(default_factory=dict)   # computed-once comparisons

    def update(self, part: str, data: bytes, rebuild: bool = True) -> AIResult | None:
        """Decode a freshly reassembled part. Returns an AIResult if an AI call was made.
        rebuild=False leaves the (slow) caption->sketch call to the caller (see rebuild_from_caption)."""
        new_ai = None
        self.raw[part] = data
        if part in ("transcript", "caption"):
            text, conf = mtext.unpack(data)
            self.view[part] = {"text": text, "confidence": conf, "labels": apply_labels(part, conf),
                               "shortened": text.endswith(mtext.ELLIPSIS)}
            if part == "caption" and rebuild:
                new_ai = self.rebuild_from_caption()
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

    def rebuild_from_caption(self) -> AIResult | None:
        """Generate the labeled AI sketch from the received caption (once)."""
        if self.rebuild is not None or "caption" not in self.view:
            return None
        cap = self.view["caption"]
        self.view["rebuild"] = {"pending": True, "labels": apply_labels("rebuild", cap["confidence"])}
        self.rebuild = self.provider.run_rebuild(cap["text"], cap["confidence"])
        self.media["rebuild.png"] = (self.rebuild.output, "image/png")
        self.view["rebuild"] = {"url": "rebuild.png", "labels": self.rebuild.labels, "confidence": cap["confidence"],
                                "model": self.rebuild.model}
        return self.rebuild


    # ------------------------------------------------------------------ ranger-side verification
    def verify_transcript(self) -> AIResult | None:
        """Re-transcribe the voice note that actually arrived and diff it against the received transcript."""
        if "transcript_vs_audio" in self.checked or "transcript" not in self.view or "audio" not in self.raw:
            return None
        self.checked.add("transcript_vs_audio")
        sent_text = self.view["transcript"]["text"]
        self.view["verify_transcript"] = {"pending": True}
        codec = CODEC_NAMES.get(self.raw["audio"][0], "unknown")
        pcm = voice.decode(self.raw["audio"][1:], codec)
        pcm = np.concatenate([pcm, np.zeros(voice.SAMPLE_RATE, dtype=pcm.dtype)])  # trailing silence for STT
        ai = self.provider.run_transcribe(voice.wav_bytes(pcm))
        cmp = compare(sent_text, ai.text())
        self.view["verify_transcript"] = {**cmp, "ranger_text": ai.text(), "sent_text": sent_text,
                                          "confidence": ai.confidence, "model": ai.model}
        return ai

    def verify_caption(self) -> AIResult | None:
        """Ask the AI whether the received caption describes the received thumbnail."""
        if "caption_vs_image" in self.checked or "caption" not in self.view or "thumbnail" not in self.raw:
            return None
        self.checked.add("caption_vs_image")
        caption = self.view["caption"]["text"]
        self.view["verify_caption"] = {"pending": True}
        ai = self.provider.run_check_caption(Image.open(io.BytesIO(self.raw["thumbnail"])), caption)
        score = ai.confidence
        verdict = "consistent" if score >= 0.7 else "partly consistent" if score >= 0.4 else "inconsistent"
        self.view["verify_caption"] = {"score": round(score, 2), "note": ai.text(), "verdict": verdict,
                                       "caption": caption, "model": ai.model}
        return ai


def _crc(b: bytes | None) -> str | None:
    return None if b is None else f"{zlib.crc32(b) & 0xFFFFFFFF:08x}"


def sent_vs_received(payload: SosPayload, received: dict, ranger: RangerView, photo_path: Path = SAMPLE_PHOTO) -> list:
    """Per part: what the hiker sent vs what the ranger reassembled (bytes, CRC32, and content-level diff)."""
    rows = []
    for ct in (ContentType.TRANSCRIPT, ContentType.CAPTION, ContentType.AUDIO, ContentType.THUMBNAIL):
        if ct not in payload.parts:
            continue
        name = ct.name.lower()
        sent, got = payload.parts[ct], received.get(name)
        row = {"part": name, "sent_bytes": len(sent), "sent_crc": _crc(sent),
               "received_bytes": None if got is None else len(got), "received_crc": _crc(got),
               "status": "not yet received" if got is None else "identical" if got == sent else "DIFFERENT"}
        if ct in (ContentType.TRANSCRIPT, ContentType.CAPTION):
            row["sent_text"], row["sent_conf"] = mtext.unpack(sent)
            if got is not None:
                row["received_text"], row["received_conf"] = mtext.unpack(got)
                row["diff"] = compare(row["sent_text"], row["received_text"])
        elif ct == ContentType.AUDIO:
            row["sent_seconds"] = payload.voice_seconds
            row["original_url"] = "/samples/sos_voice.wav"
            if got is not None:
                row["received_url"] = "audio.wav"
                row["codec"] = CODEC_NAMES.get(got[0], "unknown")
        elif ct == ContentType.THUMBNAIL:
            if "sent_thumbnail.png" not in ranger.media:
                ranger.media["sent_thumbnail.png"] = (mimage.thumbnail_png(sent), "image/png")
            row["sent_url"] = "sent_thumbnail.png"
            row["original_url"] = "/samples/trail_photo.jpg"
            if got is not None:
                if "image_cmp" not in ranger.cache:
                    cmp = mimage.compare_sent_received(sent, got)
                    q = mimage.quality_vs_original(Image.open(photo_path), got)
                    diff_png = cmp.pop("diff_png")
                    if diff_png:
                        ranger.media["diff_sent_received.png"] = (diff_png, "image/png")
                    ranger.media["reference.png"] = (q.pop("reference_png"), "image/png")
                    ranger.media["diff_vs_original.png"] = (q.pop("diff_png"), "image/png")
                    ranger.cache["image_cmp"] = {"network": cmp, "compression": q}
                row.update(ranger.cache["image_cmp"])
                row["received_url"] = "thumbnail.png"
                row["diff_url"] = "diff_sent_received.png"
                row["reference_url"] = "reference.png"
                row["compression_diff_url"] = "diff_vs_original.png"
        rows.append(row)
    return rows


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()
