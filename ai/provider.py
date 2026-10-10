"""AI provider adapter: caption_image / transcribe / rebuild_image.

* MockProvider - fixed outputs; used by tests and as the offline fallback.
* NicrronNinooProvider - sponsor tool.  Its API base URL, auth method, model
  name and request shape have NOT been supplied yet, so it refuses to guess:
  it raises ProviderNotConfigured and get_provider() falls back to the mock.
  Fill in `_call()` once the playground details are available.

Every call returns an AIResult with confidence; results below the threshold are
labeled "unverified", rebuilds are always labeled "AI reconstruction".
"""
from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass, field

from PIL import Image, ImageDraw

log = logging.getLogger("trail_sos.ai")

RECONSTRUCTION_LABEL = "AI reconstruction, not the real photo"
UNVERIFIED_LABEL = "unverified"


def confidence_threshold() -> float:
    return float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.7"))


@dataclass
class AIResult:
    kind: str                  # caption | transcript | rebuild
    output: object             # str for text, bytes (PNG) for rebuild
    confidence: float
    model: str
    latency_ms: int
    input_summary: str
    labels: list = field(default_factory=list)

    def text(self) -> str:
        return self.output if isinstance(self.output, str) else ""


def apply_labels(kind: str, confidence: float, threshold: float | None = None) -> list[str]:
    th = confidence_threshold() if threshold is None else threshold
    labels = []
    if kind == "rebuild":
        labels.append(RECONSTRUCTION_LABEL)
    if confidence < th:
        labels.append(UNVERIFIED_LABEL)
    return labels


class ProviderNotConfigured(RuntimeError):
    pass


class AIProvider:
    name = "base"
    model = "none"

    def caption_image(self, image: Image.Image) -> tuple[str, float]:
        raise NotImplementedError

    def transcribe(self, audio_wav: bytes) -> tuple[str, float]:
        raise NotImplementedError

    def rebuild_image(self, caption: str) -> bytes:
        """Returns PNG bytes of a sketch generated from the caption."""
        raise NotImplementedError

    # Timed wrappers that build labeled AIResults (used by the app + tracing)
    def run_caption(self, image: Image.Image) -> AIResult:
        t0 = time.perf_counter()
        text, conf = self.caption_image(image)
        return AIResult("caption", text, conf, self.model, _ms(t0), f"photo {image.width}x{image.height}",
                        apply_labels("caption", conf))

    def run_transcribe(self, audio_wav: bytes) -> AIResult:
        t0 = time.perf_counter()
        text, conf = self.transcribe(audio_wav)
        return AIResult("transcript", text, conf, self.model, _ms(t0), f"voice clip {len(audio_wav)} B WAV",
                        apply_labels("transcript", conf))

    def run_rebuild(self, caption: str, caption_confidence: float) -> AIResult:
        t0 = time.perf_counter()
        png = self.rebuild_image(caption)
        # a rebuild can be no more trustworthy than the caption it came from
        return AIResult("rebuild", png, caption_confidence, self.model, _ms(t0), caption,
                        apply_labels("rebuild", caption_confidence))


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


class MockProvider(AIProvider):
    name = "mock"
    model = "mock-fixed-v1"

    def __init__(self, caption_conf: float = 0.86, transcript_conf: float = 0.92):
        self.caption_conf = caption_conf
        self.transcript_conf = transcript_conf

    def caption_image(self, image):
        return ("Hiker in orange jacket sitting on a dirt trail, holding injured red-wrapped ankle; "
                "pine trees, snowy peak behind, clear sky.", self.caption_conf)

    def transcribe(self, audio_wav):
        return ("Help. I fell and hurt my ankle on the north ridge trail, near mile four. I can't walk.",
                self.transcript_conf)

    def rebuild_image(self, caption):
        return sketch_from_caption(caption)


def sketch_from_caption(caption: str) -> bytes:
    """Deterministic placeholder sketch, visibly stamped as an AI reconstruction."""
    W, H = 240, 240
    im = Image.new("RGB", (W, H), (236, 232, 222))
    d = ImageDraw.Draw(im)
    low = caption.lower()
    if "peak" in low or "mountain" in low:
        d.polygon([(0, 120), (70, 50), (130, 110), (190, 40), (240, 100), (240, 130), (0, 130)], outline=(90, 90, 90))
    if "trail" in low:
        d.polygon([(95, 240), (150, 240), (128, 130), (118, 130)], outline=(120, 90, 60))
    if "tree" in low or "pine" in low:
        for x in (20, 190):
            d.polygon([(x, 190), (x + 15, 140), (x + 30, 190)], outline=(40, 100, 50))
    if "hiker" in low or "person" in low:
        d.ellipse([112, 160, 128, 176], outline=(60, 60, 60))
        d.line([(120, 176), (120, 205)], fill=(200, 100, 30), width=3)
        d.line([(120, 205), (100, 220)], fill=(60, 60, 60), width=2)
        d.line([(120, 205), (142, 215)], fill=(60, 60, 60), width=2)
    d.rectangle([0, 0, W, 22], fill=(180, 30, 30))
    d.text((6, 5), "AI RECONSTRUCTION - NOT THE REAL PHOTO", fill=(255, 255, 255))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


class NicrronNinooProvider(AIProvider):
    """Sponsor provider.  Awaiting API details (base URL, auth, model, sample request)."""
    name = "nicrronninoo"

    def __init__(self):
        self.base_url = os.getenv("NICRRONNINOO_BASE_URL", "").rstrip("/")
        self.api_key = os.getenv("NICRRONNINOO_API_KEY", "")
        self.model = os.getenv("NICRRONNINOO_MODEL", "")
        if not (self.base_url and self.api_key and self.model):
            raise ProviderNotConfigured("NICRRONNINOO_BASE_URL / _API_KEY / _MODEL not set")

    def _call(self, *_a, **_k):
        raise ProviderNotConfigured(
            "NicrronNinoo request format not wired yet - paste the playground's sample request to finish this")

    def caption_image(self, image):
        return self._call("caption", image)

    def transcribe(self, audio_wav):
        return self._call("transcribe", audio_wav)

    def rebuild_image(self, caption):
        return self._call("rebuild", caption)


class FallbackProvider(AIProvider):
    """Tries the primary provider per call; on any error uses the mock and says so."""

    def __init__(self, primary: AIProvider, fallback: AIProvider):
        self.primary, self.fallback = primary, fallback
        self.name = f"{primary.name}+fallback"
        self.model = primary.model

    def _try(self, method, *args):
        try:
            self.model = self.primary.model
            return getattr(self.primary, method)(*args)
        except Exception as e:  # noqa: BLE001
            log.warning("%s.%s failed (%s); using mock provider", self.primary.name, method, e)
            self.model = self.fallback.model
            return getattr(self.fallback, method)(*args)

    def caption_image(self, image):
        return self._try("caption_image", image)

    def transcribe(self, audio_wav):
        return self._try("transcribe", audio_wav)

    def rebuild_image(self, caption):
        return self._try("rebuild_image", caption)


def get_provider() -> AIProvider:
    try:
        return FallbackProvider(NicrronNinooProvider(), MockProvider())
    except ProviderNotConfigured as e:
        log.info("NicrronNinoo not configured (%s) - using mock provider", e)
        return MockProvider()
