"""AI provider adapter: caption_image / transcribe / rebuild_image.

* MockProvider - fixed outputs; used by tests and as the offline fallback.
* NicrronNinooProvider - sponsor tool: Nicrron's OpenAI-compatible gateway
  (https://api.nicrron.ai/v1, Authorization: Bearer <key>).
    caption    -> POST /chat/completions        (vision model, image as data URL)
    transcribe -> POST /audio/transcriptions    (Whisper, verbose_json)
    rebuild    -> POST /images/generations      (async job: poll, then GET :id/content)
  Any failure falls back to the mock provider for that call, and the log says so.

Every call returns an AIResult with confidence; results below the threshold are
labeled "unverified", rebuilds are always labeled "AI reconstruction".
"""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field

import httpx
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

    def model_for(self, method: str) -> str:
        """Model that served the most recent call of `method` (for traces / UI)."""
        return self.model

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
        return AIResult("caption", text, conf, self.model_for("caption_image"), _ms(t0), f"photo {image.width}x{image.height}",
                        apply_labels("caption", conf))

    def run_transcribe(self, audio_wav: bytes) -> AIResult:
        t0 = time.perf_counter()
        text, conf = self.transcribe(audio_wav)
        return AIResult("transcript", text, conf, self.model_for("transcribe"), _ms(t0), f"voice clip {len(audio_wav)} B WAV",
                        apply_labels("transcript", conf))

    def run_rebuild(self, caption: str, caption_confidence: float) -> AIResult:
        t0 = time.perf_counter()
        png = self.rebuild_image(caption)
        # a rebuild can be no more trustworthy than the caption it came from
        return AIResult("rebuild", png, caption_confidence, self.model_for("rebuild_image"), _ms(t0), caption,
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
    """Sponsor provider: Nicrron's OpenAI-compatible gateway."""
    name = "nicrronninoo"
    CAPTION_PROMPT = (
        "You are helping a search-and-rescue ranger who will receive this over a tiny radio link. "
        "Describe the photo in at most 140 characters: the person's condition and any visible injury, "
        "posture, clothing colours, and terrain/landmarks useful for finding them. No speculation beyond "
        "what is visible. Reply ONLY as JSON: {\"caption\": string, \"confidence\": number 0-1} where "
        "confidence is how sure you are the caption is accurate.")

    def __init__(self, timeout: float = 60.0, transport: httpx.BaseTransport | None = None):
        self.base_url = os.getenv("NICRRONNINOO_BASE_URL", "https://api.nicrron.ai/v1").rstrip("/")
        self.api_key = os.getenv("NICRRONNINOO_API_KEY", "")
        self.model = os.getenv("NICRRONNINOO_MODEL", "anthropic/claude-haiku-4-5")
        self.stt_model = os.getenv("NICRRONNINOO_STT_MODEL", "openai/whisper-1")
        self.image_model = os.getenv("NICRRONNINOO_IMAGE_MODEL", "google/nano-banana")
        self.rebuild_enabled = os.getenv("NICRRONNINOO_REBUILD", "1") != "0"
        if not self.api_key:
            raise ProviderNotConfigured("NICRRONNINOO_API_KEY not set")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport,
                                    headers={"Authorization": f"Bearer {self.api_key}"})

    def model_for(self, method: str) -> str:
        return {"caption_image": self.model, "transcribe": self.stt_model,
                "rebuild_image": self.image_model}.get(method, self.model)

    def _ok(self, r: httpx.Response) -> httpx.Response:
        if r.status_code >= 400:
            raise RuntimeError(f"NicrronNinoo HTTP {r.status_code}: {r.text[:200]}")
        return r

    def caption_image(self, image):
        im = image.convert("RGB")
        im.thumbnail((512, 512))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        r = self._ok(self._client.post("/chat/completions", json={
            "model": self.model, "max_tokens": 200, "temperature": 0,
            "messages": [{"role": "user", "content": [{"type": "text", "text": self.CAPTION_PROMPT},
                                                      {"type": "image_url", "image_url": {"url": url}}]}]}))
        content = r.json()["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            raise RuntimeError(f"caption reply was not JSON: {content[:120]}")
        d = json.loads(m.group(0))
        conf = float(d.get("confidence", 0.5))
        # self-reported by the model; capped so it can never look more certain than it is
        return str(d["caption"]).strip(), max(0.0, min(conf, 0.95))

    def transcribe(self, audio_wav):
        r = self._ok(self._client.post("/audio/transcriptions",
                                       files={"file": ("sos.wav", audio_wav, "audio/wav")},
                                       data={"model": self.stt_model, "response_format": "verbose_json"}))
        d = r.json()
        segs = d.get("segments") or []
        if segs:  # Whisper's own token probabilities, duration-weighted; no-speech segments pull it down
            w = [max(sg.get("end", 0) - sg.get("start", 0), 0.01) for sg in segs]
            lp = sum(wi * sg.get("avg_logprob", -1.0) for wi, sg in zip(w, segs)) / sum(w)
            nos = sum(wi * sg.get("no_speech_prob", 0.0) for wi, sg in zip(w, segs)) / sum(w)
            conf = math.exp(lp) * (1 - nos)
        else:
            conf = 0.5
        return d.get("text", "").strip(), max(0.0, min(conf, 1.0))

    def rebuild_image(self, caption):
        if not self.rebuild_enabled:
            raise RuntimeError("NICRRONNINOO_REBUILD=0 (image generation disabled to save cost)")
        prompt = ("Simple grayscale pencil sketch, plain background, no text: " + caption)[:900]
        job = self._ok(self._client.post("/images/generations", json={"model": self.image_model, "prompt": prompt}))
        jid = job.json()["id"]
        deadline = time.time() + 60
        while True:
            st = self._ok(self._client.get(f"/images/generations/{jid}")).json().get("status")
            if st == "succeeded":
                break
            if st in ("failed", "expired") or time.time() > deadline:
                raise RuntimeError(f"image job {jid} ended as {st}")
            time.sleep(1.5)
        img = Image.open(io.BytesIO(self._ok(self._client.get(f"/images/generations/{jid}/content",
                                                               follow_redirects=True)).content))
        return stamp_reconstruction(img)


def stamp_reconstruction(img: Image.Image, size: int = 240) -> bytes:
    """Downscale and stamp a generated image so it can never pass as the real photo."""
    im = img.convert("RGB")
    im.thumbnail((size, size))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 22], fill=(180, 30, 30))
    d.text((6, 5), "AI RECONSTRUCTION - NOT THE REAL PHOTO", fill=(255, 255, 255))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


class FallbackProvider(AIProvider):
    """Tries the primary provider per call; on any error uses the mock and says so."""

    def __init__(self, primary: AIProvider, fallback: AIProvider):
        self.primary, self.fallback = primary, fallback
        self.name = primary.name
        self.model = primary.model
        self._served: dict = {}
        self.fallbacks: list = []  # (method, error) for the UI / logs

    def model_for(self, method: str) -> str:
        return self._served.get(method, self.primary.model_for(method))

    def _try(self, method, *args):
        try:
            out = getattr(self.primary, method)(*args)
            self._served[method] = self.primary.model_for(method)
            return out
        except Exception as e:  # noqa: BLE001
            log.warning("%s.%s failed (%s); FALLBACK to mock provider", self.primary.name, method, e)
            self.fallbacks.append((method, str(e)[:200]))
            self._served[method] = f"{self.fallback.model} (fallback: {self.primary.name} failed)"
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
