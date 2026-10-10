"""Builds the bundled sample data (all fake): a 5 s voice clip and a trail photo."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from media import voice  # noqa: E402

SAMPLES = ROOT / "samples"


def make_voice() -> None:
    raw = SAMPLES / "sos_voice_raw.wav"  # Windows SAPI text-to-speech, synthetic speaker
    pcm = voice.load_wav(raw, seconds=5.0)
    (SAMPLES / "sos_voice.wav").write_bytes(voice.wav_bytes(pcm))


def make_photo() -> None:
    W, H = 640, 480
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    for y in range(H // 2):  # sky
        c = int(150 + 90 * y / (H / 2))
        d.line([(0, y), (W, y)], fill=(120, 170, c))
    d.polygon([(0, 260), (140, 110), (260, 230), (380, 90), (520, 220), (640, 150), (640, 300), (0, 300)],
              fill=(95, 105, 120))
    d.polygon([(330, 125), (380, 90), (425, 130), (395, 120), (360, 135)], fill=(240, 240, 245))
    d.rectangle([0, 280, W, H], fill=(70, 110, 60))
    d.polygon([(250, H), (390, H), (330, 280), (310, 280)], fill=(150, 125, 90))  # trail
    for x in (40, 90, 520, 580, 610, 150):
        d.polygon([(x, 360), (x + 25, 270), (x + 50, 360)], fill=(35, 75, 40))
        d.rectangle([x + 20, 360, x + 30, 380], fill=(80, 55, 35))
    # hiker sitting on the trail: orange jacket, blue backpack, red-wrapped ankle
    d.ellipse([300, 345, 324, 369], fill=(225, 185, 150))
    d.rectangle([296, 368, 330, 410], fill=(235, 110, 30))
    d.rectangle([326, 372, 344, 404], fill=(40, 60, 140))
    d.line([(300, 410), (262, 425)], fill=(50, 50, 70), width=9)
    d.line([(326, 410), (360, 418)], fill=(50, 50, 70), width=9)
    d.ellipse([352, 410, 372, 428], fill=(200, 40, 40))
    im = im.filter(ImageFilter.SMOOTH)
    im.save(SAMPLES / "trail_photo.jpg", quality=88)


if __name__ == "__main__":
    make_voice()
    make_photo()
    print("samples written to", SAMPLES)
