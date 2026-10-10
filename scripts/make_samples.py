"""Builds the bundled sample data (all fake): a 5 s voice clip, a trail photo and a 5 s video clip."""
from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from media import voice  # noqa: E402

SAMPLES = ROOT / "samples"
W, H = 640, 480


def make_voice() -> None:
    raw = SAMPLES / "sos_voice_raw.wav"  # Windows SAPI text-to-speech, synthetic speaker
    pcm = voice.load_wav(raw, seconds=5.0)
    (SAMPLES / "sos_voice.wav").write_bytes(voice.wav_bytes(pcm))


def draw_scene(arm: float = 0.0, pan: float = 0.0, wave: bool = False) -> Image.Image:
    """The trail scene. `arm` = waving arm angle (rad), `pan` = camera pan in pixels."""
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    for y in range(H // 2):  # sky
        c = int(150 + 90 * y / (H / 2))
        d.line([(0, y), (W, y)], fill=(120, 170, c))
    p = lambda pts: [(x - pan, y) for x, y in pts]  # noqa: E731
    d.polygon(p([(-60, 260), (140, 110), (260, 230), (380, 90), (520, 220), (640, 150), (720, 200), (720, 300),
                 (-60, 300)]), fill=(95, 105, 120))
    d.polygon(p([(330, 125), (380, 90), (425, 130), (395, 120), (360, 135)]), fill=(240, 240, 245))
    d.rectangle([0, 280, W, H], fill=(70, 110, 60))
    d.polygon(p([(250, H), (390, H), (330, 280), (310, 280)]), fill=(150, 125, 90))  # trail
    for x in (40, 90, 520, 580, 610, 150):
        d.polygon(p([(x, 360), (x + 25, 270), (x + 50, 360)]), fill=(35, 75, 40))
        d.rectangle([x + 20 - pan, 360, x + 30 - pan, 380], fill=(80, 55, 35))
    # hiker sitting on the trail: orange jacket, blue backpack, red-wrapped ankle
    ox = -pan
    d.ellipse([300 + ox, 345, 324 + ox, 369], fill=(225, 185, 150))
    d.rectangle([296 + ox, 368, 330 + ox, 410], fill=(235, 110, 30))
    d.rectangle([326 + ox, 372, 344 + ox, 404], fill=(40, 60, 140))
    d.line([(300 + ox, 410), (262 + ox, 425)], fill=(50, 50, 70), width=9)
    d.line([(326 + ox, 410), (360 + ox, 418)], fill=(50, 50, 70), width=9)
    d.ellipse([352 + ox, 410, 372 + ox, 428], fill=(200, 40, 40))
    if wave:  # raised arm waving for attention
        sx, sy = 298 + ox, 374
        ex, ey = sx - 34 * math.cos(arm), sy - 40 * math.sin(arm) - 10
        d.line([(sx, sy), (ex, ey)], fill=(235, 110, 30), width=8)
        d.ellipse([ex - 6, ey - 6, ex + 6, ey + 6], fill=(225, 185, 150))
    return im.filter(ImageFilter.SMOOTH)


def make_photo() -> None:
    draw_scene().save(SAMPLES / "trail_photo.jpg", quality=88)


def make_video(seconds: float = 5.0, fps: int = 10) -> None:
    """A phone-style clip: the hiker waves while the camera slowly pans (H.264 MP4 via bundled ffmpeg)."""
    import imageio_ffmpeg
    n = int(seconds * fps)
    writer = imageio_ffmpeg.write_frames(str(SAMPLES / "trail_clip.mp4"), (W, H), fps=fps, codec="libx264",
                                         pix_fmt_out="yuv420p", quality=7, macro_block_size=16)
    writer.send(None)
    for i in range(n):
        t = i / fps
        frame = draw_scene(arm=0.9 + 0.6 * math.sin(2 * math.pi * 1.2 * t), pan=12 * t, wave=True)
        writer.send(frame.tobytes())
    writer.close()


if __name__ == "__main__":
    make_voice()
    make_photo()
    make_video()
    print("samples written to", SAMPLES)
