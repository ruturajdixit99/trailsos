"""Transfer lab: send text / audio / image over the simulated mesh many times and measure
how fast each arrives and how much is lost on the way.

Every content variant is sent as its own message (so the comparison is clean), across
`trials` seeded runs.  All numbers come from simulator events.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from media import chunking, voice
from mesh.ledger import NetworkLedger
from mesh.packet import ContentType
from mesh.sim import MeshSim, SimConfig, demo_topology
from mesh.status import StatusTracker

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TEXT = "Help. I hurt my ankle on North Ridge trail, mile four. I can't walk."
SAMPLE_CAPTION = "Hiker in orange jacket sitting on a dirt trail, holding an injured ankle; pine trees, snowy peak."


def _mulaw(pcm: np.ndarray) -> bytes:
    """G.711 mu-law: the classic 'uncompressed' 8 kHz telephone voice, 1 byte per sample."""
    x = pcm.astype(np.float64) / 32768.0
    y = np.sign(x) * np.log1p(255 * np.abs(x)) / np.log1p(255)
    return ((y + 1) / 2 * 255).round().astype(np.uint8).tobytes()


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    kind: str            # text | audio | image | mixed
    note: str


VARIANTS = [
    Variant("text", "Text", "text", "voice transcript, UTF-8"),
    Variant("audio", "Audio (Codec2)", "audio", "5 s voice, Codec2 1200 bps"),
    Variant("image", "Image (thumbnail)", "image", "48x48 grayscale WebP"),
    Variant("sos", "Full SOS", "mixed", "text + caption + audio + image in one message"),
    Variant("audio_raw", "Audio (uncompressed)", "audio", "5 s voice, G.711 8 kHz"),
    Variant("image_raw", "Image (original)", "image", "640x480 JPEG"),
]
VARIANT_BY_KEY = {v.key: v for v in VARIANTS}


@lru_cache(maxsize=1)
def payloads() -> dict[str, dict]:
    """Bytes for each variant, built once from the bundled samples (no AI calls: repeatable)."""
    from PIL import Image

    from media import image as mimage
    from media import text as mtext
    pcm = voice.load_wav(ROOT / "samples" / "sos_voice.wav", seconds=5.0)
    photo = Image.open(ROOT / "samples" / "trail_photo.jpg")
    text = mtext.pack(SAMPLE_TEXT, 0.9, mtext.TRANSCRIPT_BUDGET)
    caption = mtext.pack(SAMPLE_CAPTION, 0.86, mtext.CAPTION_BUDGET)
    codec2 = bytes([1]) + voice.encode(pcm)[0]
    thumb = mimage.make_thumbnail(photo)[0]
    parts = {
        "text": {ContentType.TRANSCRIPT: text},
        "audio": {ContentType.AUDIO: codec2},
        "image": {ContentType.THUMBNAIL: thumb},
        "sos": {ContentType.TRANSCRIPT: text, ContentType.CAPTION: caption,
                ContentType.AUDIO: codec2, ContentType.THUMBNAIL: thumb},
        "audio_raw": {ContentType.AUDIO: _mulaw(pcm)},
        "image_raw": {ContentType.THUMBNAIL: (ROOT / "samples" / "trail_photo.jpg").read_bytes()},
    }
    out = {}
    for k, p in parts.items():
        n_data = sum(len(chunking.split(b)) for b in p.values())
        n_par = sum(chunking.n_groups(len(chunking.split(b))) for b in p.values())
        out[k] = {"parts": p, "bytes": sum(len(b) for b in p.values()), "packets": n_data + n_par,
                  "data_packets": n_data}
    return out


@dataclass
class LabConfig:
    variants: tuple = ("text", "audio", "image", "sos")
    radio: str = "simple"
    loss: float = 0.2
    rate_bps: float = 1000.0
    lora_preset: str = "long_fast"
    km: float = 1.8
    terrain_db: float = 0.0
    trials: int = 30
    seed: int = 0
    knockout: str | None = None
    knockout_at: float = 15.0
    deadline_s: float = 3600.0     # not delivered by then = failed (sim seconds)

    def sim_config(self, seed: int) -> SimConfig:
        return SimConfig(radio=self.radio, loss=self.loss, rate_bps=self.rate_bps, lora_preset=self.lora_preset,
                         km_per_unit=self.km, terrain_db=self.terrain_db, seed=seed)


def run_trial(cfg: LabConfig, variant: str, seed: int) -> dict:
    p = payloads()[variant]
    sim = MeshSim(demo_topology(), cfg.sim_config(seed))
    msg = sim.send_message("Hiker", "Ranger", p["parts"])
    tr, led = StatusTracker(msg, sim.cfg.rate_bps), NetworkLedger(msg)
    for ev in sim.events:
        tr.on_event(ev)
        led.on_event(ev)
    sim.listeners += [tr.on_event, led.on_event]
    if cfg.knockout:
        sim.run_until(cfg.knockout_at)
        sim.set_alive(cfg.knockout, False)
    sim.run(max_t=cfg.deadline_s)

    # progress curve: share of data chunks present at the ranger over time
    have, curve = set(), [(0.0, 0.0)]
    t_first = t_all = None
    for e in sim.events:
        if e.msg_id != msg:
            continue
        if (e.kind == "received" and not e.parity) or e.kind == "parity_recovered":
            if (e.part, e.index) not in have:
                have.add((e.part, e.index))
                curve.append((round(e.t, 2), len(have) / p["data_packets"]))
                t_first = t_first if t_first is not None else e.t
        if e.kind == "message_complete":
            t_all = e.t
    delivered = sim.tx[msg].state == "delivered"
    got = sim.rx_parts.get(msg, {})
    intact = all(got.get(ContentType(ct).name.lower()) == b for ct, b in p["parts"].items())
    snap, tot = tr.snapshot(), led.totals()
    useful = p["bytes"]
    return {
        "variant": variant, "seed": seed,
        "delivered": delivered, "intact": delivered and intact,
        "state": snap["state"] if snap["state"] in ("delivered", "failed") else "timed out",
        "t_first": None if t_first is None else round(t_first, 2),
        "t_arrive": None if t_all is None else round(t_all, 2),
        "t_confirm": tr.t_done if delivered else None,
        "received_pct": round(100 * len(have) / p["data_packets"], 1),
        "frames": tot["frames_on_air"], "frames_lost": tot["frames_lost"],
        "frame_loss_pct": tot["frame_loss_pct"],
        "acks": tot["acks_sent"], "acks_lost": tot["acks_lost"],
        "resends": snap["resends"], "parity": snap["parity_recoveries"], "nacks": snap["nacks"],
        "reroutes": snap["reroutes"],
        "air_bytes": snap["bytes_on_air"], "useful_bytes": useful,
        "overhead_x": round(snap["bytes_on_air"] / useful, 1) if useful else None,
        "goodput_bps": round(useful * 8 / t_all, 1) if t_all else 0.0,
        "lost_by": {k: v for k, v in tot["lost_by_reason"].items() if v},
        "curve": curve,
    }


def _q(xs: list, q: float):
    if not xs:
        return None
    return round(float(np.quantile(np.array(xs, dtype=float), q)), 2)


def summarize_variant(rows: list[dict], variant: str, grid: list[float]) -> dict:
    v = VARIANT_BY_KEY[variant]
    p = payloads()[variant]
    ok = [r for r in rows if r["delivered"]]
    arrive = [r["t_arrive"] for r in rows if r["t_arrive"] is not None]
    causes: dict[str, float] = {}
    for r in rows:
        for k, n in r["lost_by"].items():
            causes[k] = causes.get(k, 0) + n / len(rows)
    curve = []  # mean share of content at the ranger at each grid time (step function per trial)
    for t in grid:
        vals = []
        for r in rows:
            frac = 0.0
            for tt, f in r["curve"]:
                if tt <= t:
                    frac = f
                else:
                    break
            vals.append(frac)
        curve.append(round(100 * sum(vals) / len(vals), 2))
    return {
        "key": variant, "label": v.label, "kind": v.kind, "note": v.note,
        "bytes": p["bytes"], "packets": p["packets"], "trials": len(rows),
        "delivery_pct": round(100 * len(ok) / len(rows), 1),
        "intact_pct": round(100 * sum(r["intact"] for r in rows) / len(rows), 1),
        "t_arrive": {"p10": _q(arrive, .1), "p50": _q(arrive, .5), "p90": _q(arrive, .9),
                     "min": _q(arrive, 0), "max": _q(arrive, 1)},
        "t_first_p50": _q([r["t_first"] for r in rows if r["t_first"] is not None], .5),
        "t_confirm_p50": _q([r["t_confirm"] for r in ok], .5),
        "frame_loss_pct": round(statistics.mean(r["frame_loss_pct"] for r in rows), 1),
        "frames": round(statistics.mean(r["frames"] for r in rows), 1),
        "frames_lost": round(statistics.mean(r["frames_lost"] for r in rows), 1),
        "resends": round(statistics.mean(r["resends"] for r in rows), 1),
        "parity": round(statistics.mean(r["parity"] for r in rows), 2),
        "nacks": round(statistics.mean(r["nacks"] for r in rows), 2),
        "air_bytes": round(statistics.mean(r["air_bytes"] for r in rows)),
        "overhead_x": round(statistics.mean(r["air_bytes"] for r in rows) / p["bytes"], 1),
        "goodput_bps": _q([r["goodput_bps"] for r in ok], .5),
        "received_pct": round(statistics.mean(r["received_pct"] for r in rows), 1),
        "lost_by": {k: round(n, 2) for k, n in sorted(causes.items(), key=lambda kv: -kv[1])},
        "curve": curve,
    }


def run_experiment(cfg: LabConfig, progress=None) -> dict:
    rows = []
    total = len(cfg.variants) * cfg.trials
    for v in cfg.variants:
        for i in range(cfg.trials):
            rows.append(run_trial(cfg, v, cfg.seed + i))
            if progress:
                progress(len(rows) / total)
    t_max = max([r["t_arrive"] or 0 for r in rows] + [1.0])
    grid = [round(t_max * k / 80, 2) for k in range(81)]
    summary = [summarize_variant([r for r in rows if r["variant"] == v], v, grid) for v in cfg.variants]
    for r in rows:
        r.pop("curve")
    return {"config": cfg.__dict__ | {"variants": list(cfg.variants)}, "grid": grid, "variants": summary,
            "trials": rows}


SWEEPS = {
    "loss": {"label": "Packet loss per link", "unit": "%", "radio": "simple",
             "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
    "km": {"label": "Hop distance (LoRa)", "unit": "km", "radio": "lora",
           "values": [0.8, 1.2, 1.6, 2.0, 2.4]},
    "terrain_db": {"label": "Extra terrain loss (LoRa)", "unit": "dB", "radio": "lora",
                   "values": [0, 3, 6, 9, 12]},
}


def run_sweep(cfg: LabConfig, param: str, trials: int = 12, progress=None) -> dict:
    sw = SWEEPS[param]
    points = []
    n = len(sw["values"])
    for i, x in enumerate(sw["values"]):
        c = LabConfig(**{**cfg.__dict__, "trials": trials, "radio": sw["radio"], param: x})
        res = run_experiment(c, (lambda f, i=i: progress((i + f) / n)) if progress else None)
        points.append({"x": x * 100 if param == "loss" else x,
                       "variants": [{k: s[k] for k in ("key", "delivery_pct", "frame_loss_pct", "overhead_x")}
                                    | {"t_p50": s["t_arrive"]["p50"], "t_p90": s["t_arrive"]["p90"]}
                                    for s in res["variants"]]})
    return {"param": param, "label": sw["label"], "unit": sw["unit"], "radio": sw["radio"], "points": points,
            "variants": list(cfg.variants)}


def variant_catalog() -> list[dict]:
    p = payloads()
    return [{"key": v.key, "label": v.label, "kind": v.kind, "note": v.note, "bytes": p[v.key]["bytes"],
             "packets": p[v.key]["packets"]} for v in VARIANTS]


