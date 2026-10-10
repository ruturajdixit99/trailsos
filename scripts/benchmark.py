"""Runs the full SOS over the 4-hop demo mesh at 0/10/20/30% loss on both rate
presets and writes BENCHMARK.md ("media per byte" evidence)."""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.provider import MockProvider  # noqa: E402
from mesh.packet import ContentType  # noqa: E402
from mesh.sim import RATE_PRESETS, MeshSim, SimConfig, demo_topology  # noqa: E402
from mesh.status import StatusTracker  # noqa: E402
from server.sos import build_sos  # noqa: E402

TRIALS = 100
LOSSES = (0.0, 0.10, 0.20, 0.30)
LORA_TRIALS = 40


def trial(parts, loss, rate, seed, **lora_kw):
    sim = MeshSim(demo_topology(), SimConfig(rate_bps=rate, loss=loss, seed=seed, **lora_kw))
    mid = sim.send_message("Hiker", "Ranger", parts)
    tr = StatusTracker(mid, rate)
    for ev in sim.events:
        tr.on_event(ev)
    sim.listeners.append(tr.on_event)
    sim.run()
    got = sim.rx_parts.get(mid, {})
    identical = all(got.get(ContentType(ct).name.lower()) == b for ct, b in parts.items())
    return tr, identical


def main() -> None:
    payload = build_sos(MockProvider())
    parts = payload.parts
    useful = sum(len(b) for b in parts.values())
    voice_s = payload.voice_seconds
    rows = []
    t0 = time.time()
    for preset, rate in RATE_PRESETS.items():
        for loss in LOSSES:
            res = [trial(parts, loss, rate, s) for s in range(TRIALS)]
            ok = [tr for tr, same in res if tr.state == "delivered" and same]
            air = statistics.mean(tr.bytes_on_air for tr, _ in res)
            dr = len(ok) / len(res)
            times = [tr.t_done - tr.t_start for tr in ok]
            rows.append({
                "preset": f"{preset} ({rate:,.0f} bps)", "loss": f"{loss:.0%}",
                "air": f"{air:,.0f}", "useful": f"{useful:,}",
                "eff": f"{useful / air:.0%}",
                "bpsv": f"{air / (voice_s * dr):,.0f}" if dr else "n/a",
                "resends": f"{statistics.mean(tr.resends for tr, _ in res):.1f}",
                "parity": f"{statistics.mean(tr.parity_recoveries for tr, _ in res):.2f}",
                "nacks": f"{statistics.mean(tr.nacks for tr, _ in res):.2f}",
                "dr": f"{dr:.1%}",
                "med": f"{statistics.median(times):.1f}" if times else "n/a",
            })
            print(rows[-1])
    hdr = ("| Rate preset | Loss/link | Bytes on air | Useful media bytes | Useful/air | "
           "Air bytes per voice-second delivered | Resends | Parity recoveries | NACKs | "
           "Delivery rate (byte-identical, e2e-acked) | Median delivery time (s) |")
    sep = "|" + "---|" * 11
    lines = [hdr, sep] + [
        f"| {r['preset']} | {r['loss']} | {r['air']} | {r['useful']} | {r['eff']} | {r['bpsv']} | {r['resends']} | "
        f"{r['parity']} | {r['nacks']} | {r['dr']} | {r['med']} |" for r in rows]
    sizes = ", ".join(f"{k} {v} B" for k, v in payload.sizes().items())
    lora_lines = ["| Preset | Hops | Scenario | Delivery rate | Median delivery (s) | Frames on air | Frame loss | "
                  "Loss causes per run (weak / collision / half-duplex / node down) |", "|" + "---|" * 8]
    for preset, km in (("long_fast", 1.8), ("long_fast", 2.2), ("medium_fast", 1.2), ("short_fast", 0.8)):
        for ko in (None, "C"):
            res = []
            for s in range(LORA_TRIALS):
                sim = MeshSim(demo_topology(), SimConfig(radio="lora", lora_preset=preset, km_per_unit=km, seed=s))
                mid = sim.send_message("Hiker", "Ranger", parts)
                if ko:
                    sim.run_until(15.0)
                    sim.set_alive(ko, False)
                sim.run()
                res.append((sim, mid))
            ok = [(sm, m) for sm, m in res if sm.tx[m].state == "delivered"]
            times = [next(e.t for e in sm.events if e.kind == "e2e_ack_received") for sm, m in ok]
            frames = statistics.mean(sum(1 for e in sm.events if e.kind in ("sent", "retried")) for sm, _ in res)
            lost = [e for sm, _ in res for e in sm.events if e.kind == "lost" and e.part != "ack"]
            by = {r: sum(1 for e in lost if e.detail["reason"] == r) / len(res)
                  for r in ("weak signal", "collision", "half-duplex", "receiver down")}
            name = sim.preset.name
            lora_lines.append(
                f"| {name} ({sim.preset.bitrate:,.0f} bps) | {km} km | {'C down' if ko else 'normal'} | "
                f"{len(ok) / len(res):.0%} | {statistics.median(times):.0f} | {frames:.0f} | "
                f"{len(lost) / len(res) / frames:.0%} | "
                f"{by['weak signal']:.0f} / {by['collision']:.0f} / {by['half-duplex']:.0f} / {by['receiver down']:.0f} |"
                if times else f"| {name} | {km} km | {'C down' if ko else 'normal'} | 0% | n/a | {frames:.0f} | n/a | n/a |")
            print(lora_lines[-1])
    md = f"""# Trail SOS benchmark

Full SOS (5 s voice + photo) from Hiker to Ranger over the 4-hop demo mesh
(Hiker -> B -> C -> D -> Ranger, alternate relay E), {TRIALS} seeded trials per row.

* Payload: {sizes} = **{useful:,} useful bytes** ({payload.codec}, thumbnail {payload.thumb_format}).
* Voice alone: {len(parts[ContentType.AUDIO])} B for {voice_s:.0f} s = **{len(parts[ContentType.AUDIO]) / voice_s:.0f} B per second of voice** before networking overhead.
* "Bytes on air" counts every transmission on every hop: data, parity, retries, hop ACKs, NACKs and e2e-acks.
* "Air bytes per voice-second delivered" = mean bytes on air / (5 s x delivery rate), i.e. the network cost of each second of voice that actually arrived.
* Simple model (this table): fixed loss per frame, airtime = bytes x 8 / rate, no collisions. Times are simulated seconds.

{chr(10).join(lines)}

## Realistic LoRa radio model ({LORA_TRIALS} seeded trials per row)

Same SOS, but frames are lost for physical reasons: Semtech airtime per Meshtastic preset, log-distance path
loss (exponent 3.6, forested mountains), 3 dB per-link shadowing, 3 dB per-packet fading, demodulation SNR
floor per spreading factor, collisions (6 dB capture), half-duplex radios and listen-before-talk.
Hops are 1.8 km unless stated; "C down" knocks relay C out 15 s into the send.

{chr(10).join(lora_lines)}

Generated by `python scripts/benchmark.py` in {time.time() - t0:.0f} s.
"""
    (ROOT / "BENCHMARK.md").write_text(md, encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
