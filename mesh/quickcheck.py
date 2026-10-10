"""Quick check: runs the CHRP track's requirements as real tests in ~15 s and reports pass/fail.

Uses the mock AI provider so results are repeatable and free; everything else (codecs,
packets, simulator, ARQ, parity, routing, status) is the real code.
"""
from __future__ import annotations

import statistics
import time

from ai.provider import MockProvider
from media import chunking
from mesh import lora
from mesh.packet import MAX_PACKET, ContentType, Packet
from mesh.sim import MeshSim, SimConfig, demo_topology, three_hop_topology
from mesh.status import StatusTracker
from server.sos import KINDS, ROOT, build_message


def _trial(parts, topo=None, seed=0, loss=0.2, knockout=None, dead=(), radio="simple"):
    sim = MeshSim(topo or demo_topology(), SimConfig(loss=loss, seed=seed, radio=radio))
    for n in dead:
        sim.set_alive(n, False)
    mid = sim.send_message("Hiker", "Ranger", parts)
    tr = StatusTracker(mid, sim.cfg.rate_bps)
    for ev in sim.events:
        tr.on_event(ev)
    sim.listeners.append(tr.on_event)
    if knockout:
        sim.run_until(10.0)
        sim.set_alive(knockout, False)
    sim.run(max_t=20000)
    got = sim.rx_parts.get(mid, {})
    intact = all(got.get(ContentType(ct).name.lower()) == b for ct, b in parts.items())
    acked = any(e.kind == "e2e_ack_received" and e.msg_id == mid for e in sim.events)
    paths = [e.detail.get("path", []) for e in sim.events if e.kind == "received" and e.msg_id == mid]
    return {"sim": sim, "tracker": tr, "intact": intact, "acked": acked, "paths": paths, "mid": mid}


def run_quickcheck(progress=None) -> dict:
    t0 = time.time()
    checks = []
    steps = 9

    def add(cid, title, requirement, passed, value, detail):
        checks.append({"id": cid, "title": title, "requirement": requirement, "passed": bool(passed),
                       "value": value, "detail": detail})
        if progress:
            progress(len(checks) / steps, title)

    mock = MockProvider()
    payloads = {k: build_message(k, mock) for k in KINDS}
    sos = payloads["sos"].parts

    # 1. mesh
    sim = MeshSim()
    route = sim.planned_route("Hiker", "Ranger")
    alt = sim.shortest_path(0, sim.by_name["Ranger"].id, {sim.by_name["C"].id})
    add("mesh", "Simulated mesh of 5+ nodes", "5 or more virtual nodes on a map, each relaying packets",
        len(sim.nodes) >= 5 and len(route) - 1 >= 3 and alt is not None,
        f"{len(sim.nodes)} phones, {len(sim.links())} links",
        f"route {' > '.join(route)} ({len(route) - 1} hops); alternate path via E exists: {alt is not None}")

    # 2. packet size
    biggest, count = 0, 0
    for p in payloads.values():
        for ct, data in p.parts.items():
            for i, ch in enumerate(chunking.split(data)):
                biggest = max(biggest, len(Packet(1, i, 1, int(ct), 0, 4, ch).encode()))
                count += 1
    add("packets", "Packets fit in a LoRa frame", f"each packet about 230 bytes (limit {MAX_PACKET})",
        biggest <= MAX_PACKET, f"largest {biggest} B", f"{count} data packets across text, voice, photo, video, SOS")

    # 3. media squeeze
    s = {k: payloads[k].sizes() for k in KINDS}
    voice_b, img_b, vid_b = s["voice"]["audio"], s["photo"]["thumbnail"], s["video"]["video"]
    orig_wav = (ROOT / "samples" / "sos_voice.wav").stat().st_size
    orig_jpg = (ROOT / "samples" / "trail_photo.jpg").stat().st_size
    orig_mp4 = (ROOT / "samples" / "trail_clip.mp4").stat().st_size
    add("squeeze", "Voice, image and video squeezed", "compress voice and video aggressively; tiny thumbnails, "
        "keyframes, AI description", voice_b <= 800 and img_b <= 800 and vid_b <= 1600,
        f"voice {voice_b} B · image {img_b} B · video {vid_b} B",
        f"5 s voice {orig_wav:,} B WAV -> Codec2 {voice_b} B ({orig_wav // voice_b}x smaller); photo {orig_jpg:,} B "
        f"-> {img_b} B thumbnail + {s['photo']['caption']} B AI caption; 5 s video {orig_mp4:,} B MP4 -> "
        f"{payloads['video'].video['frames_sent']} keyframes {vid_b} B ({orig_mp4 // vid_b}x smaller) + AI description")

    # 4. survive 20% loss over 3 hops
    N = 100
    runs = [_trial(sos, three_hop_topology(), seed=i) for i in range(N)]
    ok = sum(r["intact"] and r["acked"] for r in runs)
    times = [r["tracker"].t_done - r["tracker"].t_start for r in runs if r["acked"]]
    add("loss", "Arrives whole at 20% loss over 3 hops", "a message arrives whole even with 20% packet loss "
        "across 3 hops", ok / N >= 0.99, f"{ok}/{N} byte-identical",
        f"voice + image SOS, {N} seeded runs; median {statistics.median(times):.0f} s at 1,000 bps; "
        f"avg {statistics.mean(r['tracker'].resends for r in runs):.1f} resends")

    # 5. honesty
    fails = [_trial(sos, seed=i, dead=("D",)) for i in range(15)]
    lies = sum((r["tracker"].state == "delivered") != r["acked"] for r in runs + fails)
    failed_ok = sum(r["tracker"].state == "failed" and bool(r["tracker"].reason) for r in fails)
    add("honest", "Never says delivered when it wasn't", "honest status: never says delivered when it wasn't",
        lies == 0 and failed_ok == len(fails), f"{lies} false 'delivered'",
        f"{len(runs) + len(fails)} runs: 'delivered' only with an end-to-end ack; {failed_ok}/{len(fails)} runs "
        f"with relay D dead ended 'failed' with a reason")

    # 6. reroute
    kos = [_trial(sos, seed=i, knockout="C") for i in range(20)]
    via_e = sum(r["intact"] and r["acked"] and any("E" in p for p in r["paths"]) for r in kos)
    add("reroute", "Reroutes around a failed relay", "relaying with resends and recovery when a node drops",
        via_e >= 18, f"{via_e}/20 delivered via E", "relay C knocked out 10 s into the send")

    # 7. many devices at once
    pairs = [("Hiker", "Ranger", "sos"), ("B", "D", "text"), ("E", "Hiker", "photo"), ("D", "B", "voice"),
             ("Ranger", "C", "text"), ("C", "E", "video")]
    all_ok, total = 0, 0
    for seed in range(8):
        sim = MeshSim(demo_topology(), SimConfig(loss=0.2, seed=seed))
        ids = [(sim.send_message(a, b, payloads[k].parts), payloads[k].parts) for a, b, k in pairs]
        sim.run(max_t=20000)
        for mid, parts in ids:
            total += 1
            got = sim.rx_parts.get(mid, {})
            all_ok += sim.tx[mid].state == "delivered" and all(
                got.get(ContentType(ct).name.lower()) == b for ct, b in parts.items())
    add("multi", "6 phones sending at once", "a mesh of devices relaying each other's messages",
        all_ok == total, f"{all_ok}/{total} messages intact",
        "6 phones send text, voice, photo, video and an SOS in different directions at the same time, "
        "20% loss, 8 seeded runs")

    # 8. realistic LoRa
    semtech = lora.airtime_s(lora.LoRaPreset("x", 7, 125e3, 1, 8), 20)
    lf = [_trial(sos, seed=i, radio="lora") for i in range(10)]
    lf_ok = sum(r["intact"] and r["acked"] for r in lf)
    add("lora", "Realistic LoRa data rates and range", "realistic data rates, packet loss and range",
        abs(semtech - 0.05658) < 1e-4 and lf_ok >= 9,
        f"LongFast {lora.PRESETS['long_fast'].bitrate:,.0f} bps · {lf_ok}/10 delivered",
        "airtime matches Semtech's calculator; loss from SNR vs range, fading, collisions and half-duplex radios; "
        "SOS over 1.8 km hops")

    # 9. media per byte
    useful = sum(len(b) for b in sos.values())
    air = statistics.mean(r["tracker"].bytes_on_air for r in runs)
    add("perbyte", "Media per byte", "how much media you get through per byte",
        True, f"{useful:,} useful B per SOS",
        f"5 s of voice + a photo + transcript + caption in {useful:,} B; {air:,.0f} B on air across 3 hops at 20% loss "
        f"({air / useful:.1f}x, including every retry and ACK)")

    return {"checks": checks, "passed": sum(c["passed"] for c in checks), "total": len(checks),
            "seconds": round(time.time() - t0, 1), "ai": "mock provider (repeatable, free)"}
