"""Detailed single-run simulation report: what was sent, what was lost (and why), what arrived.

    python scripts/simulate.py                                  # simple model, 20% loss
    python scripts/simulate.py --radio lora                     # LoRa physics (LongFast)
    python scripts/simulate.py --radio lora --preset medium_fast --km 1.5 --knockout C --trace
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.provider import MockProvider  # noqa: E402
from mesh.ledger import NetworkLedger  # noqa: E402
from mesh.lora import PRESETS  # noqa: E402
from mesh.sim import RATE_PRESETS, MeshSim, SimConfig, demo_topology  # noqa: E402
from mesh.status import StatusTracker  # noqa: E402
from server.sos import build_sos  # noqa: E402

SKIP_IN_TRACE = {"ack_sent", "acked", "backoff"}


def table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    rows = [{k: str(v).replace("→", "->") for k, v in r.items()} for r in rows]  # Windows consoles: ASCII
    widths = [max(len(h), *(len(str(r.get(k, "")).replace("2192", "->")) for r in rows)) for k, h in cols] if rows else [len(h) for _, h in cols]
    line = "  ".join(h.ljust(w) for (_, h), w in zip(cols, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append("  ".join(str(r.get(k, "")).replace("2192", "->").ljust(w) for (k, _), w in zip(cols, widths)))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radio", choices=["simple", "lora"], default="simple")
    ap.add_argument("--preset", choices=list(PRESETS), default="long_fast", help="LoRa modem preset (lora mode)")
    ap.add_argument("--rate", choices=list(RATE_PRESETS), default="long_range", help="bit rate (simple mode)")
    ap.add_argument("--loss", type=float, default=0.20, help="per-link loss (simple mode)")
    ap.add_argument("--km", type=float, default=1.8, help="km per map unit = hop length (lora mode)")
    ap.add_argument("--terrain", type=float, default=0.0, help="extra terrain loss in dB (lora mode)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--knockout", help="node to knock out mid-send (B, C, D or E)")
    ap.add_argument("--knockout-at", type=float, default=15.0, help="sim seconds")
    ap.add_argument("--trace", action="store_true", help="print every frame event")
    a = ap.parse_args()

    cfg = SimConfig(radio=a.radio, lora_preset=a.preset, rate_bps=RATE_PRESETS[a.rate], loss=a.loss,
                    km_per_unit=a.km, terrain_db=a.terrain, seed=a.seed)
    sim = MeshSim(demo_topology(), cfg)
    payload = build_sos(MockProvider())
    msg = sim.send_message("Hiker", "Ranger", payload.parts)
    tracker, ledger = StatusTracker(msg, sim.cfg.rate_bps), NetworkLedger(msg)
    for ev in sim.events:
        tracker.on_event(ev)
        ledger.on_event(ev)
    sim.listeners += [tracker.on_event, ledger.on_event]
    if a.knockout:
        sim.run_until(a.knockout_at)
        sim.set_alive(a.knockout, False)
    sim.run()

    print("=" * 100)
    if sim.lora:
        p = sim.preset
        print(f"RADIO  LoRa {p.name}: SF{p.sf}, BW {p.bw_hz / 1e3:.0f} kHz, CR 4/{4 + p.cr}, {p.bitrate:,.0f} bps, "
              f"floor {p.snr_floor_db} dB SNR | hop {a.km} km, terrain +{a.terrain} dB | seed {a.seed}")
        print(f"       airtime: 219 B frame = {sim.airtime(219) * 1000:.0f} ms, hop ACK = {sim.airtime(20) * 1000:.0f} ms")
    else:
        print(f"RADIO  simple: {sim.cfg.rate_bps:,.0f} bps, {a.loss:.0%} loss per frame per link | seed {a.seed}")
    print(f"PAYLOAD {', '.join(f'{k} {v} B' for k, v in payload.sizes().items())}")
    print(f"ROUTE  planned {' > '.join(sim.planned_route('Hiker', 'Ranger'))}"
          + (f" | {a.knockout} knocked out at t={a.knockout_at}s" if a.knockout else ""))

    print("\nLINK BUDGET (mean, before per-packet fading)")
    cols = [("a", "from"), ("b", "to"), ("km", "km")]
    if sim.lora:
        cols += [("rssi_dbm", "RSSI dBm"), ("snr_db", "SNR dB"), ("margin_db", "margin dB"), ("per", "frame loss"),
                 ("etx", "ETX")]
    else:
        cols += [("per", "frame loss")]
    print(table(sim.link_info(), cols))

    if a.trace:
        print("\nFRAME TRACE (hop ACKs and backoffs hidden)")
        for e in sim.events:
            if e.msg_id != msg and e.kind not in ("node_down", "node_up") or e.kind in SKIP_IN_TRACE:
                continue
            what = f"{e.part or ''}{'-parity' if e.parity else ''}{'#' + str(e.index) if e.index is not None else ''}"
            rf = f" rssi {e.detail['rssi_dbm']} snr {e.detail['snr_db']}" if "snr_db" in e.detail else ""
            why = e.detail.get("reason", "")
            arrow = f"{e.node}->{e.peer}" if e.peer else e.node
            print(f"{e.t:8.2f}s  {e.kind:17s} {arrow:14s} {what:20s} {why}{rf}")

    snap = ledger.snapshot()
    t = snap["totals"]
    print("\nSENT vs RECEIVED, PER LINK (data/parity/control frames; ACKs counted separately)")
    print(table(snap["links"], [("link", "link"), ("frames", "sent"), ("delivered", "received"), ("lost", "lost"),
                                ("loss_pct", "loss %"), ("lost_random", "fade/rand"), ("lost_crc", "crc"),
                                ("lost_collision", "collision"), ("lost_half_duplex", "half-dup"),
                                ("lost_node_down", "node down"), ("acks_sent", "acks"), ("acks_lost", "acks lost")]))

    print("\nPER CHUNK")
    rows = []
    for c in snap["chunks"]:
        hops = " ".join(f"{h['from'][0]}>{h['to'][0]}:{h['tries']}{'ok' if h['ok'] else 'X'}" for h in c["hops"])
        rows.append({**c, "hops": hops, "arrived": c["arrived_t"] if c["arrived_t"] is not None else "-"})
    print(table(rows, [("label", "chunk"), ("size", "bytes"), ("source_sends", "src sends"), ("frames", "frames"),
                       ("lost", "lost"), ("status", "status"), ("arrived", "arrived t"), ("hops", "hop tries (last 8)")]))

    print("\nTOTALS")
    print(f"  Hiker transmitted   {t['source_frames']} frames, {t['source_bytes']:,} B  ({t['source_chunks']} distinct media chunks)")
    print(f"  Ranger received     {t['dest_unique_chunks']} unique chunks, {t['dest_bytes']:,} B, "
          f"{t['dest_duplicates']} duplicates discarded")
    print(f"  On air (all hops)   {t['frames_on_air']} frames: {t['frames_delivered']} delivered, {t['frames_lost']} lost "
          f"({t['frame_loss_pct']}%); {t['acks_sent']} hop ACKs, {t['acks_lost']} lost")
    print(f"  Loss causes         {', '.join(f'{k} {v}' for k, v in t['lost_by_reason'].items() if v) or 'none'}")
    print(f"  Recovery            {t['chunks_rebuilt_by_parity']} chunks rebuilt by parity, "
          f"{len(snap['nacks'])} NACKs (last 5 shown in JSON), {tracker.reroutes} reroutes")
    s = tracker.snapshot()
    print(f"\nRESULT  {s['label'].upper()}  in {s['elapsed_s']} s (sim) | route {' > '.join(s['route'])} | "
          f"e2e-ack received: {s['e2e_acked']}")
    print("=" * 100)


if __name__ == "__main__":
    main()
