"""Plain-language run summary, built from the status tracker, the ledger and raw events.

Like the rest of the status code it only reports what events prove: "delivered"
appears only when the tracker saw the e2e-ack.
"""
from __future__ import annotations

from mesh.ledger import NetworkLedger
from mesh.status import StatusTracker

CAUSE_LABELS = {"loss": "random loss", "crc_fail": "corrupted (CRC)", "weak signal": "weak signal / fade",
                "collision": "collision", "half-duplex": "half-duplex (receiver busy transmitting)",
                "receiver down": "node down", "sender down": "sender down", "out of range": "out of range"}
ROLES = {"Hiker": "sender", "Ranger": "destination"}


def _fmt_t(t) -> str:
    return "–" if t is None else f"{t:.1f} s"


def phones(sim, ledger: NetworkLedger) -> list[dict]:
    """Per-phone stats for every node in the mesh."""
    out = []
    for n in sim.nodes:
        st = ledger.nodes.get(n.name, {})
        out.append({"name": n.name, "role": ROLES.get(n.name, "relay"), "alive": n.alive,
                    "queue": len(n.queue) + (1 if n.busy else 0),
                    "tx": st.get("tx", 0), "rx": st.get("rx", 0), "relayed": st.get("relayed", 0),
                    "dropped": st.get("dropped", 0), "dup": st.get("dup", 0), "acks_tx": st.get("acks_tx", 0)})
    return out


def summarize(sim, tracker: StatusTracker, ledger: NetworkLedger, payload_sizes: dict, radio: dict) -> dict:
    s = tracker.snapshot()
    led = ledger.snapshot()
    t = led["totals"]
    msg = tracker.msg_id
    ev = [e for e in sim.events if e.msg_id in (msg, None)]

    # ---- verdict / headline
    if s["state"] == "delivered":
        verdict = "delivered"
        headline = (f"DELIVERED ✓✓ in {s['elapsed_s']} s (sim). {ledger.dst}'s phone confirmed every byte "
                    f"with an end-to-end ack.")
    elif s["state"] == "failed":
        verdict = "failed"
        headline = f"NOT DELIVERED. {s['reason']}"
    else:
        verdict = "in_progress"
        headline = f"In progress: {s['label']}. ETA about {s['eta_s']} s (sim)."

    bullets: list[str] = []
    n_relays = sum(1 for n in sim.nodes if n.name not in ROLES)
    if radio.get("model") == "lora":
        radio_txt = (f"Realistic LoRa {radio['preset']} (SF{radio['sf']}, {radio['bitrate']:,} bps, "
                     f"{radio['km_per_unit']} km hops, terrain +{radio['terrain_db']} dB)")
    else:
        radio_txt = f"Simple model ({radio.get('bitrate', 0):,.0f} bps, {radio.get('loss', 0):.0%} loss per link)"
    bullets.append(f"Network: {len(sim.nodes)} phones ({', '.join(n.name for n in sim.nodes)}) · this message "
                   f"{ledger.src} → {ledger.dst} · {radio_txt}.")

    useful = sum(payload_sizes.values())
    bullets.append("Payload: " + ", ".join(f"{k} {v} B" for k, v in payload_sizes.items())
                   + f" = {useful:,} useful bytes.")

    # ---- route + disruptions
    planned = next((e.detail.get("route") for e in ev if e.kind == "message_queued"), [])
    used = s["route"] if s["route_is_observed"] else []
    route_txt = f"Planned {' → '.join(planned) or 'none'}"
    if used and used != planned:
        route_txt += f"; final chunks actually went {' → '.join(used)}"
    elif used:
        route_txt += " (used as planned)"
    bullets.append(route_txt + ".")
    for e in ev:
        if e.kind == "node_down":
            reroutes = [r for r in ev if r.kind == "rerouted" and r.t >= e.t]
            follow = (f"; first reroute {reroutes[0].t - e.t:.1f} s later at {reroutes[0].node} via "
                      f"{' → '.join(reroutes[0].detail.get('via', []))}") if reroutes else ""
            bullets.append(f"{e.node} was knocked out at {e.t:.1f} s and lost "
                           f"{e.detail.get('queued_packets_lost', 0)} queued packets{follow}.")
        elif e.kind == "node_up":
            bullets.append(f"{e.node} came back online at {e.t:.1f} s.")

    # ---- arrival order
    arrivals = {e.part: e.t for e in ev if e.kind == "reassembled"}
    if arrivals:
        bullets.append(f"Arrival at {ledger.dst}: " + ", ".join(f"{p} {_fmt_t(arrivals[p])}" for p in
                                                         ("transcript", "caption", "audio", "thumbnail", "video")
                                                         if p in arrivals) + ".")
        if "transcript" in arrivals and s["state"] == "delivered":
            bullets.append(f"First useful information (transcript) reached {ledger.dst} after {arrivals['transcript']:.1f} s, "
                           f"{s['elapsed_s'] - arrivals['transcript']:.1f} s before the full message was confirmed.")
    missing = [p for p in ("transcript", "caption", "audio", "thumbnail", "video") if p in s["parts"] and p not in arrivals]
    if missing:
        bullets.append(f"Not (yet) reassembled at {ledger.dst}: " + ", ".join(
            f"{p} {s['parts'][p]['have']}/{s['parts'][p]['of']} chunks" for p in missing) + ".")

    # ---- traffic
    bullets.append(f"Sent vs received: {ledger.src} transmitted {t['source_frames']} frames ({t['source_bytes']:,} B incl. retries); "
                   f"{ledger.dst} received {t['dest_unique_chunks']} unique chunks ({t['dest_bytes']:,} B) and discarded "
                   f"{t['dest_duplicates']} duplicates.")
    bullets.append(f"On air across all hops: {t['frames_on_air']} frames, {t['frames_delivered']} delivered, "
                   f"{t['frames_lost']} lost ({t['frame_loss_pct']}%); {t['acks_sent']} hop ACKs ({t['acks_lost']} lost).")
    causes = sorted(((k, v) for k, v in t["lost_by_reason"].items() if v), key=lambda kv: -kv[1])
    if causes:
        bullets.append("Why frames were lost: " + ", ".join(f"{CAUSE_LABELS.get(k, k)} {v}" for k, v in causes) + ".")
    busy = [l for l in led["links"] if l["frames"] >= 3]
    if busy:
        worst = max(busy, key=lambda l: (l["loss_pct"], l["frames"]))
        best = max(busy, key=lambda l: (l["delivered"], -l["loss_pct"]))
        bullets.append(f"Busiest link {best['link']} ({best['frames']} frames, {best['loss_pct']}% lost); "
                       f"worst link {worst['link']} ({worst['lost']}/{worst['frames']} lost, {worst['loss_pct']}%).")

    # ---- recovery + efficiency
    bullets.append(f"Recovery: {s['resends']} hop retransmissions, {s['reroutes']} reroutes, {s['nacks']} NACKs, "
                   f"{s['parity_recoveries']} chunks rebuilt from parity, {t['chunks_dropped_somewhere']} chunk copies "
                   f"dropped en route.")
    if s["bytes_on_air"]:
        bullets.append(f"Efficiency: {useful:,} useful bytes cost {s['bytes_on_air']:,} bytes on air "
                       f"({100 * useful / s['bytes_on_air']:.0f}% useful); observed frame loss "
                       f"{s['observed_loss'] * 100:.0f}%, effective throughput {s['effective_rate_bps']:,.0f} bps.")
    ai = []
    for e in ev:
        if e.kind == "e2e_ack_received":
            ai.append(f"End-to-end ack (CRC32 {e.detail.get('crc32')}) reached {ledger.src} at {e.t:.1f} s.")
            break
    bullets += ai

    md = f"**{headline}**\n\n" + "\n".join(f"- {b}" for b in bullets)
    return {"verdict": verdict, "headline": headline, "bullets": bullets, "markdown": md,
            "phones": phones(sim, ledger)}
