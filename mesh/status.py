"""Honest per-message status, derived ONLY from simulator events.

States: queued -> in_transit (hop n of m) -> partial -> delivered | failed
* `delivered` is set by exactly one event: e2e_ack_received (the source got the
  destination's end-to-end ack, whose CRC matched the whole message).
* `partial` lists only parts that emitted a `reassembled` event.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mesh.packet import HEADER_SIZE, airtime_s
from mesh.sim import ACK_SIZE, Event

PART_ORDER = ("transcript", "caption", "audio", "thumbnail", "video")


@dataclass
class StatusTracker:
    msg_id: int
    rate_bps: float
    ack_margin_s: float = 0.25
    state: str = "queued"
    reason: str = ""
    route_planned: list = field(default_factory=list)
    route_taken: list = field(default_factory=list)
    hop: int = 0
    part_chunks: dict = field(default_factory=dict)    # part -> {"sizes": [...], "have": set()}
    reassembled: dict = field(default_factory=dict)    # part -> bytes length
    sent: int = 0
    resends: int = 0
    lost: int = 0
    tx_frames: int = 0
    bytes_on_air: int = 0
    parity_recoveries: int = 0
    reroutes: int = 0
    nacks: int = 0
    e2e_acked: bool = False
    all_parts_at_dest: bool = False
    eta_s: float | None = None
    t_last: float = 0.0
    t_start: float | None = None
    t_done: float | None = None
    eta_history: list = field(default_factory=list)

    # ------------------------------------------------------------------ events
    def on_event(self, ev: Event) -> bool:
        """Apply one event. Returns True if it concerned this message."""
        if ev.msg_id != self.msg_id:
            if ev.kind in ("node_down", "node_up"):
                self._recompute(ev.t)
            return False
        k = ev.kind
        self.t_last = ev.t
        if k == "message_queued":
            self.route_planned = list(ev.detail.get("route", []))
            for part, sizes in ev.detail.get("chunk_sizes", {}).items():
                self.part_chunks[part] = {"sizes": sizes, "have": set()}
        elif k in ("sent", "retried"):
            self.tx_frames += 1
            self.bytes_on_air += ev.size
            if self.t_start is None:
                self.t_start = ev.t
            if k == "sent":
                self.sent += 1
            else:
                self.resends += 1
            if self.state == "queued":
                self.state = "in_transit"
            self.hop = max(self.hop, ev.detail.get("hop", 1))
        elif k == "ack_sent":
            self.tx_frames += 1
            self.bytes_on_air += ev.size
        elif k == "lost":
            self.lost += 1
        elif k in ("e2e_ack_sent", "nack_sent"):
            self.tx_frames += 1
            self.bytes_on_air += ev.size
            if k == "nack_sent":
                self.nacks += 1
        elif k == "rerouted":
            self.reroutes += 1
            via = ev.detail.get("via")
            if via:
                prefix = self.route_planned[: self.route_planned.index(ev.node)] if ev.node in self.route_planned else []
                self.route_planned = prefix + via
        elif k == "received":
            if not ev.parity and ev.part in self.part_chunks:
                self.part_chunks[ev.part]["have"].add(ev.index)
            path = ev.detail.get("path")
            if path:
                self.route_taken = path
            self.hop = max(self.hop, ev.detail.get("hop", 0))
        elif k == "parity_recovered":
            self.parity_recoveries += 1
            if ev.part in self.part_chunks:
                self.part_chunks[ev.part]["have"].add(ev.index)
        elif k == "reassembled":
            self.reassembled[ev.part] = ev.size
            if ev.part in self.part_chunks:
                self.part_chunks[ev.part]["have"] = set(range(len(self.part_chunks[ev.part]["sizes"])))
            if self.state not in ("delivered", "failed"):
                self.state = "partial"
        elif k == "message_complete":
            self.all_parts_at_dest = True
        elif k == "e2e_ack_received":
            self.e2e_acked = True
            self.state = "delivered"
            self.t_done = ev.t
        elif k == "message_failed":
            if self.state != "delivered":
                self.state = "failed"
                self.reason = ev.detail.get("reason", "unknown")
                self.t_done = ev.t
        self._recompute(ev.t)
        return True

    # ------------------------------------------------------------------ ETA
    def observed_loss(self) -> float:
        # Laplace-smoothed per-frame loss from what we have actually seen
        return (self.lost + 1) / (self.tx_frames + 2) if self.tx_frames else 0.2

    def _t_hop(self, size: int) -> float:
        p = min(self.observed_loss(), 0.95)
        attempt = airtime_s(size, self.rate_bps) + airtime_s(ACK_SIZE, self.rate_bps) + self.ack_margin_s
        return attempt / ((1 - p) ** 2)

    def _hops_total(self) -> int:
        r = self.route_taken if len(self.route_taken) > 1 else self.route_planned
        return max(len(r) - 1, 1)

    def _recompute(self, now: float) -> None:
        if self.state in ("delivered", "failed"):
            self.eta_s = 0.0 if self.state == "delivered" else None
        else:
            missing = [s + HEADER_SIZE for pc in self.part_chunks.values()
                       for i, s in enumerate(pc["sizes"]) if i not in pc["have"]]
            m = self._hops_total()
            if missing:
                eta = sum(self._t_hop(s) for s in missing) + (m - 1) * self._t_hop(max(missing))
            else:
                eta = 0.0
            eta += m * self._t_hop(HEADER_SIZE + 4)  # e2e-ack trip back
            self.eta_s = max(eta, 0.0)
        self.eta_history.append((now, self.eta_s))

    def effective_rate_bps(self) -> float:
        p = min(self.observed_loss(), 0.95)
        return self.rate_bps * (1 - p) ** 2

    # ------------------------------------------------------------------ view
    def parts_view(self) -> dict:
        out = {}
        for part in PART_ORDER:
            if part not in self.part_chunks:
                continue
            pc = self.part_chunks[part]
            out[part] = {"done": part in self.reassembled, "have": len(pc["have"]), "of": len(pc["sizes"])}
        return out

    def label(self) -> str:
        if self.state == "in_transit":
            return f"in_transit (hop {max(self.hop, 1)} of {self._hops_total()})"
        if self.state == "partial":
            bits = []
            for part, v in self.parts_view().items():
                if part in ("transcript", "caption"):
                    if v["done"]:
                        bits.append(part)
                else:
                    bits.append(f"{part} {v['have']}/{v['of']}")
            tail = " - all parts here, awaiting end-to-end ack" if self.all_parts_at_dest else ""
            return "partial: " + ", ".join(bits) + tail
        if self.state == "failed":
            return f"failed: {self.reason}"
        return self.state

    def snapshot(self) -> dict:
        return {
            "msg_id": f"{self.msg_id:08x}",
            "state": self.state,
            "label": self.label(),
            "delivered": self.state == "delivered",
            "e2e_acked": self.e2e_acked,
            "arrived_parts": [p for p in PART_ORDER if p in self.reassembled],
            "parts": self.parts_view(),
            "route": self.route_taken if len(self.route_taken) > 1 else self.route_planned,
            "route_is_observed": len(self.route_taken) > 1,
            "hop": self.hop,
            "hops_total": self._hops_total(),
            "resends": self.resends,
            "lost_frames": self.lost,
            "parity_recoveries": self.parity_recoveries,
            "reroutes": self.reroutes,
            "nacks": self.nacks,
            "bytes_on_air": self.bytes_on_air,
            "observed_loss": round(self.observed_loss(), 3),
            "effective_rate_bps": round(self.effective_rate_bps(), 1),
            "eta_s": None if self.eta_s is None else round(self.eta_s, 1),
            "elapsed_s": round((self.t_done or self.t_last) - (self.t_start or 0.0), 1) if self.t_start is not None else 0.0,
            "reason": self.reason,
        }
