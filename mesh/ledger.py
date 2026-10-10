"""Network ledger: what was sent vs what was received, per link and per chunk.

Built only from simulator events, like status.py.  A "frame" is one radio
transmission on one hop (first try or retry); a "chunk" is one data or parity
packet of the message, which may need many frames to cross the mesh.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mesh.sim import Event

LOSS_REASONS = ("loss", "crc_fail", "weak signal", "collision", "half-duplex", "receiver down", "sender down",
                "out of range")
PART_ORDER = {"transcript": 0, "caption": 1, "audio": 2, "thumbnail": 3, "nack": 4, "e2e_ack": 5}


@dataclass
class LinkStats:
    frm: str
    to: str
    frames: int = 0           # data/parity/control frames transmitted frm -> to
    bytes: int = 0
    delivered: int = 0        # frames that passed CRC at `to`
    lost: dict = field(default_factory=lambda: {r: 0 for r in LOSS_REASONS})
    acks_sent: int = 0        # hop ACKs sent back to -> frm
    acks_lost: int = 0

    @property
    def lost_total(self) -> int:
        return sum(self.lost.values())

    def view(self) -> dict:
        return {"link": f"{self.frm}→{self.to}", "frames": self.frames, "bytes": self.bytes,
                "delivered": self.delivered, "lost": self.lost_total,
                "lost_random": self.lost["loss"] + self.lost["weak signal"], "lost_crc": self.lost["crc_fail"],
                "lost_collision": self.lost["collision"], "lost_half_duplex": self.lost["half-duplex"],
                "lost_node_down": self.lost["receiver down"] + self.lost["sender down"],
                "loss_pct": round(100 * self.lost_total / self.frames, 1) if self.frames else 0.0,
                "acks_sent": self.acks_sent, "acks_lost": self.acks_lost}


@dataclass
class ChunkStats:
    label: str
    part: str
    index: int
    parity: bool
    size: int = 0
    source_sends: int = 0                       # first sends by the source (1 + NACK/probe resends)
    frames: int = 0                             # all transmissions on all hops
    lost: int = 0
    hops: list = field(default_factory=list)    # [{"from","to","tries","lost","ok"}]
    at: str = ""                                # last node known to hold it
    arrived_t: float | None = None
    path: list = field(default_factory=list)
    recovered: bool = False
    dropped: list = field(default_factory=list)  # reasons
    duplicates_at_dest: int = 0

    def _hop(self, frm: str, to: str) -> dict:
        if self.hops and self.hops[-1]["from"] == frm and self.hops[-1]["to"] == to and not self.hops[-1]["ok"]:
            return self.hops[-1]
        h = {"from": frm, "to": to, "tries": 0, "lost": 0, "ok": False}
        self.hops.append(h)
        return h

    @property
    def status(self) -> str:
        if self.arrived_t is not None:
            return "arrived"
        if self.recovered:
            return "rebuilt by parity"
        if self.dropped and not self.at:
            return "dropped"
        if self.frames:
            return f"in transit (at {self.at})" if self.at else "in transit"
        return "queued"

    def view(self) -> dict:
        return {"label": self.label, "part": self.part, "index": self.index, "parity": self.parity,
                "size": self.size, "source_sends": self.source_sends, "frames": self.frames, "lost": self.lost,
                "hops": self.hops[-8:], "status": self.status, "arrived_t": self.arrived_t,
                "path": self.path, "recovered": self.recovered, "dropped": self.dropped[-2:],
                "dup_at_dest": self.duplicates_at_dest}


@dataclass
class NetworkLedger:
    msg_id: int
    src: str = "Hiker"
    dst: str = "Ranger"
    links: dict = field(default_factory=dict)
    chunks: dict = field(default_factory=dict)
    nodes: dict = field(default_factory=dict)   # name -> {tx, rx, relayed, dropped, dup}
    lost_by_reason: dict = field(default_factory=lambda: {r: 0 for r in LOSS_REASONS})
    src_frames: int = 0
    src_bytes: int = 0
    dst_unique: int = 0
    dst_bytes: int = 0
    dst_dups: int = 0
    nacks: list = field(default_factory=list)
    timeline: list = field(default_factory=list)  # compact rows for the UI timeline

    def _link(self, a: str, b: str) -> LinkStats:
        k = (a, b)
        if k not in self.links:
            self.links[k] = LinkStats(a, b)
        return self.links[k]

    def _node(self, n: str) -> dict:
        return self.nodes.setdefault(n, {"tx": 0, "rx": 0, "relayed": 0, "dropped": 0, "dup": 0, "acks_tx": 0})

    def _chunk(self, ev: Event) -> ChunkStats | None:
        if ev.part is None or ev.part == "ack" or ev.index is None:
            return None
        label = f"{ev.part}{'-parity' if ev.parity else ''}#{ev.index}"
        if label not in self.chunks:
            self.chunks[label] = ChunkStats(label, ev.part, ev.index, ev.parity)
        return self.chunks[label]

    def on_event(self, ev: Event) -> None:
        if ev.msg_id != self.msg_id:
            return
        k = ev.kind
        c = self._chunk(ev)
        if k in ("sent", "retried"):
            link = self._link(ev.node, ev.peer)
            link.frames += 1
            link.bytes += ev.size
            self._node(ev.node)["tx"] += 1
            if c:
                c.size = ev.size
                c.frames += 1
                c._hop(ev.node, ev.peer)["tries"] += 1
                if ev.node == self.src and k == "sent" and c.part not in ("nack", "e2e_ack"):
                    c.source_sends += 1
            if ev.node == self.src:
                self.src_frames += 1
                self.src_bytes += ev.size
            self.timeline.append([round(ev.t, 2), "tx", ev.node, ev.peer, c.label if c else ev.part])
        elif k == "lost":
            reason = ev.detail.get("reason", "loss")
            if ev.part == "ack":
                self._link(ev.peer, ev.node).acks_lost += 1   # ack travels to -> frm of the data link
            else:
                link = self._link(ev.node, ev.peer)
                link.lost[reason] = link.lost.get(reason, 0) + 1
                self.lost_by_reason[reason] = self.lost_by_reason.get(reason, 0) + 1
                if c:
                    c.lost += 1
                    c._hop(ev.node, ev.peer)["lost"] += 1
                self.timeline.append([round(ev.t, 2), "lost", ev.node, ev.peer, c.label if c else ev.part, reason])
        elif k == "ack_sent":
            self._link(ev.peer, ev.node).acks_sent += 1
            self._node(ev.node)["acks_tx"] += 1
        elif k == "control_received":
            self._link(ev.peer, ev.node).delivered += 1
            self._node(ev.node)["rx"] += 1
            if c:
                c._hop(ev.peer, ev.node)["ok"] = True
                c.at = ev.node
                c.arrived_t = round(ev.t, 2)
                c.path = ev.detail.get("path", [])
        elif k in ("relayed", "received"):
            self._link(ev.peer, ev.node).delivered += 1
            self._node(ev.node)["rx"] += 1
            if k == "relayed":
                self._node(ev.node)["relayed"] += 1
            if c:
                h = c._hop(ev.peer, ev.node)
                h["ok"] = True
                c.at = ev.node
                if k == "received":
                    c.arrived_t = round(ev.t, 2)
                    c.path = ev.detail.get("path", [])
                    self.dst_unique += 1
                    self.dst_bytes += ev.size
                    self.timeline.append([round(ev.t, 2), "arrived", ev.node, ev.peer, c.label])
        elif k == "duplicate":
            self._link(ev.peer, ev.node).delivered += 1
            self._node(ev.node)["rx"] += 1
            self._node(ev.node)["dup"] += 1
            if ev.node == self.dst:
                self.dst_dups += 1
                if c:
                    c.duplicates_at_dest += 1
        elif k == "dropped":
            self._node(ev.node)["dropped"] += 1
            if c:
                c.dropped.append(f"{ev.node}: {ev.detail.get('reason', '')}")
                c.at = ""
            self.timeline.append([round(ev.t, 2), "dropped", ev.node, None, c.label if c else ev.part])
        elif k == "parity_recovered" and c:
            c.recovered = True
            self.timeline.append([round(ev.t, 2), "rebuilt", ev.node, None, c.label])
        elif k == "nack_sent":
            self.nacks.append({"t": round(ev.t, 2), "missing": ev.detail.get("missing", [])})

    # ------------------------------------------------------------------ views
    def totals(self) -> dict:
        frames = sum(l.frames for l in self.links.values())
        lost = sum(l.lost_total for l in self.links.values())
        acks = sum(l.acks_sent for l in self.links.values())
        acks_lost = sum(l.acks_lost for l in self.links.values())
        media = [c for c in self.chunks.values() if c.part in ("transcript", "caption", "audio", "thumbnail")]
        return {
            "source_chunks": len([c for c in media if c.source_sends]),
            "source_frames": self.src_frames, "source_bytes": self.src_bytes,
            "dest_unique_chunks": self.dst_unique, "dest_bytes": self.dst_bytes, "dest_duplicates": self.dst_dups,
            "chunks_rebuilt_by_parity": sum(1 for c in media if c.recovered and c.arrived_t is None),
            "chunks_dropped_somewhere": sum(1 for c in media if c.dropped),
            "frames_on_air": frames, "frames_lost": lost, "frames_delivered": frames - lost,
            "frame_loss_pct": round(100 * lost / frames, 1) if frames else 0.0,
            "acks_sent": acks, "acks_lost": acks_lost,
            "lost_by_reason": dict(self.lost_by_reason),
        }

    def snapshot(self) -> dict:
        chunks = sorted(self.chunks.values(), key=lambda c: (PART_ORDER.get(c.part, 9), c.parity, c.index))
        return {"totals": self.totals(),
                "links": [l.view() for l in sorted(self.links.values(), key=lambda l: (l.frm, l.to))],
                "chunks": [c.view() for c in chunks],
                "nodes": self.nodes,
                "nacks": self.nacks[-5:]}
