"""A live mesh session: one shared simulated network, any phone sending to any phone.

Every message has its own honest StatusTracker, NetworkLedger and receiving-phone view
(RangerView). They all share the same radios, so they compete for airtime exactly as
real Meshtastic traffic would.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from mesh.ledger import NetworkLedger
from mesh.sim import MeshSim
from mesh.status import StatusTracker
from mesh.summary import summarize
from server.sos import KIND_LABEL, RangerView, SosPayload, sent_vs_received

log = logging.getLogger("trail_sos.session")

SCENARIOS = {
    "everyone_to_ranger": {
        "label": "Everyone reports to Ranger",
        "about": "Hiker sends an SOS; relays B, C, D and E each send text, a photo, a voice note and a video to the Ranger.",
        "sends": [("Hiker", "Ranger", "sos", 0, None), ("B", "Ranger", "text", 2, "B: trail is clear at mile two."),
                  ("C", "Ranger", "photo", 4, None), ("D", "Ranger", "voice", 6, None),
                  ("E", "Ranger", "video", 8, None)],
    },
    "ranger_broadcast": {
        "label": "Ranger broadcasts to all",
        "about": "The Ranger sends the same warning to every phone (5 separate deliveries, each confirmed on its own).",
        "sends": [("Ranger", n, "text", i, "Ranger: storm at 16:00, shelter in place.")
                  for i, n in enumerate(("Hiker", "B", "C", "D", "E"))],
    },
    "two_way": {
        "label": "Two-way chat",
        "about": "Hiker sends a voice note and photo; the Ranger replies with text; Hiker sends a short video.",
        "sends": [("Hiker", "Ranger", "voice", 0, None), ("Hiker", "Ranger", "photo", 1, None),
                  ("Ranger", "Hiker", "text", 3, "Ranger: got your location. Team on the way, ETA 40 min."),
                  ("Hiker", "Ranger", "video", 6, None)],
    },
    "busy_mesh": {
        "label": "Busy mesh",
        "about": "Five phones talk at once in different directions: the radios contend for the same channel.",
        "sends": [("Hiker", "Ranger", "sos", 0, None), ("B", "D", "text", 0.5, "B to D: meet at the bridge."),
                  ("E", "Hiker", "photo", 1, None), ("D", "B", "voice", 1.5, None),
                  ("Ranger", "C", "text", 2, "Ranger to C: report your position."),
                  ("C", "E", "video", 2.5, None)],
    },
}


@dataclass
class Message:
    msg_id: int
    src: str
    dst: str
    kind: str
    payload: SosPayload
    tracker: StatusTracker
    ledger: NetworkLedger
    view: RangerView
    trace_session: str
    created_t: float
    number: int
    rebuild_task: Optional[asyncio.Task] = None
    ai_tasks: list = field(default_factory=list)
    done: bool = False

    @property
    def hex(self) -> str:
        return f"{self.msg_id:08x}"

    def brief(self) -> dict:
        s = self.tracker.snapshot()
        return {"id": self.hex, "n": self.number, "src": self.src, "dst": self.dst, "kind": self.kind,
                "kind_label": KIND_LABEL[self.kind], "state": s["state"], "label": s["label"],
                "eta_s": s["eta_s"], "elapsed_s": s["elapsed_s"], "route": s["route"], "parts": s["parts"],
                "bytes": sum(self.payload.sizes().values()), "resends": s["resends"],
                "sent_t": round(self.created_t, 1),
                "verified": {k: (self.view.view.get(k) or {}).get("verdict") for k in ("verify_transcript",
                                                                                         "verify_caption")}}


@dataclass
class Session:
    sim: MeshSim
    request: dict
    speed: float
    radio: dict
    knockout: Optional[str] = None
    knockout_after_rx: int = 4
    messages: dict = field(default_factory=dict)     # msg_id -> Message
    order: list = field(default_factory=list)
    focus: Optional[int] = None
    pending: list = field(default_factory=list)
    knocked: bool = False
    rx_count: int = 0
    task: Optional[asyncio.Task] = None
    scenario: Optional[str] = None
    on_launch: list = field(default_factory=list)    # callbacks(Message)

    def launch(self, src: str, dst: str, kind: str, payload: SosPayload, provider) -> Message:
        """Put a built message on the air now (called directly or from a scheduled sim event)."""
        msg_id = self.sim.send_message(src, dst, payload.parts)
        m = Message(msg_id, src, dst, kind, payload, StatusTracker(msg_id, self.sim.cfg.rate_bps,
                                                                    self.sim.cfg.ack_margin_s),
                    NetworkLedger(msg_id, src, dst), RangerView(provider), f"sos-{msg_id:08x}", self.sim.t,
                    len(self.order) + 1)
        for ev in self.sim.events:  # the message_queued event already happened
            if ev.msg_id == msg_id:
                m.tracker.on_event(ev)
                m.ledger.on_event(ev)
        self.messages[msg_id] = m
        self.order.append(msg_id)
        if self.focus is None:
            self.focus = msg_id
        for cb in self.on_launch:
            cb(m)
        return m

    @property
    def focused(self) -> Optional[Message]:
        return self.messages.get(self.focus) if self.focus is not None else None

    def find(self, hex_id: str) -> Optional[Message]:
        for m in self.messages.values():
            if m.hex == hex_id:
                return m
        return None

    def phones(self) -> list[dict]:
        out = []
        for n in self.sim.nodes:
            tx = rx = relayed = dropped = 0
            for m in self.messages.values():
                st = m.ledger.nodes.get(n.name, {})
                tx += st.get("tx", 0)
                rx += st.get("rx", 0)
                relayed += st.get("relayed", 0)
                dropped += st.get("dropped", 0)
            inbox = [m.brief() for m in self.messages.values() if m.dst == n.name]
            outbox = [m.brief() for m in self.messages.values() if m.src == n.name]
            out.append({"name": n.name, "alive": n.alive, "queue": len(n.queue) + (1 if n.busy else 0),
                        "tx": tx, "rx": rx, "relayed": relayed, "dropped": dropped,
                        "inbox": inbox[-4:], "outbox": outbox[-4:],
                        "role": "sender" if outbox and not inbox else "receiver" if inbox and not outbox
                        else "sender + receiver" if inbox else "relay"})
        return out

    def summary(self, m: Message) -> dict:
        return summarize(self.sim, m.tracker, m.ledger, m.payload.sizes(), self.radio)

    def compare(self, m: Message) -> list:
        return sent_vs_received(m.payload, self.sim.rx_parts.get(m.msg_id, {}), m.view)
