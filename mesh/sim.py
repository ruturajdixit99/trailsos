"""Discrete-event LoRa mesh simulator.

Model
-----
* Nodes have map coordinates; a link exists iff distance <= range.
* Each node has one radio and a FIFO queue; it sends one packet at a time and
  waits for a hop-by-hop ACK (stop-and-wait ARQ).  Airtime = bytes*8/rate.
* Every transmission (data or ACK) is lost independently with the link's loss
  probability.  Half of the losses are modelled as corrupted frames that reach
  the receiver and are rejected by CRC32.
* A sender retries up to `max_retries` times, then marks the next hop as
  suspect (route cost penalty that grows with each failure, cleared by any
  ACK from it, expiring after `suspect_ttl_s`) and tries an alternate route.  No alternate
  route -> the packet is dropped at that node.
* The destination reassembles parts (XOR parity rebuilds one lost chunk per
  group of 4).  When every part is complete it sends ONE e2e-ack carrying the
  CRC32 of the whole message.  If it stalls while incomplete, it sends a NACK
  listing missing chunks.  The source probes when it hears nothing, and gives
  up (failed) after `max_probes` probes.
* All randomness comes from one seeded random.Random.

Simplifications (see README): no collisions/CSMA, ACKs do not wait for the
radio queue, routing uses global topology knowledge + local suspicion.
"""
from __future__ import annotations

import heapq
import itertools
import math
import random
import struct
import zlib
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

from media import chunking
from mesh import lora
from mesh.packet import (
    HEADER_SIZE, MEDIA_TYPES, PARITY_FLAG, ChecksumError, ContentType, Packet, PacketError,
    airtime_s, part_bit,
)

RATE_PRESETS = {"long_range": 1000.0, "fast": 21875.0}
ACK_SIZE = HEADER_SIZE + 1
SEND_ORDER = (ContentType.TRANSCRIPT, ContentType.CAPTION, ContentType.AUDIO, ContentType.THUMBNAIL, ContentType.VIDEO)
WHOLE_PART = 0xFFFF


# --------------------------------------------------------------------------- topology
@dataclass
class NodeSpec:
    name: str
    x: float
    y: float


def demo_topology() -> list[NodeSpec]:
    """Hiker -> B -> C -> D -> Ranger, with E bridging B and D (alternate path)."""
    return [
        NodeSpec("Hiker", 0.0, 0.0),
        NodeSpec("B", 1.0, 0.0),
        NodeSpec("C", 2.0, 0.0),
        NodeSpec("D", 3.0, 0.0),
        NodeSpec("Ranger", 4.0, 0.0),
        NodeSpec("E", 2.0, 0.7),
    ]


def three_hop_topology() -> list[NodeSpec]:
    """Hiker -> B -> D -> Ranger (3 hops), with E as an alternate relay for B."""
    return [
        NodeSpec("Hiker", 0.0, 0.0),
        NodeSpec("B", 1.0, 0.0),
        NodeSpec("D", 2.0, 0.0),
        NodeSpec("Ranger", 3.0, 0.0),
        NodeSpec("E", 1.5, 0.9),
    ]


@dataclass
class SimConfig:
    rate_bps: float = 1000.0
    loss: float = 0.20
    range_km: float = 1.6
    max_retries: int = 3
    seed: int = 0
    ack_margin_s: float = 0.25
    suspect_ttl_s: float = 120.0
    max_probes: int = 3
    max_nacks: int = 8          # NACKs without progress before the destination goes quiet
    link_loss: dict = field(default_factory=dict)  # {(a,b): p} overrides, unordered names
    # radio model: "simple" = fixed loss % + bytes*8/rate airtime (the spec model)
    #              "lora"   = LoRa PHY: Semtech airtime, path loss/SNR, fading,
    #                         collisions + capture, half-duplex, listen-before-talk
    radio: str = "simple"
    lora_preset: str = "long_fast"
    terrain_db: float = 0.0
    km_per_unit: float = 1.8
    max_backoffs: int = 8


@dataclass
class Event:
    t: float
    kind: str
    node: str
    peer: Optional[str] = None
    msg_id: Optional[int] = None
    part: Optional[str] = None
    index: Optional[int] = None
    parity: bool = False
    size: int = 0
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v not in (None, {}, False) or k == "t"}
        d["t"] = round(self.t, 3)
        return d


def _part_name(ctype: int) -> str:
    try:
        return ContentType(ctype & ~PARITY_FLAG).name.lower()
    except ValueError:
        return f"type{ctype}"


# --------------------------------------------------------------------------- node state
@dataclass
class _Node:
    id: int
    name: str
    x: float
    y: float
    alive: bool = True
    queue: deque = field(default_factory=deque)
    busy: bool = False
    cur: Optional[Packet] = None
    cur_nh: Optional[int] = None
    attempts: int = 0
    tried: set = field(default_factory=set)
    token: int = 0
    suspects: dict = field(default_factory=dict)  # neighbor id -> (suspect-until time, failure rounds)
    seen: set = field(default_factory=set)        # (msg, ctype, index, gen)
    path_of: dict = field(default_factory=dict)   # seen-key -> path list (sim bookkeeping)
    came_from: dict = field(default_factory=dict)  # seen-key -> previous hop id (split horizon)


@dataclass
class _TxMsg:
    msg_id: int
    src: int
    dst: int
    packets: dict           # (ctype, index) -> Packet (gen 0)
    order: list
    gen: dict               # (ctype, index) -> current generation
    crc: int
    parts_mask: int
    state: str = "sending"  # sending | delivered | failed
    probes: int = 0
    timer_tok: int = 0
    e2e_seen: int = 0


@dataclass
class _RxMsg:
    msg_id: int
    src: int
    mask: int
    asm: dict = field(default_factory=dict)       # ctype -> PartAssembler
    done_parts: dict = field(default_factory=dict)  # ctype -> bytes
    complete: bool = False
    crc: int = 0
    nack_tok: int = 0
    nacks: int = 0
    stall: int = 0  # NACKs sent since the last new packet arrived
    acks_sent: int = 0
    last_ack_t: float = -1e9


class MeshSim:
    def __init__(self, topology: list[NodeSpec] | None = None, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        topo = topology or demo_topology()
        self.nodes = [_Node(i, s.name, s.x, s.y) for i, s in enumerate(topo)]
        self.by_name = {n.name: n for n in self.nodes}
        self.rng = random.Random(self.cfg.seed)
        self.t = 0.0
        self._q: list = []
        self._seq = itertools.count()
        self._msg_ids = itertools.count(0x5A0001)
        self.events: list[Event] = []
        self.listeners: list[Callable[[Event], None]] = []
        self.tx: dict[int, _TxMsg] = {}
        self.rx: dict[tuple[int, int], _RxMsg] = {}
        self.rx_parts: dict[int, dict[str, bytes]] = {}  # msg_id -> reassembled parts at dst
        self.lora = self.cfg.radio == "lora"
        self.active_tx: list = []  # (sender_id, start, end) of frames on air (LoRa mode)
        if self.lora:
            self.preset = lora.PRESETS[self.cfg.lora_preset]
            self.env = lora.RadioEnv(terrain_db=self.cfg.terrain_db, km_per_unit=self.cfg.km_per_unit)
            self.cfg.rate_bps = self.preset.bitrate
            self.cfg.suspect_ttl_s = max(self.cfg.suspect_ttl_s, 900.0)  # LoRa runs take minutes
            self.cfg.max_probes = max(self.cfg.max_probes, 8)  # an SOS keeps trying for longer
            srng = random.Random(self.cfg.seed * 7919 + 17)  # shadowing: fixed per seed, symmetric per link
            self.shadow = {}
            for a in self.nodes:
                for b in self.nodes:
                    if a.id < b.id:
                        self.shadow[(a.id, b.id)] = self.shadow[(b.id, a.id)] = srng.gauss(0, self.env.shadow_sigma_db)

    # ------------------------------------------------------------------ radio physics
    def airtime(self, size: int) -> float:
        return lora.airtime_s(self.preset, size) if self.lora else airtime_s(size, self.cfg.rate_bps)

    def link_km(self, a: _Node, b: _Node) -> float:
        return self.dist(a, b) * (self.cfg.km_per_unit if self.lora else 1.0)

    def mean_snr(self, a: _Node, b: _Node) -> float:
        return self.env.mean_snr(self.preset, self.link_km(a, b), self.shadow[(a.id, b.id)])

    def mean_rssi(self, a: _Node, b: _Node) -> float:
        return self.env.mean_rssi(self.link_km(a, b), self.shadow[(a.id, b.id)])

    def link_per(self, a: _Node, b: _Node) -> float:
        """Fading-only packet error rate of one frame on a->b (LoRa mode)."""
        return self.env.per(self.preset, self.mean_snr(a, b))

    def link_cost(self, a: _Node, b: _Node) -> float:
        if not self.lora:
            return self.dist(a, b)
        ok = (1 - self.link_per(a, b)) * (1 - self.link_per(b, a))  # data and ack must both survive
        return 1.0 / max(ok, 1e-3)  # expected transmissions (ETX)

    def link_info(self) -> list[dict]:
        out = []
        for a in self.nodes:
            for b in self.nodes:
                if a.id < b.id and self.in_range(a, b):
                    d = {"a": a.name, "b": b.name, "km": round(self.link_km(a, b), 2)}
                    if self.lora:
                        snr = self.mean_snr(a, b)
                        d.update(rssi_dbm=round(self.mean_rssi(a, b), 1), snr_db=round(snr, 1),
                                 margin_db=round(snr - self.preset.snr_floor_db, 1),
                                 per=round(self.link_per(a, b), 3), etx=round(self.link_cost(a, b), 2))
                    else:
                        d.update(per=self.link_loss(a, b))
                    out.append(d)
        return out

    def _audible(self, frm: _Node, at: _Node) -> bool:
        return self.mean_snr(frm, at) >= self.preset.snr_floor_db - 3.0

    def _on_air(self, nid: int, start: float, end: float) -> None:
        self.active_tx = [x for x in self.active_tx if x[2] > self.t - 60.0]
        self.active_tx.append((nid, start, end))

    def channel_busy(self, n: _Node) -> bool:
        """Listen-before-talk: own radio busy, or an audible frame is on air."""
        for sid, st, en in self.active_tx:
            if st <= self.t < en and (sid == n.id or self._audible(self.nodes[sid], n)):
                return True
        return False

    def _rx_lora(self, s: _Node, d: _Node, start: float, end: float) -> tuple[bool, str, dict]:
        if not s.alive:
            return False, "sender down", {}
        if not d.alive:
            return False, "receiver down", {}
        if not self.in_range(s, d):
            return False, "out of range", {}
        rssi = self.mean_rssi(s, d) + self.rng.gauss(0, self.env.fade_sigma_db)
        snr = rssi - self.env.noise_dbm(self.preset)
        info = {"rssi_dbm": round(rssi, 1), "snr_db": round(snr, 1)}
        for sid, st, en in self.active_tx:
            if st < end and en > start and sid != s.id:
                if sid == d.id:
                    return False, "half-duplex", info
                other = self.nodes[sid]
                if self._audible(other, d) and rssi - self.mean_rssi(other, d) < self.env.capture_db:
                    info["interferer"] = other.name
                    return False, "collision", info
        if snr < self.preset.snr_floor_db:
            return False, "weak signal", info
        return True, "", info

    # ------------------------------------------------------------------ engine
    def schedule(self, delay: float, fn, *args) -> None:
        heapq.heappush(self._q, (self.t + delay, next(self._seq), fn, args))

    def run_until(self, t_end: float) -> None:
        while self._q and self._q[0][0] <= t_end:
            t, _, fn, args = heapq.heappop(self._q)
            self.t = t
            fn(*args)
        self.t = max(self.t, t_end)

    def run(self, max_t: float = 1e7) -> None:
        while self._q and self._q[0][0] <= max_t:
            t, _, fn, args = heapq.heappop(self._q)
            self.t = t
            fn(*args)

    @property
    def idle(self) -> bool:
        return not self._q

    def emit(self, kind: str, node: _Node | str, **kw) -> Event:
        name = node if isinstance(node, str) else node.name
        peer = kw.pop("peer", None)
        if isinstance(peer, _Node):
            peer = peer.name
        ev = Event(self.t, kind, name, peer=peer, **kw)
        self.events.append(ev)
        for fn in self.listeners:
            fn(ev)
        return ev

    def _pev(self, kind, node, pkt: Packet, **kw) -> Event:
        return self.emit(kind, node, msg_id=pkt.msg_id, part=_part_name(pkt.ctype), index=pkt.index,
                         parity=pkt.is_parity, size=pkt.size, **kw)

    # ------------------------------------------------------------------ links / routing
    def dist(self, a: _Node, b: _Node) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def in_range(self, a: _Node, b: _Node) -> bool:
        if a.id == b.id:
            return False
        if self.lora:  # usable link: mean SNR within 3 dB of the demodulation floor
            return self.mean_snr(a, b) >= self.preset.snr_floor_db - 3.0
        return self.dist(a, b) <= self.cfg.range_km

    def links(self) -> list[tuple[str, str]]:
        return [(a.name, b.name) for a in self.nodes for b in self.nodes
                if a.id < b.id and self.in_range(a, b)]

    def link_loss(self, a: _Node, b: _Node) -> float:
        ll = self.cfg.link_loss
        return ll.get((a.name, b.name), ll.get((b.name, a.name), self.cfg.loss))

    def shortest_path(self, src: int, dst: int, exclude: set = frozenset(),
                      first_hop_penalty: dict | None = None) -> Optional[list[int]]:
        """Dijkstra over usable links (distance, or ETX in LoRa mode), skipping `exclude`.
        `first_hop_penalty` multiplies the cost of src's own links (local suspicion)."""
        best = {src: 0.0}
        prev: dict[int, int] = {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                path = [dst]
                while path[-1] != src:
                    path.append(prev[path[-1]])
                return path[::-1]
            if d > best.get(u, math.inf):
                continue
            for v in self.nodes:
                if v.id in exclude or not self.in_range(self.nodes[u], v):
                    continue
                cost = self.link_cost(self.nodes[u], v)
                if u == src and first_hop_penalty:
                    cost *= first_hop_penalty.get(v.id, 1.0)
                nd = d + cost
                if nd < best.get(v.id, math.inf):
                    best[v.id] = nd
                    prev[v.id] = u
                    heapq.heappush(pq, (nd, v.id))
        return None

    def _penalties(self, n: _Node) -> dict:
        """Suspected neighbours cost (1 + 5 x failure rounds) more; a dead node sinks fast."""
        return {k: 1.0 + 5.0 * fails for k, (until, fails) in n.suspects.items() if until > self.t}

    def next_hop(self, n: _Node, dst: int, extra_exclude: set = frozenset()) -> Optional[int]:
        excl = set(extra_exclude) - {n.id}
        path = self.shortest_path(n.id, dst, excl, self._penalties(n))
        return path[1] if path and len(path) > 1 else None

    def planned_route(self, src: str, dst: str) -> list[str]:
        p = self.shortest_path(self.by_name[src].id, self.by_name[dst].id)
        return [self.nodes[i].name for i in p] if p else []

    # ------------------------------------------------------------------ public API
    def send_message(self, src: str, dst: str, parts: dict[ContentType, bytes]) -> int:
        """Chunk + parity every part, queue at `src` in priority order. Returns msg id."""
        s, d = self.by_name[src], self.by_name[dst]
        msg_id = next(self._msg_ids)
        mask = 0
        for ct in parts:
            mask |= part_bit(ct)
        packets, order = {}, []
        for ct in SEND_ORDER:
            if ct not in parts:
                continue
            data = chunking.split(parts[ct])
            par = chunking.parity_chunks(data)
            total = len(data)
            for g in range(len(par)):
                for i in range(g * chunking.GROUP, min((g + 1) * chunking.GROUP, total)):
                    p = Packet(msg_id, i, total, int(ct), s.id, d.id, data[i], parts_mask=mask)
                    packets[(int(ct), i)] = p
                    order.append((int(ct), i))
                pp = Packet(msg_id, g, total, int(ct) | PARITY_FLAG, s.id, d.id, par[g], parts_mask=mask)
                packets[(pp.ctype, g)] = pp
                order.append((pp.ctype, g))
        crc = _message_crc(parts)
        m = _TxMsg(msg_id, s.id, d.id, packets, order, {k: 0 for k in order}, crc, mask)
        self.tx[msg_id] = m
        route = self.planned_route(src, dst)
        self.emit("message_queued", s, msg_id=msg_id, peer=d,
                  detail={"packets": len(order), "bytes": sum(p.size for p in packets.values()),
                          "route": route,
                          "chunk_sizes": {ContentType(ct).name.lower(): [len(packets[(int(ct), i)].payload)
                                                                        for i in range(packets[(int(ct), 0)].total)]
                                          for ct in SEND_ORDER if ct in parts},
                          "parts": {ContentType(ct).name.lower(): len(b) for ct, b in parts.items()}})
        for k in order:
            s.queue.append(packets[k])
        self.schedule(0.0, self._pump, s.id)
        return msg_id

    def set_alive(self, name: str, alive: bool) -> None:
        n = self.by_name[name]
        if n.alive == alive:
            return
        n.alive = alive
        if not alive:
            lost = len(n.queue) + (1 if n.busy else 0)
            n.queue.clear()
            n.busy, n.cur, n.cur_nh = False, None, None
            n.token += 1
            self.emit("node_down", n, detail={"queued_packets_lost": lost})
        else:
            n.seen.clear()
            self.emit("node_up", n)
            self.schedule(0.0, self._pump, n.id)

    # ------------------------------------------------------------------ radio send path
    def _attempt_time(self, size: int) -> float:
        return self.airtime(size) + self.airtime(ACK_SIZE) + self.cfg.ack_margin_s

    def _pump(self, nid: int) -> None:
        n = self.nodes[nid]
        if not n.alive or n.busy or not n.queue:
            return
        pkt = n.queue.popleft()
        if pkt.ctype not in (ContentType.E2E_ACK, ContentType.NACK) and pkt.src == n.id:
            m = self.tx.get(pkt.msg_id)
            if m and m.state != "sending":  # message already settled; skip stale resend
                self.schedule(0.0, self._pump, nid)
                return
        prev = n.came_from.get((pkt.msg_id, pkt.ctype, pkt.index, pkt.src, pkt.gen))
        back = {prev} if prev is not None and prev != pkt.dst else set()  # split horizon: never bounce back
        nh = self.next_hop(n, pkt.dst, back)
        if nh is None:
            self._pev("dropped", n, pkt, detail={"reason": f"no route from {n.name} to {self.nodes[pkt.dst].name}"})
            self._after_settle(n, pkt)
            self.schedule(0.0, self._pump, nid)
            return
        n.busy, n.cur, n.cur_nh, n.attempts, n.tried = True, pkt, nh, 0, {nh} | back
        self._attempt(n)

    def _attempt(self, n: _Node, backoffs: int = 0) -> None:
        pkt, nh = n.cur, self.nodes[n.cur_nh]
        n.token += 1
        tok = n.token
        if self.lora and backoffs < self.cfg.max_backoffs and self.channel_busy(n):
            slot = 4 * self.preset.t_sym + self.airtime(ACK_SIZE) / 4
            wait = self.rng.uniform(1, 2 ** min(backoffs + 2, 6)) * slot
            self._pev("backoff", n, pkt, peer=nh, detail={"wait_ms": round(wait * 1000), "n": backoffs + 1})
            self.schedule(wait, self._retry_after_backoff, n.id, tok, backoffs + 1)
            return
        self._pev("sent" if n.attempts == 0 else "retried", n, pkt, peer=nh,
                  detail={"attempt": n.attempts + 1, "hop": pkt.hop + 1,
                          "airtime_ms": round(self.airtime(pkt.size) * 1000, 1)})
        air = self.airtime(pkt.size)
        if self.lora:
            self._on_air(n.id, self.t, self.t + air)
        self.schedule(air, self._arrive, n.id, nh.id, pkt, tok, self.t)
        self.schedule(air + self.airtime(ACK_SIZE) + self.cfg.ack_margin_s, self._timeout, n.id, tok)

    def _retry_after_backoff(self, nid: int, tok: int, backoffs: int) -> None:
        n = self.nodes[nid]
        if n.alive and n.busy and n.token == tok:
            self._attempt(n, backoffs)

    def _channel_ok(self, a: _Node, b: _Node) -> tuple[bool, str]:
        if not a.alive:
            return False, "sender down"
        if not b.alive:
            return False, "receiver down"
        if not self.in_range(a, b):
            return False, "out of range"
        if self.rng.random() < self.link_loss(a, b):
            return False, "loss"
        return True, ""

    def _arrive(self, sid: int, did: int, pkt: Packet, tok: int, start: float = 0.0) -> None:
        s, d = self.nodes[sid], self.nodes[did]
        raw = pkt.with_hop(pkt.hop + 1).encode()
        rf: dict = {}
        if self.lora:
            ok, why, rf = self._rx_lora(s, d, start, self.t)
            if why == "collision":  # overlapping frames arrive garbled: CRC32 must reject them
                bad = bytearray(raw)
                for _ in range(4):
                    bad[self.rng.randrange(len(bad))] ^= 0xFF
                try:
                    Packet.decode(bytes(bad))
                    ok, why = True, ""  # pragma: no cover - would be a CRC miss
                except PacketError:
                    pass
        else:
            ok, why = self._channel_ok(s, d)
        if not ok and why == "loss" and self.rng.random() < 0.5:
            # corrupted frame: reaches the receiver, rejected by CRC32
            bad = bytearray(raw)
            bad[self.rng.randrange(len(bad))] ^= 1 << self.rng.randrange(8)
            try:
                Packet.decode(bytes(bad))
                why = "loss"  # pragma: no cover - crc32 always catches single-bit flips
            except ChecksumError:
                why = "crc_fail"
            except PacketError:
                why = "crc_fail"
        if not ok:
            self._pev("lost", s, pkt, peer=d, detail={"reason": why, **rf})
            return
        rx = Packet.decode(raw)
        self._send_hop_ack(d, s, rx, tok)
        self._on_receive(d, s, rx, rf)

    def _send_hop_ack(self, d: _Node, s: _Node, pkt: Packet, tok: int) -> None:
        ack = Packet(pkt.msg_id, pkt.index, 0, int(ContentType.ACK), d.id, s.id, bytes([pkt.ctype]))
        self.emit("ack_sent", d, peer=s, msg_id=pkt.msg_id, part=_part_name(pkt.ctype), index=pkt.index,
                  size=ack.size)
        air = self.airtime(ack.size)
        if self.lora:
            self._on_air(d.id, self.t, self.t + air)
        self.schedule(air, self._ack_arrive, d.id, s.id, ack, tok, self.t)

    def _ack_arrive(self, did: int, sid: int, ack: Packet, tok: int, start: float = 0.0) -> None:
        d, s = self.nodes[did], self.nodes[sid]
        rf: dict = {}
        if self.lora:
            ok, why, rf = self._rx_lora(d, s, start, self.t)
        else:
            ok, why = self._channel_ok(d, s)
        if not ok:
            self.emit("lost", d, peer=s, msg_id=ack.msg_id, part="ack", index=ack.index, size=ack.size,
                      detail={"reason": why, "ack_for": _part_name(ack.payload[0]), **rf})
            return
        if s.token != tok or not s.busy:
            return  # stale
        pkt = s.cur
        s.suspects.pop(d.id, None)  # it answered: no longer suspect
        self._pev("acked", s, pkt, peer=d)
        s.busy, s.cur, s.cur_nh = False, None, None
        self._after_settle(s, pkt)
        self._pump(s.id)

    def _timeout(self, nid: int, tok: int) -> None:
        n = self.nodes[nid]
        if not n.alive or n.token != tok or not n.busy:
            return
        pkt = n.cur
        n.attempts += 1
        if n.attempts <= self.cfg.max_retries:
            if self.lora:  # random retransmit jitter breaks repeating hidden-terminal collisions
                wait = self.rng.uniform(0.2, 1.0) * n.attempts * self.airtime(230)
                self.schedule(wait, self._retry_after_backoff, n.id, n.token, 0)
            else:
                self._attempt(n)
            return
        bad = n.cur_nh
        _, fails = n.suspects.get(bad, (0.0, 0))
        n.suspects[bad] = (self.t + self.cfg.suspect_ttl_s, fails + 1)
        self.emit("link_down", n, peer=self.nodes[bad], msg_id=pkt.msg_id,
                  detail={"after_attempts": n.attempts})
        alt = self.next_hop(n, pkt.dst, extra_exclude=n.tried)
        if alt is not None:
            n.tried.add(alt)
            n.cur_nh, n.attempts = alt, 0
            path = self.shortest_path(alt, pkt.dst, (n.tried | {n.id}) - {alt, pkt.dst})
            via = [n.name] + [self.nodes[i].name for i in (path or [alt])]
            self._pev("rerouted", n, pkt, peer=self.nodes[alt], detail={"via": via})
            self._attempt(n)
            return
        self._pev("dropped", n, pkt, detail={
            "reason": f"{self.nodes[bad].name} unreachable from {n.name} after "
                      f"{self.cfg.max_retries + 1} attempts; no alternate route to {self.nodes[pkt.dst].name}"})
        n.busy, n.cur, n.cur_nh = False, None, None
        self._after_settle(n, pkt)
        self._pump(n.id)

    # ------------------------------------------------------------------ receive path
    def _on_receive(self, n: _Node, frm: _Node, pkt: Packet, rf: dict | None = None) -> None:
        rf = rf or {}
        seen_key = (pkt.msg_id, pkt.ctype, pkt.index, pkt.src, pkt.gen)
        dup = seen_key in n.seen
        n.seen.add(seen_key)
        if not dup:
            n.path_of[seen_key] = frm.path_of.get(seen_key, [frm.name]) + [n.name]
            n.came_from[seen_key] = frm.id
        if pkt.dst == n.id:
            self._at_destination(n, frm, pkt, dup, n.path_of.get(seen_key, []), rf)
            return
        if dup:
            self._pev("duplicate", n, pkt, peer=frm)
            return
        self._pev("relayed", n, pkt, peer=frm, detail={"hop": pkt.hop, **rf})
        if pkt.ctype in (ContentType.E2E_ACK, ContentType.NACK):
            n.queue.appendleft(pkt)  # control traffic jumps the queue
        else:
            n.queue.append(pkt)
        self._pump(n.id)

    def _at_destination(self, n: _Node, frm: _Node, pkt: Packet, dup: bool, path: list, rf: dict) -> None:
        if pkt.ctype in (ContentType.E2E_ACK, ContentType.NACK):
            self._pev("duplicate" if dup else "control_received", n, pkt, peer=frm, detail={"path": path})
            if pkt.ctype == ContentType.E2E_ACK:
                self._source_got_e2e(n, pkt, dup)
            elif not dup:
                self._source_got_nack(n, pkt)
            return
        key = (pkt.msg_id, n.id)
        r = self.rx.get(key)
        if r is None:
            r = self.rx[key] = _RxMsg(pkt.msg_id, pkt.src, pkt.parts_mask)
        if r.complete:
            self._pev("duplicate", n, pkt, peer=frm, detail={"note": "message already complete"})
            # source has not heard our e2e-ack yet (probe / late dup): re-ack, rate limited
            if self.t - r.last_ack_t > self._e2e_wait() / 2:  # roughly once per source probe interval
                self._send_e2e(n, r)
            return
        if dup:
            self._pev("duplicate", n, pkt, peer=frm)
            return
        self._pev("received", n, pkt, peer=frm, detail={"hop": pkt.hop, "path": path, **rf})
        r.stall = 0  # the path works again: NACKing is worthwhile
        base = int(pkt.base_type)
        if base in r.done_parts:
            self._arm_nack(n, r)
            return
        a = r.asm.get(base)
        if a is None:
            a = r.asm[base] = chunking.PartAssembler(pkt.total)
        if pkt.is_parity:
            a.add_parity(pkt.index, pkt.payload)
        else:
            a.add_data(pkt.index, pkt.payload)
        for idx in a.try_recover():
            self.emit("parity_recovered", n, msg_id=pkt.msg_id, part=_part_name(base), index=idx)
        if a.complete:
            body = a.assemble()
            r.done_parts[base] = body
            self.rx_parts.setdefault(pkt.msg_id, {})[_part_name(base)] = body
            self.emit("reassembled", n, msg_id=pkt.msg_id, part=_part_name(base), size=len(body),
                      detail={"chunks": a.total, "recovered": a.recovered, "crc32": zlib.crc32(body)})
        if all(int(ct) in r.done_parts for ct in MEDIA_TYPES if r.mask & part_bit(ct)):
            r.complete = True
            r.nack_tok += 1
            r.crc = _message_crc({ContentType(ct): b for ct, b in r.done_parts.items()})
            self.emit("message_complete", n, msg_id=pkt.msg_id, detail={"crc32": r.crc})
            self._send_e2e(n, r)
        else:
            self._arm_nack(n, r)

    def _send_e2e(self, n: _Node, r: _RxMsg) -> None:
        r.acks_sent += 1
        r.last_ack_t = self.t
        p = Packet(r.msg_id, r.acks_sent, 0, int(ContentType.E2E_ACK), n.id, r.src, struct.pack(">I", r.crc))
        self.emit("e2e_ack_sent", n, msg_id=r.msg_id, peer=self.nodes[r.src], size=p.size,
                  detail={"crc32": r.crc, "n": r.acks_sent})
        n.queue.appendleft(p)
        self._pump(n.id)

    def _arm_nack(self, n: _Node, r: _RxMsg) -> None:
        r.nack_tok += 1
        tok = r.nack_tok
        wait = 2 * (self.cfg.max_retries + 1) * self._attempt_time(230) * 2 ** min(r.stall, 3)  # back off
        self.schedule(wait, self._nack_fire, n.id, r.msg_id, tok)

    def _nack_fire(self, nid: int, msg_id: int, tok: int) -> None:
        n = self.nodes[nid]
        r = self.rx.get((msg_id, nid))
        if r is None or r.complete or r.nack_tok != tok or not n.alive:
            return
        if r.stall >= self.cfg.max_nacks:
            return
        want: list[tuple[int, int]] = []
        for ct in MEDIA_TYPES:
            if not r.mask & part_bit(ct) or int(ct) in r.done_parts:
                continue
            a = r.asm.get(int(ct))
            if a is None:
                want.append((int(ct), WHOLE_PART))
                continue
            for g in a.unrecoverable_groups():
                for i in range(g * chunking.GROUP, min((g + 1) * chunking.GROUP, a.total)):
                    if i not in a.data:
                        want.append((int(ct), i))
        want = want[:66]
        r.nacks += 1
        r.stall += 1
        payload = b"".join(struct.pack(">BH", c, i) for c, i in want)
        p = Packet(msg_id, r.nacks, 0, int(ContentType.NACK), n.id, r.src, payload)
        self.emit("nack_sent", n, msg_id=msg_id, peer=self.nodes[r.src], size=p.size,
                  detail={"missing": [f"{_part_name(c)}#{'all' if i == WHOLE_PART else i}" for c, i in want]})
        n.queue.appendleft(p)
        self._pump(n.id)
        self._arm_nack(n, r)

    # ------------------------------------------------------------------ source side
    def _after_settle(self, n: _Node, pkt: Packet) -> None:
        """Called when a node finished with a packet (acked or dropped)."""
        m = self.tx.get(pkt.msg_id)
        if not m or m.src != n.id or m.state != "sending":
            return
        if any(q.msg_id == m.msg_id and q.src == m.src and q.base_type in MEDIA_TYPES for q in n.queue):
            return
        self._arm_e2e_timer(m)

    def _e2e_wait(self) -> float:
        """How long the source waits for an e2e-ack before probing."""
        hops = max(len(self.nodes) - 1, 1)
        return (hops + 2) * (self.cfg.max_retries + 1) * self._attempt_time(230)

    def _arm_e2e_timer(self, m: _TxMsg) -> None:
        m.timer_tok += 1
        tok = m.timer_tok
        wait = self._e2e_wait()
        self.schedule(wait, self._e2e_timeout, m.msg_id, tok)

    def _e2e_timeout(self, msg_id: int, tok: int) -> None:
        m = self.tx[msg_id]
        if m.state != "sending" or m.timer_tok != tok:
            return
        src = self.nodes[m.src]
        if m.probes >= self.cfg.max_probes or not src.alive:
            m.state = "failed"
            drops = [e for e in self.events if e.kind == "dropped" and e.msg_id == msg_id]
            reason = (f"No end-to-end ack from {self.nodes[m.dst].name} after {m.probes} probes"
                      + (f"; last drop: {drops[-1].detail['reason']}" if drops else ""))
            self.emit("message_failed", src, msg_id=msg_id, detail={"reason": reason})
            return
        m.probes += 1
        k = m.order[0]
        m.gen[k] += 1
        self.emit("probe", src, msg_id=msg_id, detail={"n": m.probes})
        src.queue.append(replace(m.packets[k], gen=m.gen[k] & 0xFF))
        self._pump(src.id)
        self._arm_e2e_timer(m)

    def _source_got_nack(self, n: _Node, pkt: Packet) -> None:
        m = self.tx.get(pkt.msg_id)
        if not m or m.state != "sending":
            return
        want = [struct.unpack(">BH", pkt.payload[i:i + 3]) for i in range(0, len(pkt.payload), 3)]
        keys = []
        for c, i in want:
            if i == WHOLE_PART:
                keys += [k for k in m.order if (k[0] & ~PARITY_FLAG) == c]
            elif (c, i) in m.packets:
                keys.append((c, i))
        queued = {(q.ctype, q.index) for q in n.queue if q.msg_id == m.msg_id}
        added = 0
        for k in keys:
            if k in queued:
                continue
            m.gen[k] += 1
            n.queue.append(replace(m.packets[k], gen=m.gen[k] & 0xFF))
            added += 1
        self.emit("nack_received", n, msg_id=pkt.msg_id, detail={"resending": added})
        m.timer_tok += 1  # e2e timer re-armed once the resends drain
        self._pump(n.id)

    def _source_got_e2e(self, n: _Node, pkt: Packet, dup: bool) -> None:
        m = self.tx.get(pkt.msg_id)
        if not m or m.state != "sending":
            return
        (crc,) = struct.unpack(">I", pkt.payload[:4])
        if crc != m.crc:
            self.emit("e2e_ack_mismatch", n, msg_id=pkt.msg_id, detail={"got": crc, "want": m.crc})
            return
        m.state = "delivered"
        m.timer_tok += 1
        self.emit("e2e_ack_received", n, msg_id=pkt.msg_id, peer=self.nodes[m.dst], detail={"crc32": crc})



def _message_crc(parts: dict) -> int:
    crc = 0
    for ct in SEND_ORDER:
        if ct in parts:
            crc = zlib.crc32(bytes([int(ct)]) + struct.pack(">H", len(parts[ct])) + parts[ct], crc)
    return crc & 0xFFFFFFFF
