"""FastAPI app: runs the mesh simulation in (scaled) real time and streams it over WebSocket."""
from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ai.provider import get_provider
from media import voice
from mesh.sim import RATE_PRESETS, MeshSim, SimConfig, demo_topology
from mesh.status import StatusTracker
from observability.prism import PrismTracer
from server.sos import ROOT, RangerView, build_sos

load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("trail_sos.app")

STATIC = Path(__file__).parent / "static"
SRC, DST = "Hiker", "Ranger"
TICK_S = 0.1


class SosRequest(BaseModel):
    loss: float = 0.20
    rate: str = "long_range"
    seed: int = 7
    speed: float = 10.0
    knockout: Optional[str] = None          # node to knock out mid-send
    knockout_after_rx: int = 4              # ...once the ranger has this many chunks


@dataclass
class Run:
    sim: MeshSim
    tracker: StatusTracker
    ranger: RangerView
    msg_id: int
    session: str
    speed: float
    payload_sizes: dict
    codec: str
    knockout: Optional[str] = None
    knockout_after_rx: int = 4
    knocked: bool = False
    rx_count: int = 0
    pending: list = field(default_factory=list)
    finished: bool = False


class Hub:
    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.run: Optional[Run] = None
        self.task: Optional[asyncio.Task] = None
        self.provider = get_provider()
        self.tracer = PrismTracer()
        self.handshake = "pending" if self.tracer.enabled else "tracing disabled"

    async def broadcast(self, msg: dict) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


hub = Hub()


@asynccontextmanager
async def lifespan(_app):
    def doctor():
        ok, msg = hub.tracer.handshake()
        hub.handshake = msg
        log.info("PRISM handshake: %s", msg)
    if hub.tracer.enabled:
        threading.Thread(target=doctor, daemon=True).start()
    log.info("AI provider: %s | voice codec: %s", hub.provider.name,
             "codec2-1200" if voice.HAVE_CODEC2 else "opus fallback")
    yield


app = FastAPI(title="Trail SOS", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/samples", StaticFiles(directory=ROOT / "samples"), name="samples")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
async def config():
    sim = MeshSim(demo_topology(), SimConfig())
    return {
        "nodes": [{"name": n.name, "x": n.x, "y": n.y} for n in sim.nodes],
        "links": sim.links(),
        "range_km": sim.cfg.range_km,
        "rate_presets": RATE_PRESETS,
        "provider": hub.provider.name,
        "tracing": hub.handshake,
        "codec": "codec2-1200" if voice.HAVE_CODEC2 else "opus-6k (fallback)",
        "src": SRC, "dst": DST,
    }


@app.post("/api/sos")
async def send_sos(req: SosRequest):
    if req.rate not in RATE_PRESETS:
        raise HTTPException(400, f"rate must be one of {list(RATE_PRESETS)}")
    if not 0.0 <= req.loss < 1.0:
        raise HTTPException(400, "loss must be in [0, 1)")
    if hub.task and not hub.task.done():
        hub.task.cancel()
    cfg = SimConfig(rate_bps=RATE_PRESETS[req.rate], loss=req.loss, seed=req.seed)
    sim = MeshSim(demo_topology(), cfg)
    payload = await asyncio.to_thread(build_sos, hub.provider)
    pending: list = []
    sim.listeners.append(pending.append)
    msg_id = sim.send_message(SRC, DST, payload.parts)
    session = f"sos-{msg_id:08x}"
    for r in payload.ai_results:
        hub.tracer.record_ai_call(session, r)
    tracker = StatusTracker(msg_id, cfg.rate_bps, cfg.ack_margin_s)
    run = Run(sim, tracker, RangerView(hub.provider), msg_id, session, max(req.speed, 0.1),
              payload.sizes(), payload.codec, req.knockout, req.knockout_after_rx, pending=pending)
    for ev in pending:
        tracker.on_event(ev)
    hub.run = run
    await hub.broadcast({"type": "start", "msg_id": f"{msg_id:08x}", "sizes": run.payload_sizes,
                         "codec": payload.codec,
                         "sender_ai": {r.kind: {"text": r.text(), "confidence": r.confidence, "labels": r.labels}
                                       for r in payload.ai_results},
                         "config": {"loss": req.loss, "rate_bps": cfg.rate_bps, "seed": req.seed}})
    run.pending.clear()
    hub.task = asyncio.create_task(drive(run))
    return {"msg_id": f"{msg_id:08x}", "sizes": run.payload_sizes, "codec": payload.codec}


@app.post("/api/node/{name}/{action}")
async def node_action(name: str, action: str):
    run = hub.run
    if run is None or name not in run.sim.by_name:
        raise HTTPException(404, "no such node / no active run")
    if name in (SRC, DST) and action == "down":
        raise HTTPException(400, "knock out a relay, not an endpoint")
    run.sim.set_alive(name, action == "up")
    return {"ok": True}


@app.post("/api/speed/{speed}")
async def set_speed(speed: float):
    if hub.run:
        hub.run.speed = max(0.1, speed)
    return {"speed": speed}


@app.get("/api/status")
async def status():
    if not hub.run:
        return {"state": "idle"}
    return {**hub.run.tracker.snapshot(), "ranger": hub.run.ranger.view, "sim_t": round(hub.run.sim.t, 2)}


@app.get("/api/media/{msg_id}/{name}")
async def media(msg_id: str, name: str):
    run = hub.run
    if not run or f"{run.msg_id:08x}" != msg_id or name not in run.ranger.media:
        raise HTTPException(404, "not received (yet)")
    data, mime = run.ranger.media[name]
    return Response(data, media_type=mime, headers={"Cache-Control": "no-store"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    hub.clients.add(ws)
    try:
        if hub.run:
            await ws.send_json(_tick_msg(hub.run, []))
        while True:
            await ws.receive_text()  # keepalive; controls go over HTTP
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(ws)


def _tick_msg(run: Run, events: list) -> dict:
    return {"type": "tick", "sim_t": round(run.sim.t, 2), "speed": run.speed,
            "events": [e.to_dict() for e in events],
            "status": run.tracker.snapshot(),
            "ranger": run.ranger.view,
            "alive": {n.name: n.alive for n in run.sim.nodes},
            "queues": {n.name: len(n.queue) + (1 if n.busy else 0) for n in run.sim.nodes}}


def _network_metadata(run: Run) -> dict:
    s = run.tracker.snapshot()
    return {"bytes_sent": s["bytes_on_air"], "packets": run.tracker.sent, "resends": s["resends"],
            "hops": s["hops_total"], "route": " -> ".join(s["route"]),
            "parity_recoveries": s["parity_recoveries"], "final_delivery_state": s["state"],
            "reroutes": s["reroutes"], "lost_frames": s["lost_frames"], "nacks": s["nacks"],
            "sim_elapsed_s": s["elapsed_s"], "rate_bps": run.sim.cfg.rate_bps, "loss": run.sim.cfg.loss,
            "seed": run.sim.cfg.seed, "msg_id": s["msg_id"]}


async def drive(run: Run) -> None:
    """Advance sim time at `speed`x wall clock, stream events + honest status."""
    try:
        while True:
            run.sim.run_until(run.sim.t + TICK_S * run.speed)
            events = list(run.pending)
            run.pending.clear()
            for ev in events:
                run.tracker.on_event(ev)
                if ev.msg_id == run.msg_id and ev.kind == "received":
                    run.rx_count += 1
                if ev.msg_id == run.msg_id and ev.kind == "reassembled":
                    data = run.sim.rx_parts[run.msg_id][ev.part]
                    ai = await asyncio.to_thread(run.ranger.update, ev.part, data)
                    if ai is not None:
                        hub.tracer.record_ai_call(run.session, ai)
            if run.knockout and not run.knocked and run.rx_count >= run.knockout_after_rx:
                run.knocked = True
                run.sim.set_alive(run.knockout, False)
                events += run.pending
                for ev in run.pending:
                    run.tracker.on_event(ev)
                run.pending.clear()
            await hub.broadcast(_tick_msg(run, events))
            if run.tracker.state in ("delivered", "failed"):
                run.finished = True
                n = hub.tracer.finalize(run.session, _network_metadata(run))
                await hub.broadcast({"type": "done", "status": run.tracker.snapshot(), "traces_queued": n,
                                     "tracing": hub.handshake})
                return
            if run.sim.idle:  # should not happen: the source always settles
                log.error("simulation went idle without a final state")
                return
            await asyncio.sleep(TICK_S)
    except asyncio.CancelledError:
        pass
