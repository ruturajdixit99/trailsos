"""FastAPI app: runs the mesh simulation in (scaled) real time and streams it over WebSocket."""
from __future__ import annotations

import asyncio
import logging
import threading
import uuid
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
from mesh.lab import SWEEPS, VARIANT_BY_KEY, LabConfig, run_experiment, run_sweep, variant_catalog
from mesh.ledger import NetworkLedger
from mesh.lora import PRESETS as LORA_PRESETS
from mesh.status import StatusTracker
from mesh.summary import summarize
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
    radio: str = "simple"                   # "simple" (fixed loss %) | "lora" (RF physics)
    preset: str = "long_fast"               # Meshtastic modem preset (lora)
    km: float = 1.8                         # hop length: km per map unit (lora)
    terrain_db: float = 0.0                 # extra obstruction loss (lora)


def _sim_config(req: "SosRequest") -> SimConfig:
    return SimConfig(rate_bps=RATE_PRESETS.get(req.rate, 1000.0), loss=req.loss, seed=req.seed, radio=req.radio,
                     lora_preset=req.preset, km_per_unit=req.km, terrain_db=req.terrain_db)


def _radio_info(sim: MeshSim) -> dict:
    if sim.lora:
        p = sim.preset
        return {"model": "lora", "preset": p.name, "sf": p.sf, "bw_khz": p.bw_hz / 1e3, "cr": f"4/{4 + p.cr}",
                "bitrate": round(p.bitrate), "snr_floor_db": p.snr_floor_db,
                "airtime_ms_219B": round(sim.airtime(219) * 1000), "airtime_ms_ack": round(sim.airtime(20) * 1000),
                "km_per_unit": sim.cfg.km_per_unit, "terrain_db": sim.cfg.terrain_db}
    return {"model": "simple", "bitrate": sim.cfg.rate_bps, "loss": sim.cfg.loss,
            "airtime_ms_219B": round(sim.airtime(219) * 1000)}


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
    ledger: Optional[NetworkLedger] = None
    radio: dict = field(default_factory=dict)
    start_msg: dict = field(default_factory=dict)
    rebuild_task: Optional[asyncio.Task] = None
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
    """Home: the Transfer Lab (compare speed and loss of text / audio / image)."""
    return FileResponse(STATIC / "lab.html")


@app.get("/demo")
async def demo():
    """Live phone-to-phone demo of a single SOS."""
    return FileResponse(STATIC / "index.html")


# ------------------------------------------------------------------ transfer lab
class LabRequest(BaseModel):
    variants: list[str] = ["text", "audio", "image", "sos"]
    radio: str = "simple"
    loss: float = 0.2
    rate: str = "long_range"
    preset: str = "long_fast"
    km: float = 1.8
    terrain_db: float = 0.0
    trials: int = 30
    seed: int = 0
    knockout: Optional[str] = None
    deadline_s: float = 3600.0
    sweep: Optional[str] = None         # loss | km | terrain_db
    sweep_trials: int = 12


LAB_JOBS: dict[str, dict] = {}


def _lab_config(req: LabRequest) -> LabConfig:
    return LabConfig(variants=tuple(req.variants), radio=req.radio, loss=req.loss,
                     rate_bps=RATE_PRESETS.get(req.rate, 1000.0), lora_preset=req.preset, km=req.km,
                     terrain_db=req.terrain_db, trials=req.trials, seed=req.seed, knockout=req.knockout or None,
                     deadline_s=req.deadline_s)


@app.get("/lab")
async def lab_page():
    return FileResponse(STATIC / "lab.html")


@app.get("/api/lab/variants")
async def lab_variants():
    return {"variants": await asyncio.to_thread(variant_catalog), "sweeps": SWEEPS,
            "lora_presets": {k: {"name": p.name, "bitrate": round(p.bitrate)} for k, p in LORA_PRESETS.items()},
            "rate_presets": RATE_PRESETS}


@app.post("/api/lab/run")
async def lab_run(req: LabRequest):
    bad = [v for v in req.variants if v not in VARIANT_BY_KEY]
    if bad or not req.variants:
        raise HTTPException(400, f"unknown or empty variants: {bad}")
    if not 1 <= req.trials <= 300 or not 0 <= req.loss < 1 or req.sweep not in (None, *SWEEPS):
        raise HTTPException(400, "trials 1-300, loss in [0,1), sweep one of " + ", ".join(SWEEPS))
    job_id = uuid.uuid4().hex[:10]
    job = LAB_JOBS[job_id] = {"status": "running", "progress": 0.0, "stage": "experiment", "result": None,
                              "error": None}
    cfg = _lab_config(req)

    def work():
        try:
            def prog(f, stage):
                job["progress"], job["stage"] = round(f, 3), stage
            result = run_experiment(cfg, lambda f: prog(f * (0.7 if req.sweep else 1.0), "experiment"))
            if req.sweep:
                result["sweep"] = run_sweep(cfg, req.sweep, req.sweep_trials,
                                            lambda f: prog(0.7 + 0.3 * f, "sweep"))
            job["result"], job["status"], job["progress"] = result, "done", 1.0
        except Exception as e:  # noqa: BLE001
            log.exception("lab job failed")
            job["status"], job["error"] = "error", str(e)

    threading.Thread(target=work, daemon=True).start()
    return {"job": job_id}


@app.get("/api/lab/job/{job_id}")
async def lab_job(job_id: str):
    job = LAB_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return job


@app.get("/api/config")
async def config():
    sim = MeshSim(demo_topology(), SimConfig())
    return {
        "nodes": [{"name": n.name, "x": n.x, "y": n.y} for n in sim.nodes],
        "links": sim.links(),
        "range_km": sim.cfg.range_km,
        "rate_presets": RATE_PRESETS,
        "lora_presets": {k: {"name": p.name, "sf": p.sf, "bw_khz": p.bw_hz / 1e3, "bitrate": round(p.bitrate)}
                         for k, p in LORA_PRESETS.items()},
        "provider": hub.provider.name,
        "tracing": hub.handshake,
        "codec": "codec2-1200" if voice.HAVE_CODEC2 else "opus-6k (fallback)",
        "src": SRC, "dst": DST,
    }


@app.post("/api/sos")
async def send_sos(req: SosRequest):
    if req.radio not in ("simple", "lora") or req.preset not in LORA_PRESETS:
        raise HTTPException(400, "radio must be simple|lora and preset a Meshtastic preset")
    if req.rate not in RATE_PRESETS:
        raise HTTPException(400, f"rate must be one of {list(RATE_PRESETS)}")
    if not 0.0 <= req.loss < 1.0:
        raise HTTPException(400, "loss must be in [0, 1)")
    if hub.task and not hub.task.done():
        hub.task.cancel()
    sim = MeshSim(demo_topology(), _sim_config(req))
    cfg = sim.cfg
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
    run.ledger = NetworkLedger(msg_id, SRC, DST)
    for ev in pending:
        tracker.on_event(ev)
        run.ledger.on_event(ev)
    hub.run = run
    run.radio = _radio_info(sim)
    run.start_msg = {"type": "start", "msg_id": f"{msg_id:08x}", "sizes": run.payload_sizes,
                         "codec": payload.codec,
                         # the hiker sees exactly what goes on air (and whether it had to be shortened)
                         "sender_ai": {r.kind: {**payload.sent_text[r.kind], "confidence": r.confidence,
                                                "labels": r.labels, "model": r.model}
                                       for r in payload.ai_results},
                         "config": {"loss": req.loss, "rate_bps": cfg.rate_bps, "seed": req.seed},
                         "radio": _radio_info(sim), "links": sim.link_info(),
                         "route": sim.planned_route(SRC, DST),
                         "speed": run.speed, "request": req.model_dump()}
    await hub.broadcast(run.start_msg)
    run.pending.clear()
    hub.task = asyncio.create_task(drive(run))
    return {"msg_id": f"{msg_id:08x}", "sizes": run.payload_sizes, "codec": payload.codec}


@app.post("/api/links")
async def links(req: SosRequest):
    """Link budget for a radio setting, without sending anything (map preview)."""
    sim = MeshSim(demo_topology(), _sim_config(req))
    return {"radio": _radio_info(sim), "links": sim.link_info(), "route": sim.planned_route(SRC, DST)}


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
    return {**hub.run.tracker.snapshot(), "ranger": hub.run.ranger.view, "sim_t": round(hub.run.sim.t, 2),
            "ledger": hub.run.ledger.snapshot()}


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
            # late joiner: replay the run so far (radio settings, links, every event)
            await ws.send_json(hub.run.start_msg)
            await ws.send_json(_tick_msg(hub.run, [e for e in hub.run.sim.events if e.msg_id in (hub.run.msg_id, None)]))
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
            "ledger": run.ledger.snapshot() if run.ledger else None,
            "summary": _summary(run),
            "alive": {n.name: n.alive for n in run.sim.nodes},
            "queues": {n.name: len(n.queue) + (1 if n.busy else 0) for n in run.sim.nodes}}


def _summary(run: Run) -> Optional[dict]:
    if not run.ledger:
        return None
    return summarize(run.sim, run.tracker, run.ledger, run.payload_sizes, run.radio)


def _network_metadata(run: Run) -> dict:
    s = run.tracker.snapshot()
    return {"bytes_sent": s["bytes_on_air"], "packets": run.tracker.sent, "resends": s["resends"],
            "hops": s["hops_total"], "route": " -> ".join(s["route"]),
            "parity_recoveries": s["parity_recoveries"], "final_delivery_state": s["state"],
            "reroutes": s["reroutes"], "lost_frames": s["lost_frames"], "nacks": s["nacks"],
            "sim_elapsed_s": s["elapsed_s"], "rate_bps": run.sim.cfg.rate_bps, "loss": run.sim.cfg.loss,
            "seed": run.sim.cfg.seed, "msg_id": s["msg_id"]}


async def _rebuild(run: Run) -> None:
    try:
        ai = await asyncio.to_thread(run.ranger.rebuild_from_caption)
        if ai is not None:
            hub.tracer.record_ai_call(run.session, ai)
    except Exception as e:  # noqa: BLE001 - a failed sketch must never break the SOS
        log.warning("AI rebuild failed: %s", e)


async def drive(run: Run) -> None:
    """Advance sim time at `speed`x wall clock, stream events + honest status."""
    try:
        while True:
            run.sim.run_until(run.sim.t + TICK_S * run.speed)
            events = list(run.pending)
            run.pending.clear()
            for ev in events:
                run.tracker.on_event(ev)
                run.ledger.on_event(ev)
                if ev.msg_id == run.msg_id and ev.kind == "received":
                    run.rx_count += 1
                if ev.msg_id == run.msg_id and ev.kind == "reassembled":
                    data = run.sim.rx_parts[run.msg_id][ev.part]
                    run.ranger.update(ev.part, data, rebuild=False)
                    if ev.part == "caption":  # slow AI sketch: off the sim loop so the mesh keeps moving
                        run.rebuild_task = asyncio.create_task(_rebuild(run))
            if run.knockout and not run.knocked and run.rx_count >= run.knockout_after_rx:
                run.knocked = True
                run.sim.set_alive(run.knockout, False)
                events += run.pending
                for ev in run.pending:
                    run.tracker.on_event(ev)
                    run.ledger.on_event(ev)
                run.pending.clear()
            await hub.broadcast(_tick_msg(run, events))
            if run.tracker.state in ("delivered", "failed"):
                run.finished = True
                if run.rebuild_task:  # let the sketch land so its trace carries the final network state
                    await asyncio.wait([run.rebuild_task], timeout=90)
                    await hub.broadcast(_tick_msg(run, []))
                n = hub.tracer.finalize(run.session, _network_metadata(run))
                await hub.broadcast({"type": "done", "status": run.tracker.snapshot(), "traces_queued": n,
                                     "ledger": run.ledger.snapshot(), "summary": _summary(run),
                                     "tracing": hub.handshake})
                return
            if run.sim.idle:  # should not happen: the source always settles
                log.error("simulation went idle without a final state")
                return
            await asyncio.sleep(TICK_S)
    except asyncio.CancelledError:
        pass
