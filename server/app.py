"""FastAPI app: runs the mesh simulation in (scaled) real time and streams it over WebSocket.

A *session* is one shared simulated mesh. Any phone can send text / voice / photo / video /
SOS to any other phone; many messages can be in flight at once. The page focuses one
message at a time (status, ledger, summary, sent-vs-received) and shows all of them in
the phones' inboxes and the message list.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ai.provider import get_provider
from media import voice
from mesh.lab import SWEEPS, VARIANT_BY_KEY, LabConfig, run_experiment, run_sweep, variant_catalog
from mesh.lora import PRESETS as LORA_PRESETS
from mesh.sim import RATE_PRESETS, MeshSim, SimConfig, demo_topology
from observability.prism import PrismTracer
from server.session import SCENARIOS, Message, Session
from server.sos import KIND_LABEL, KINDS, ROOT, build_message

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
    knockout_after_rx: int = 4              # ...once this many chunks have reached their destinations
    radio: str = "simple"                   # "simple" (fixed loss %) | "lora" (RF physics)
    preset: str = "long_fast"               # Meshtastic modem preset (lora)
    km: float = 1.8                         # hop length: km per map unit (lora)
    terrain_db: float = 0.0                 # extra obstruction loss (lora)


class SendRequest(SosRequest):
    src: str = "Hiker"
    dst: str = "Ranger"
    kind: str = "text"
    text: Optional[str] = None
    new_session: bool = False               # start a fresh network with these settings first


class ScenarioRequest(SosRequest):
    name: str = "everyone_to_ranger"


def _sim_config(req: SosRequest) -> SimConfig:
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


def _validate(req: SosRequest) -> None:
    if req.radio not in ("simple", "lora") or req.preset not in LORA_PRESETS:
        raise HTTPException(400, "radio must be simple|lora and preset a Meshtastic preset")
    if req.rate not in RATE_PRESETS:
        raise HTTPException(400, f"rate must be one of {list(RATE_PRESETS)}")
    if not 0.0 <= req.loss < 1.0:
        raise HTTPException(400, "loss must be in [0, 1)")


class Hub:
    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.session: Optional[Session] = None
        self.provider = get_provider()
        self.tracer = PrismTracer()
        self.handshake = "pending" if self.tracer.enabled else "tracing disabled"

    @property
    def run(self) -> Optional[Session]:  # backwards-compatible name
        return self.session

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
    """Home: the live phone-to-phone mesh demo."""
    return FileResponse(STATIC / "index.html")


@app.get("/demo")
async def demo():
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


JOBS: dict[str, dict] = {}
LAB_JOBS = JOBS


def _lab_config(req: LabRequest) -> LabConfig:
    return LabConfig(variants=tuple(req.variants), radio=req.radio, loss=req.loss,
                     rate_bps=RATE_PRESETS.get(req.rate, 1000.0), lora_preset=req.preset, km=req.km,
                     terrain_db=req.terrain_db, trials=req.trials, seed=req.seed, knockout=req.knockout or None,
                     deadline_s=req.deadline_s)


def _start_job(work_fn) -> str:
    job_id = uuid.uuid4().hex[:10]
    job = JOBS[job_id] = {"status": "running", "progress": 0.0, "stage": "starting", "result": None, "error": None}

    def run():
        try:
            def prog(f, stage):
                job["progress"], job["stage"] = round(f, 3), stage
            job["result"] = work_fn(prog)
            job["status"], job["progress"] = "done", 1.0
        except Exception as e:  # noqa: BLE001
            log.exception("job failed")
            job["status"], job["error"] = "error", str(e)

    threading.Thread(target=run, daemon=True).start()
    return job_id


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
    cfg = _lab_config(req)

    def work(prog):
        result = run_experiment(cfg, lambda f: prog(f * (0.7 if req.sweep else 1.0), "experiment"))
        if req.sweep:
            result["sweep"] = run_sweep(cfg, req.sweep, req.sweep_trials, lambda f: prog(0.7 + 0.3 * f, "sweep"))
        return result

    return {"job": _start_job(work)}


@app.get("/api/lab/job/{job_id}")
@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return job


# ------------------------------------------------------------------ quick check
@app.post("/api/quickcheck")
async def quickcheck():
    from mesh.quickcheck import run_quickcheck
    return {"job": _start_job(lambda prog: run_quickcheck(prog))}


# ------------------------------------------------------------------ config
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
        "kinds": {k: KIND_LABEL[k] for k in KINDS},
        "scenarios": {k: {"label": v["label"], "about": v["about"], "messages": len(v["sends"])}
                      for k, v in SCENARIOS.items()},
    }


@app.post("/api/links")
async def links(req: SosRequest):
    """Link budget for a radio setting, without sending anything (map preview)."""
    sim = MeshSim(demo_topology(), _sim_config(req))
    return {"radio": _radio_info(sim), "links": sim.link_info(), "route": sim.planned_route(SRC, DST)}


# ------------------------------------------------------------------ sessions & sending
async def _new_session(req: SosRequest, scenario: Optional[str] = None) -> Session:
    _validate(req)
    old = hub.session
    if old and old.task and not old.task.done():
        old.task.cancel()
    sim = MeshSim(demo_topology(), _sim_config(req))
    s = Session(sim, req.model_dump(), max(req.speed, 0.1), _radio_info(sim), req.knockout or None,
                req.knockout_after_rx, scenario=scenario)
    sim.listeners.append(s.pending.append)
    s.on_launch.append(lambda m: _on_launch(s, m))
    hub.session = s
    s.task = asyncio.create_task(drive(s))
    await hub.broadcast(_session_msg(s))
    return s


def _on_launch(s: Session, m: Message) -> None:
    for r in m.payload.ai_results:
        hub.tracer.record_ai_call(m.trace_session, r)


async def _build(kind: str, text: Optional[str]):
    return await asyncio.to_thread(build_message, kind, hub.provider, text)


def _check_pair(s: Session, src: str, dst: str, kind: str) -> None:
    names = {n.name for n in s.sim.nodes}
    if src not in names or dst not in names or src == dst:
        raise HTTPException(400, f"src and dst must be two different phones of {sorted(names)}")
    if kind not in KINDS:
        raise HTTPException(400, f"kind must be one of {KINDS}")


@app.post("/api/sos")
async def send_sos(req: SosRequest):
    """Fresh network + the classic Hiker -> Ranger SOS (voice + photo)."""
    _validate(req)
    payload = await _build("sos", None)
    s = await _new_session(req)
    m = s.launch(SRC, DST, "sos", payload, hub.provider)
    s.focus = m.msg_id
    await hub.broadcast(_focus_msg(s, m))
    return {"msg_id": m.hex, "sizes": payload.sizes(), "codec": payload.codec}


@app.post("/api/send")
async def send(req: SendRequest):
    """Any phone -> any phone. Adds to the running network (or starts one)."""
    s = hub.session
    if s is None or req.new_session:
        s = await _new_session(req)
    _check_pair(s, req.src, req.dst, req.kind)
    if not s.sim.by_name[req.src].alive:
        raise HTTPException(400, f"{req.src} is knocked out")
    payload = await _build(req.kind, req.text)
    m = s.launch(req.src, req.dst, req.kind, payload, hub.provider)
    s.focus = m.msg_id
    await hub.broadcast(_focus_msg(s, m))
    return {"msg_id": m.hex, "sizes": payload.sizes(), "n": m.number}


@app.post("/api/scenario")
async def scenario(req: ScenarioRequest):
    sc = SCENARIOS.get(req.name)
    if not sc:
        raise HTTPException(400, f"scenario must be one of {list(SCENARIOS)}")
    _validate(req)
    payloads = await asyncio.gather(*[_build(kind, text) for _, _, kind, _, text in sc["sends"]])
    s = await _new_session(req, scenario=req.name)
    first = None
    for (src, dst, kind, delay, _), payload in zip(sc["sends"], payloads):
        if delay <= 0:
            m = s.launch(src, dst, kind, payload, hub.provider)
            first = first or m
        else:  # staggered sends happen in simulated time
            s.sim.schedule(delay, s.launch, src, dst, kind, payload, hub.provider)
    if first:
        s.focus = first.msg_id
        await hub.broadcast(_focus_msg(s, first))
    return {"scenario": req.name, "messages": len(sc["sends"])}


@app.post("/api/focus/{msg_id}")
async def focus(msg_id: str):
    s = hub.session
    m = s.find(msg_id) if s else None
    if not m:
        raise HTTPException(404, "no such message")
    s.focus = m.msg_id
    await hub.broadcast(_focus_msg(s, m))
    await hub.broadcast(_tick_msg(s, [e for e in s.sim.events if e.msg_id in (m.msg_id, None)]))
    return {"ok": True}


@app.post("/api/rebuild/{msg_id}")
async def rebuild(msg_id: str):
    """On-demand AI sketch from a received caption (automatic only for SOS messages)."""
    s = hub.session
    m = s.find(msg_id) if s else None
    if not m or "caption" not in m.view.view:
        raise HTTPException(404, "no received caption for that message")
    m.rebuild_task = asyncio.create_task(_rebuild(m))
    return {"ok": True}


@app.post("/api/node/{name}/{action}")
async def node_action(name: str, action: str):
    s = hub.session
    if s is None or name not in s.sim.by_name:
        raise HTTPException(404, "no such node / no active network")
    s.sim.set_alive(name, action == "up")
    return {"ok": True}


@app.post("/api/speed/{speed}")
async def set_speed(speed: float):
    if hub.session:
        hub.session.speed = max(0.1, speed)
    return {"speed": speed}


@app.get("/api/status")
async def status():
    s = hub.session
    m = s.focused if s else None
    if not m:
        return {"state": "idle"}
    return {**m.tracker.snapshot(), "ranger": m.view.view, "sim_t": round(s.sim.t, 2),
            "ledger": m.ledger.snapshot(), "messages": [s.messages[i].brief() for i in s.order]}


@app.get("/api/media/{msg_id}/{name}")
async def media(msg_id: str, name: str):
    s = hub.session
    m = s.find(msg_id) if s else None
    if not m or name not in m.view.media:
        raise HTTPException(404, "not received (yet)")
    data, mime = m.view.media[name]
    return Response(data, media_type=mime, headers={"Cache-Control": "no-store"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    hub.clients.add(ws)
    try:
        s = hub.session
        if s:  # late joiner: replay the network and the focused message so far
            await ws.send_json(_session_msg(s))
            if s.focused:
                await ws.send_json(_focus_msg(s, s.focused))
            await ws.send_json(_tick_msg(s, [e for e in s.sim.events
                                             if e.msg_id in ((s.focus,) if s.focus else ()) or e.msg_id is None]))
        while True:
            await ws.receive_text()  # keepalive; controls go over HTTP
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(ws)


# ------------------------------------------------------------------ messages to the page
def _session_msg(s: Session) -> dict:
    return {"type": "session", "radio": s.radio, "links": s.sim.link_info(), "request": s.request,
            "speed": s.speed, "scenario": s.scenario,
            "scenario_label": SCENARIOS[s.scenario]["label"] if s.scenario else None}


def _focus_msg(s: Session, m: Message) -> dict:
    p = m.payload
    return {"type": "start", "msg_id": m.hex, "n": m.number, "src": m.src, "dst": m.dst, "kind": m.kind,
            "kind_label": KIND_LABEL[m.kind], "sizes": p.sizes(), "codec": p.codec, "video": p.video,
            # the sender sees exactly what goes on air (and whether it had to be shortened)
            "sender_ai": {r.kind: {**p.sent_text[r.kind], "confidence": r.confidence, "labels": r.labels,
                                   "model": r.model} for r in p.ai_results},
            "sender_text": p.sent_text.get("text"),
            "config": {"loss": s.sim.cfg.loss, "rate_bps": s.sim.cfg.rate_bps, "seed": s.sim.cfg.seed},
            "radio": s.radio, "links": s.sim.link_info(), "route": s.sim.planned_route(m.src, m.dst),
            "speed": s.speed, "request": s.request}


def _tick_msg(s: Session, events: list) -> dict:
    m = s.focused
    base = {"type": "tick", "sim_t": round(s.sim.t, 2), "speed": s.speed,
            "events": [e.to_dict() for e in events],
            "alive": {n.name: n.alive for n in s.sim.nodes},
            "queues": {n.name: len(n.queue) + (1 if n.busy else 0) for n in s.sim.nodes},
            "messages": [s.messages[i].brief() for i in s.order],
            "phones": s.phones(), "focus": m.hex if m else None}
    if m:
        base.update({"status": m.tracker.snapshot(), "ranger": m.view.view, "ledger": m.ledger.snapshot(),
                     "summary": s.summary(m), "compare": s.compare(m),
                     "msg": {"id": m.hex, "src": m.src, "dst": m.dst, "kind": m.kind, "n": m.number}})
    return base


def _network_metadata(s: Session, m: Message) -> dict:
    st = m.tracker.snapshot()
    return {"bytes_sent": st["bytes_on_air"], "packets": m.tracker.sent, "resends": st["resends"],
            "hops": st["hops_total"], "route": " -> ".join(st["route"]),
            "parity_recoveries": st["parity_recoveries"], "final_delivery_state": st["state"],
            "reroutes": st["reroutes"], "lost_frames": st["lost_frames"], "nacks": st["nacks"],
            "sim_elapsed_s": st["elapsed_s"], "rate_bps": s.sim.cfg.rate_bps, "loss": s.sim.cfg.loss,
            "seed": s.sim.cfg.seed, "msg_id": st["msg_id"], "src": m.src, "dst": m.dst, "kind": m.kind,
            "messages_in_flight": sum(1 for x in s.messages.values() if not x.done)}


async def _rebuild(m: Message) -> None:
    try:
        ai = await asyncio.to_thread(m.view.rebuild_from_caption)
        if ai is not None:
            hub.tracer.record_ai_call(m.trace_session, ai)
    except Exception as e:  # noqa: BLE001 - a failed sketch must never break the SOS
        log.warning("AI rebuild failed: %s", e)


async def _ranger_ai(m: Message, fn) -> None:
    """Run a receiver-side AI check off the sim loop and trace it."""
    try:
        ai = await asyncio.to_thread(fn)
        if ai is not None:
            hub.tracer.record_ai_call(m.trace_session, ai)
    except Exception as e:  # noqa: BLE001 - a failed check must never break the message
        log.warning("receiver AI check failed: %s", e)


async def _settle(s: Session, m: Message) -> None:
    """A message reached delivered/failed: let its AI checks land, then trace it and tell the page."""
    waiting = [t for t in [m.rebuild_task, *m.ai_tasks] if t]
    if waiting:
        await asyncio.wait(waiting, timeout=90)
    n = hub.tracer.finalize(m.trace_session, _network_metadata(s, m))
    if hub.session is not s:  # the network was replaced meanwhile: don't confuse the page
        return
    await hub.broadcast({"type": "done", "msg_id": m.hex, "status": m.tracker.snapshot(), "traces_queued": n,
                         "ledger": m.ledger.snapshot(), "summary": s.summary(m), "tracing": hub.handshake,
                         "focused": s.focus == m.msg_id})


def _on_event(s: Session, ev) -> None:
    targets = [s.messages[ev.msg_id]] if ev.msg_id in s.messages else (
        list(s.messages.values()) if ev.msg_id is None else [])
    for m in targets:
        m.tracker.on_event(ev)
        m.ledger.on_event(ev)
    if ev.msg_id not in s.messages:
        return
    m = s.messages[ev.msg_id]
    if ev.kind == "received":
        s.rx_count += 1
    if ev.kind == "reassembled":
        m.view.update(ev.part, s.sim.rx_parts[m.msg_id][ev.part], rebuild=False)
        raw = m.view.raw.keys()
        if ev.part == "caption" and m.kind == "sos":  # automatic sketch for the SOS only (it costs money)
            m.rebuild_task = asyncio.create_task(_rebuild(m))
        # verification as soon as both halves are here: transcript vs audio, caption vs image/video
        if ev.part in ("transcript", "audio") and {"transcript", "audio"} <= raw:
            m.ai_tasks.append(asyncio.create_task(_ranger_ai(m, m.view.verify_transcript)))
        if ev.part in ("caption", "thumbnail", "video") and "caption" in raw and ({"thumbnail", "video"} & raw):
            m.ai_tasks.append(asyncio.create_task(_ranger_ai(m, m.view.verify_caption)))


async def drive(s: Session) -> None:
    """Advance sim time at `speed`x wall clock; stream events + honest status for every message."""
    quiet = 0
    try:
        while True:
            s.sim.run_until(s.sim.t + TICK_S * s.speed)
            events = list(s.pending)
            s.pending.clear()
            for ev in events:
                _on_event(s, ev)
            if s.knockout and not s.knocked and s.rx_count >= s.knockout_after_rx:
                s.knocked = True
                s.sim.set_alive(s.knockout, False)
                extra = list(s.pending)
                s.pending.clear()
                for ev in extra:
                    _on_event(s, ev)
                events += extra
            for m in s.messages.values():
                if not m.done and m.tracker.state in ("delivered", "failed"):
                    m.done = True
                    asyncio.create_task(_settle(s, m))
            busy = events or any(not m.done for m in s.messages.values())
            quiet = 0 if busy else quiet + 1
            if busy or quiet % 10 == 1:  # idle network: refresh about once a second
                focus = s.focus
                shown = [e for e in events if e.msg_id is None or e.msg_id == focus or e.kind in
                         ("sent", "retried", "ack_sent", "lost")]  # every message animates on the map
                await hub.broadcast(_tick_msg(s, shown))
            await asyncio.sleep(TICK_S)
    except asyncio.CancelledError:
        pass
