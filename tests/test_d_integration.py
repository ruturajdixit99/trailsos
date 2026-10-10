"""D. Integration tests: AI adapter, PRISM tracing, app without keys, HTTP+WebSocket smoke."""
import json
import os

import httpx
import pytest
from dotenv import load_dotenv

from ai.provider import (
    RECONSTRUCTION_LABEL, MockProvider, NicrronNinooProvider, ProviderNotConfigured, get_provider,
)
from media.text import CAPTION_BUDGET, unpack
from mesh.packet import ContentType
from observability.prism import AGENT_ID, PrismTracer
from server.sos import RangerView, build_sos
from tests.conftest import LIVE, ROOT
from tests.helpers import run_trial

load_dotenv(ROOT / ".env")
REQUIRED = {"project_id", "model", "input_messages", "output_message", "latency_ms"}


# 16 --------------------------------------------------------------------------
def test_mock_provider_end_to_end(sos_payload):
    t = run_trial(sos_payload.parts, seed=3)
    assert t.state == "delivered"
    rv = RangerView(MockProvider())
    for part, data in t.sim.rx_parts[t.msg_id].items():
        rv.update(part, data)
    sent_tr = sos_payload.ai_results[0].text()
    assert rv.view["transcript"]["text"] == sent_tr
    assert rv.view["caption"]["text"] == sos_payload.ai_results[1].text()
    assert RECONSTRUCTION_LABEL in rv.view["rebuild"]["labels"]
    assert rv.media["rebuild.png"][0][:8] == b"\x89PNG\r\n\x1a\n"
    assert rv.view["audio"]["seconds"] == pytest.approx(5.0, abs=0.05)


def test_nicrronninoo_without_config_falls_back(monkeypatch):
    for k in ("NICRRONNINOO_BASE_URL", "NICRRONNINOO_API_KEY", "NICRRONNINOO_MODEL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ProviderNotConfigured):
        NicrronNinooProvider()
    assert isinstance(get_provider(), MockProvider)


class FakeNicrron:
    """Mock of Nicrron's OpenAI-compatible gateway (chat vision, Whisper, async image jobs)."""

    def __init__(self):
        self.requests = []
        self.polls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.headers["Authorization"] == "Bearer sk-test"
        path = request.url.path
        if path == "/v1/chat/completions":
            body = json.loads(request.content)
            assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
            return httpx.Response(200, json={"choices": [{"message": {"content":
                '```json\n{"caption": "Hiker sitting on trail holding left ankle, orange jacket", '
                '"confidence": 0.81}\n```'}}]})
        if path == "/v1/audio/transcriptions":
            assert b"openai/whisper-1" in request.content
            return httpx.Response(200, json={"text": "Help, I hurt my ankle.", "segments": [
                {"start": 0, "end": 3, "avg_logprob": -0.1, "no_speech_prob": 0.01},
                {"start": 3, "end": 5, "avg_logprob": -0.9, "no_speech_prob": 0.05}]})
        if path == "/v1/images/generations":
            return httpx.Response(202, json={"id": "img-1", "status": "queued"})
        if path == "/v1/images/generations/img-1":
            self.polls += 1
            return httpx.Response(200, json={"id": "img-1", "status": "succeeded" if self.polls > 1 else "running"})
        if path == "/v1/images/generations/img-1/content":
            from PIL import Image
            buf = __import__("io").BytesIO()
            Image.new("RGB", (1024, 1024), (200, 200, 200)).save(buf, "PNG")
            return httpx.Response(200, content=buf.getvalue(), headers={"content-type": "image/png"})
        return httpx.Response(404, json={"error": path})


def test_nicrronninoo_adapter_against_mock_gateway(monkeypatch):
    import math
    import time as _time

    from PIL import Image
    monkeypatch.setenv("NICRRONNINOO_API_KEY", "sk-test")
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    fake = FakeNicrron()
    p = NicrronNinooProvider(transport=httpx.MockTransport(fake.handler))
    cap = p.run_caption(Image.open(ROOT / "samples" / "trail_photo.jpg"))
    assert cap.text().startswith("Hiker sitting") and cap.confidence == pytest.approx(0.81)
    assert cap.model == "anthropic/claude-haiku-4-5" and cap.labels == []
    tr = p.run_transcribe(b"RIFF....")
    want = math.exp((3 * -0.1 + 2 * -0.9) / 5) * (1 - (3 * 0.01 + 2 * 0.05) / 5)
    assert tr.confidence == pytest.approx(want) and tr.model == "openai/whisper-1"
    assert tr.labels == ["unverified"]          # 0.64 < 0.7: Whisper itself was unsure -> labeled
    rb = p.run_rebuild(cap.text(), cap.confidence)
    assert rb.output[:8] == b"\x89PNG\r\n\x1a\n" and RECONSTRUCTION_LABEL in rb.labels
    from PIL import Image as _I
    assert max(_I.open(__import__("io").BytesIO(rb.output)).size) <= 240   # downscaled + stamped
    assert rb.model == "google/nano-banana"


def test_nicrronninoo_failure_falls_back_to_mock(monkeypatch):
    from ai.provider import FallbackProvider
    monkeypatch.setenv("NICRRONNINOO_API_KEY", "sk-test")
    broken = NicrronNinooProvider(transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))
    fb = FallbackProvider(broken, MockProvider())
    r = fb.run_transcribe(b"x")
    assert r.text() == MockProvider().transcribe(b"")[0]
    assert "fallback" in r.model and fb.fallbacks and fb.fallbacks[0][0] == "transcribe"


@pytest.mark.live
@pytest.mark.skipif(not LIVE.get("NICRRONNINOO_API_KEY"), reason="no NicrronNinoo key in .env")
def test_nicrronninoo_live_budget(monkeypatch):
    """One real caption + transcript call (no image generation, to keep it cheap)."""
    for k, v in LIVE.items():
        if k.startswith("NICRRONNINOO_"):
            monkeypatch.setenv(k, v)
    p = NicrronNinooProvider()
    payload = build_sos(p)
    cap_text, cap_conf = unpack(payload.parts[ContentType.CAPTION])
    tr_text, tr_conf = unpack(payload.parts[ContentType.TRANSCRIPT])
    print(f"\nLIVE caption ({cap_conf:.2f}): {cap_text}\nLIVE transcript ({tr_conf:.2f}): {tr_text}")
    assert len(payload.parts[ContentType.CAPTION]) <= CAPTION_BUDGET
    assert cap_text and "ankle" in tr_text.lower()
    assert all("mock" not in r.model for r in payload.ai_results)


# 17 --------------------------------------------------------------------------
class FakePrism:
    def __init__(self):
        self.requests = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.headers.get("X-PRISMtrace-Key") != "pt-sk-test":
            return httpx.Response(401, json={"error": "bad key"})
        if request.url.path == "/api/setup-doctor/handshake":
            return httpx.Response(200, json={"ok": True})
        body = json.loads(request.content)
        missing = REQUIRED - body.keys()
        if missing:
            return httpx.Response(422, json={"missing": sorted(missing)})
        return httpx.Response(200, json={"id": "t1", "session_id": body.get("session_id")})


def test_prism_one_trace_per_ai_call(sos_payload):
    fake = FakePrism()
    tr = PrismTracer("https://prism.test", "11111111-2222-3333-4444-555555555555", "pt-sk-test",
                     transport=httpx.MockTransport(fake.handler))
    t = run_trial(sos_payload.parts, seed=1)
    session = f"sos-{t.msg_id:08x}"
    calls = list(sos_payload.ai_results)
    rv = RangerView(MockProvider())
    calls.append(rv.update("caption", t.sim.rx_parts[t.msg_id]["caption"]))
    for c in calls:
        tr.record_ai_call(session, c)
    snap = t.tracker.snapshot()
    network = {"bytes_sent": snap["bytes_on_air"], "packets": t.tracker.sent, "resends": snap["resends"],
               "hops": snap["hops_total"], "route": " -> ".join(snap["route"]),
               "parity_recoveries": snap["parity_recoveries"], "final_delivery_state": snap["state"]}
    assert tr.finalize(session, network) == 3
    tr.flush()
    tr.close()
    traces = [r for r in fake.requests if r.url.path == "/api/traces"]
    assert len(traces) == 3 and not tr.errors
    kinds = set()
    for r in traces:
        assert r.method == "POST"
        assert r.headers["X-PRISMtrace-Key"] == "pt-sk-test"
        assert "authorization" not in {k.lower() for k in r.headers}
        b = json.loads(r.content)
        assert REQUIRED <= b.keys()
        assert b["session_id"] == session and b["agent_id"] == AGENT_ID
        assert isinstance(b["input_messages"], list) and b["input_messages"][0]["role"] == "user"
        md = b["metadata"]
        for k in ("bytes_sent", "packets", "resends", "hops", "route", "parity_recoveries",
                  "final_delivery_state", "ai_confidence"):
            assert k in md
        assert md["final_delivery_state"] == "delivered"
        kinds.add(md["ai_call"])
    assert kinds == {"caption", "transcript", "rebuild"}


def test_prism_handshake_mock():
    fake = FakePrism()
    ok, msg = PrismTracer("https://prism.test", "p", "pt-sk-test",
                          transport=httpx.MockTransport(fake.handler)).handshake()
    assert ok and msg == "CREDENTIAL OK"
    ok, msg = PrismTracer("https://prism.test", "p", "pt-sk-wrong",
                          transport=httpx.MockTransport(fake.handler)).handshake()
    assert not ok and msg.startswith("HTTP 401")


@pytest.mark.live
@pytest.mark.skipif(not (LIVE.get("PRISMTRACE_API_KEY") and LIVE.get("PRISMTRACE_PROJECT_ID")),
                    reason="no PRISM credentials in .env")
def test_prism_handshake_live():
    ok, msg = PrismTracer(LIVE.get("PRISMTRACE_HOST"), LIVE["PRISMTRACE_PROJECT_ID"], LIVE["PRISMTRACE_API_KEY"]).handshake()
    print(f"\nPRISM live handshake: {msg}")
    assert ok, msg


# 18 --------------------------------------------------------------------------
def test_tracing_disabled_without_keys(caplog):
    tr = PrismTracer(project_id="", api_key="")
    assert not tr.enabled
    assert "tracing disabled" in caplog.text
    tr.record_ai_call("s", MockProvider().run_transcribe(b""))
    assert tr.finalize("s", {}) == 0
    assert tr.handshake()[0] is False


# 19 (+18: the app runs end to end with tracing disabled) --------------------
def _drive_app(body):
    from fastapi.testclient import TestClient

    from server import app as appmod
    appmod.hub.tracer = PrismTracer(project_id="", api_key="")
    appmod.hub.handshake = "tracing disabled"
    states, kinds = [], set()
    with TestClient(appmod.app) as client:
        assert client.get("/").status_code == 200 and client.get("/demo").status_code == 200
        assert client.get("/api/config").json()["nodes"]
        with client.websocket_connect("/ws") as ws:
            r = client.post("/api/sos", json=body)
            assert r.status_code == 200, r.text
            msg_id = r.json()["msg_id"]
            while True:
                m = ws.receive_json()
                if m["type"] == "tick":
                    states.append(m["status"]["state"])
                    kinds.update(e["kind"] for e in m["events"])
                elif m["type"] == "done":
                    final = m
                    break
        media = {n: client.get(f"/api/media/{msg_id}/{n}") for n in ("audio.wav", "thumbnail.png", "rebuild.png")}
    return states, kinds, final, media


def test_api_smoke_send_sos_to_delivered():
    states, kinds, final, media = _drive_app({"loss": 0.2, "rate": "fast", "seed": 7, "speed": 8})
    assert final["status"]["state"] == "delivered"
    assert final["tracing"] == "tracing disabled"
    assert "e2e_ack_received" in kinds
    i = states.index("delivered")
    assert i > 0 and set(states[:i]) <= {"queued", "in_transit", "partial"}
    assert set(states[i:]) == {"delivered"}
    assert all(r.status_code == 200 for r in media.values())
    assert media["audio.wav"].content[:4] == b"RIFF"


def test_api_knockout_reroutes():
    states, kinds, final, _ = _drive_app({"loss": 0.2, "rate": "fast", "seed": 3, "speed": 2, "knockout": "C"})
    assert "node_down" in kinds and "rerouted" in kinds
    assert final["status"]["state"] == "delivered"
    assert "E" in final["status"]["route"]


def test_api_lora_mode_streams_ledger():
    states, kinds, final, media = _drive_app({"radio": "lora", "preset": "short_turbo", "km": 0.6, "seed": 2,
                                              "speed": 6})
    assert final["status"]["state"] == "delivered"
    led = final["ledger"]
    assert led["totals"]["frames_on_air"] == led["totals"]["frames_delivered"] + led["totals"]["frames_lost"]
    assert led["links"] and led["chunks"]
    assert "received" in kinds


def test_lab_api_job_roundtrip():
    import time

    from fastapi.testclient import TestClient

    from server import app as appmod
    with TestClient(appmod.app) as client:
        assert "Transfer Lab" in client.get("/").text
        cat = client.get("/api/lab/variants").json()
        assert {v["key"] for v in cat["variants"]} >= {"text", "audio", "image"}
        assert client.post("/api/lab/run", json={"variants": ["nope"]}).status_code == 400
        job = client.post("/api/lab/run", json={"variants": ["text", "image"], "trials": 5, "sweep": "loss",
                                                "sweep_trials": 3}).json()["job"]
        for _ in range(200):
            j = client.get(f"/api/lab/job/{job}").json()
            if j["status"] != "running":
                break
            time.sleep(0.05)
        assert j["status"] == "done", j
        r = j["result"]
        assert [v["key"] for v in r["variants"]] == ["text", "image"] and len(r["trials"]) == 10
        assert len(r["sweep"]["points"]) == 6
