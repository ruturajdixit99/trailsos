"""Block Convey PRISM tracing (https://blockconvey.com/docs).

POST {PRISMTRACE_HOST}/api/traces  with header  X-PRISMtrace-Key: pt-sk-...
One trace per AI call (caption, transcript, rebuild).  session_id = SOS message
id, agent_id = trail-sos, network facts in metadata.  AI calls are recorded as
they happen and sent once the message settles (delivered/failed), so the
metadata holds the final network numbers.  Sending happens on a background
worker thread: tracing never blocks the mesh.  No keys -> "tracing disabled".
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import uuid
from typing import Optional

import httpx

log = logging.getLogger("trail_sos.prism")
AGENT_ID = "trail-sos"
DEFAULT_HOST = "https://prism.blockconvey.com"


class PrismTracer:
    def __init__(self, host: str | None = None, project_id: str | None = None, api_key: str | None = None,
                 transport: Optional[httpx.BaseTransport] = None, timeout: float = 10.0):
        self.host = (host if host is not None else os.getenv("PRISMTRACE_HOST", DEFAULT_HOST)).rstrip("/")
        self.project_id = project_id if project_id is not None else os.getenv("PRISMTRACE_PROJECT_ID", "")
        self.api_key = api_key if api_key is not None else os.getenv("PRISMTRACE_API_KEY", "")
        self.enabled = bool(self.host and self.project_id and self.api_key)
        self._pending: dict[str, list] = {}
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        self.sent: list[dict] = []
        self.errors: list[str] = []
        if not self.enabled:
            log.warning("PRISM keys missing - tracing disabled")
            return
        self._client = httpx.Client(base_url=self.host, transport=transport, timeout=timeout,
                                    headers={"X-PRISMtrace-Key": self.api_key,
                                             "Content-Type": "application/json"})
        self._worker = threading.Thread(target=self._run, name="prism-tracer", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ recording
    def record_ai_call(self, session_id: str, result) -> None:
        """Remember an AI call (ai.provider.AIResult) for this SOS message."""
        if not self.enabled:
            return
        with self._lock:
            self._pending.setdefault(session_id, []).append(result)

    def finalize(self, session_id: str, network: dict) -> int:
        """Message settled: enqueue one trace per recorded AI call. Returns count."""
        if not self.enabled:
            return 0
        with self._lock:
            calls = self._pending.pop(session_id, [])
        for r in calls:
            self._q.put(self.build_trace(session_id, r, network))
        return len(calls)

    def build_trace(self, session_id: str, r, network: dict) -> dict:
        output = r.output if isinstance(r.output, str) else f"<image png {len(r.output)} bytes>"
        return {
            "project_id": self.project_id,
            "model": r.model,
            "input_messages": [{"role": "user", "content": f"[{r.kind}] {r.input_summary}"}],
            "output_message": output,
            "latency_ms": int(r.latency_ms),
            "session_id": session_id,
            "agent_id": AGENT_ID,
            "agent_name": "Trail SOS",
            "trace_id": f"{session_id}-{r.kind}-{uuid.uuid4().hex[:8]}",
            "metadata": {
                "ai_call": r.kind,
                "ai_confidence": round(float(r.confidence), 3),
                "ai_labels": list(r.labels),
                **network,
            },
        }

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            try:
                resp = self._client.post("/api/traces", json=item)
                if resp.status_code >= 400:
                    self.errors.append(f"{resp.status_code}: {resp.text[:200]}")
                    log.warning("PRISM trace rejected %s: %s", resp.status_code, resp.text[:200])
                else:
                    self.sent.append(item)
            except Exception as e:  # noqa: BLE001 - tracing must never crash the app
                self.errors.append(str(e))
                log.warning("PRISM trace failed: %s", e)
            finally:
                self._q.task_done()

    def flush(self, timeout: float = 10.0) -> None:
        if self.enabled:
            done = threading.Event()
            threading.Thread(target=lambda: (self._q.join(), done.set()), daemon=True).start()
            done.wait(timeout)

    def close(self) -> None:
        if self.enabled:
            self.flush()
            self._q.put(None)
            self._client.close()

    # ------------------------------------------------------------------ handshake
    def handshake(self, send_test_trace: bool = False) -> tuple[bool, str]:
        if not self.enabled:
            return False, "tracing disabled (PRISMTRACE_PROJECT_ID / PRISMTRACE_API_KEY not set)"
        try:
            r = self._client.post("/api/setup-doctor/handshake",
                                  json={"project_id": self.project_id, "send_test_trace": send_test_trace})
        except Exception as e:  # noqa: BLE001
            return False, f"handshake error: {e}"
        if r.status_code == 200:
            return True, "CREDENTIAL OK"
        meaning = {401: "missing, invalid or revoked key", 403: "key and project belong to different projects",
                   404: "project not found"}.get(r.status_code, "unexpected status")
        return False, f"HTTP {r.status_code} ({meaning}): {r.text[:300]}"
