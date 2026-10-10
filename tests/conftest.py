import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os  # noqa: E402

from dotenv import dotenv_values  # noqa: E402

# Real credentials (if any) are kept aside for the explicit live tests only; every other test
# runs on the mock provider with tracing off, so `pytest` never spends money or sends traces.
LIVE = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v} if (ROOT / ".env").exists() else {}
for _k in ("NICRRONNINOO_API_KEY", "PRISMTRACE_API_KEY", "PRISMTRACE_PROJECT_ID"):
    os.environ[_k] = ""

from ai.provider import MockProvider  # noqa: E402
from server.sos import build_sos  # noqa: E402


@pytest.fixture(scope="session")
def sos_payload():
    return build_sos(MockProvider())
