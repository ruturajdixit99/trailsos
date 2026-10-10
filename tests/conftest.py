import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.provider import MockProvider  # noqa: E402
from server.sos import build_sos  # noqa: E402


@pytest.fixture(scope="session")
def sos_payload():
    return build_sos(MockProvider())
