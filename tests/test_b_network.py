"""B. Network / reliability tests (seeded)."""
import statistics

import pytest

from mesh.sim import MeshSim, NodeSpec, SimConfig, three_hop_topology
from tests.helpers import run_trial

TRIALS = 200


# 7 ---------------------------------------------------------------------------
def test_loss_model_20pct():
    sim = MeshSim([NodeSpec("A", 0, 0), NodeSpec("B", 1, 0)], SimConfig(loss=0.2, seed=42))
    a, b = sim.nodes
    lost = sum(1 for _ in range(10_000) if not sim._channel_ok(a, b)[0])
    assert abs(lost / 10_000 - 0.20) <= 0.015


def test_range_defines_links():
    sim = MeshSim()
    links = {tuple(sorted(l)) for l in sim.links()}
    assert ("B", "Hiker") in links and ("B", "E") in links and ("D", "E") in links
    assert ("Hiker", "Ranger") not in links and ("B", "D") not in links


# 8 ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def headline(sos_payload):
    return [run_trial(sos_payload.parts, seed=s, loss=0.2, topology=three_hop_topology()) for s in range(TRIALS)]


def test_headline_99pct_complete_and_identical(headline, sos_payload):
    ok = [t for t in headline if t.state == "delivered" and t.identical(sos_payload.parts)]
    rate = len(ok) / len(headline)
    resends = statistics.mean(t.tracker.resends for t in headline)
    times = [t.tracker.t_done - t.tracker.t_start for t in ok]
    print(f"\nheadline: {len(ok)}/{len(headline)} delivered byte-identical ({rate:.1%}); "
          f"avg resends {resends:.1f}; mean delivery {statistics.mean(times):.1f}s, "
          f"median {statistics.median(times):.1f}s (3 hops, 20% loss, 1000 bps)")
    assert rate >= 0.99


# 9 ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def knockouts(sos_payload):
    return [run_trial(sos_payload.parts, seed=s, loss=0.2, knockout="C") for s in range(30)]


def test_knockout_reroutes_through_e(knockouts, sos_payload):
    for t in knockouts:
        assert t.kinds("node_down"), "C was never knocked out"
        assert t.state == "delivered", t.tracker.label()
        assert t.identical(sos_payload.parts)
        paths = [e.detail["path"] for e in t.kinds("received")]
        assert any("E" in p for p in paths), "nothing arrived via E"
        assert any(e.kind == "rerouted" for e in t.sim.events)


# 10 --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def unreachable(sos_payload):
    out = []
    for s in range(10):
        out.append(run_trial(sos_payload.parts, seed=s, dead_from_start=("Ranger",)))
        out.append(run_trial(sos_payload.parts, seed=s, dead_from_start=("D",)))
    return out


def test_unreachable_ends_failed_with_reason(unreachable):
    for t in unreachable:
        assert t.state == "failed"
        assert not t.kinds("e2e_ack_received")
        assert "No end-to-end ack" in t.tracker.reason
        assert "unreachable" in t.tracker.reason or "no route" in t.tracker.reason


# 11 --------------------------------------------------------------------------
def _reassembled_times(t):
    return {e.part: e.t for e in t.kinds("reassembled")}


def test_priority_no_loss(sos_payload):
    t = run_trial(sos_payload.parts, seed=0, loss=0.0)
    r = _reassembled_times(t)
    assert max(r["transcript"], r["caption"]) < min(r["audio"], r["thumbnail"])
    assert r["transcript"] < r["caption"]


def test_priority_under_loss(headline):
    good = 0
    for t in headline:
        r = _reassembled_times(t)
        if len(r) == 4 and max(r["transcript"], r["caption"]) < min(r["audio"], r["thumbnail"]):
            good += 1
    print(f"\npriority held in {good}/{len(headline)} trials at 20% loss")
    assert good / len(headline) >= 0.95
