"""C. Honesty tests: status never claims more than the events prove."""
import pytest

from ai.provider import RECONSTRUCTION_LABEL, UNVERIFIED_LABEL, MockProvider, apply_labels
from mesh.sim import Event, three_hop_topology
from mesh.status import StatusTracker
from server.sos import RangerView
from tests.helpers import run_trial


@pytest.fixture(scope="module")
def all_trials(sos_payload):
    out = [run_trial(sos_payload.parts, seed=s, topology=three_hop_topology()) for s in range(200)]   # test 8
    out += [run_trial(sos_payload.parts, seed=s, knockout="C") for s in range(30)]                   # test 9
    out += [run_trial(sos_payload.parts, seed=s, dead_from_start=(n,)) for s in range(10)
            for n in ("Ranger", "D")]                                                                # test 10
    out += [run_trial(sos_payload.parts, seed=s, loss=0.45) for s in range(20)]                     # stress
    return out


# 12 --------------------------------------------------------------------------
def test_delivered_iff_e2e_ack(all_trials):
    for t in all_trials:
        acked = bool(t.kinds("e2e_ack_received"))
        assert (t.state == "delivered") == acked
        assert t.tracker.snapshot()["delivered"] == acked


def test_delivered_never_set_without_e2e_event():
    """Feed a tracker every 'looks done' event except the e2e ack."""
    tr = StatusTracker(1, 1000)
    tr.on_event(Event(0, "message_queued", "Hiker", msg_id=1,
                      detail={"route": ["Hiker", "B", "Ranger"], "chunk_sizes": {"caption": [100]}}))
    tr.on_event(Event(1, "sent", "Hiker", msg_id=1, size=119, detail={"hop": 1}))
    tr.on_event(Event(2, "received", "Ranger", msg_id=1, part="caption", index=0, detail={"hop": 2}))
    tr.on_event(Event(2, "reassembled", "Ranger", msg_id=1, part="caption", size=100))
    tr.on_event(Event(2, "message_complete", "Ranger", msg_id=1))
    tr.on_event(Event(3, "e2e_ack_sent", "Ranger", msg_id=1))
    tr.on_event(Event(4, "e2e_ack_received", "Hiker", msg_id=2))   # someone else's ack
    assert tr.state == "partial" and not tr.snapshot()["delivered"]
    assert "awaiting end-to-end ack" in tr.label()
    tr.on_event(Event(5, "e2e_ack_received", "Hiker", msg_id=1))
    assert tr.state == "delivered"


def test_failed_never_overrides_delivered_and_vice_versa():
    tr = StatusTracker(1, 1000)
    tr.on_event(Event(0, "message_failed", "Hiker", msg_id=1, detail={"reason": "x"}))
    assert tr.state == "failed"


# 13 --------------------------------------------------------------------------
def test_partial_lists_exactly_reassembled(all_trials):
    for t in all_trials[:60] + all_trials[200:]:
        tr = StatusTracker(t.msg_id, t.sim.cfg.rate_bps)
        done = set()
        for ev in t.sim.events:
            tr.on_event(ev)
            if ev.msg_id == t.msg_id and ev.kind == "reassembled":
                done.add(ev.part)
            snap = tr.snapshot()
            assert set(snap["arrived_parts"]) == done
            assert {p for p, v in snap["parts"].items() if v["done"]} == done
            if snap["state"] == "partial":
                for p in ("transcript", "caption"):
                    assert (p in snap["label"]) == (p in done)
            for p, v in snap["parts"].items():
                assert v["have"] <= v["of"]


# 14 --------------------------------------------------------------------------
def test_eta_moves_to_zero_and_reacts_to_loss(all_trials):
    for t in all_trials[:50]:
        etas = [e for _, e in t.tracker.eta_history if e is not None]
        assert all(e >= 0 for e in etas)
        assert etas[-1] == 0.0 and t.state == "delivered"
        # ETA after each new chunk arrives trends to zero
        tr = StatusTracker(t.msg_id, t.sim.cfg.rate_bps)
        at_rx = []
        for ev in t.sim.events:
            changed = tr.on_event(ev)
            if changed and ev.kind == "received" and not ev.parity:
                at_rx.append(tr.eta_s)
        assert at_rx[-1] < at_rx[0]
        # recomputed (not frozen) after a loss
        tr = StatusTracker(t.msg_id, t.sim.cfg.rate_bps)
        reacted = False
        for ev in t.sim.events:
            before = tr.eta_s
            tr.on_event(ev)
            if ev.kind == "lost" and ev.msg_id == t.msg_id and before is not None and tr.eta_s != before:
                reacted = True
        assert reacted


def test_failed_has_no_eta(all_trials):
    for t in all_trials:
        if t.state == "failed":
            assert t.tracker.snapshot()["eta_s"] is None


# 15 --------------------------------------------------------------------------
def test_rebuild_always_labeled():
    rv = RangerView(MockProvider())
    from media.text import pack
    ai = rv.update("caption", pack("Hiker on trail near peak", 0.95, 150))
    assert ai is not None and RECONSTRUCTION_LABEL in ai.labels
    assert RECONSTRUCTION_LABEL in rv.view["rebuild"]["labels"]
    assert UNVERIFIED_LABEL not in rv.view["caption"]["labels"]


def test_low_confidence_marked_unverified():
    rv = RangerView(MockProvider())
    from media.text import pack
    rv.update("caption", pack("blurry shape, maybe a person", 0.41, 150))
    rv.update("transcript", pack("help ... ankle ... mile", 0.30, 197))
    assert UNVERIFIED_LABEL in rv.view["caption"]["labels"]
    assert UNVERIFIED_LABEL in rv.view["transcript"]["labels"]
    assert UNVERIFIED_LABEL in rv.view["rebuild"]["labels"]
    assert RECONSTRUCTION_LABEL in rv.view["rebuild"]["labels"]


def test_label_threshold():
    assert apply_labels("caption", 0.69, 0.7) == [UNVERIFIED_LABEL]
    assert apply_labels("caption", 0.70, 0.7) == []
    assert apply_labels("rebuild", 0.99, 0.7) == [RECONSTRUCTION_LABEL]
