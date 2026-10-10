"""E. Realistic LoRa radio mode + network ledger (sent vs received accounting)."""
import pytest

from mesh import lora
from mesh.ledger import NetworkLedger
from mesh.sim import MeshSim, SimConfig, demo_topology
from tests.helpers import run_trial


def lora_trial(parts, seed, knockout=None, **kw):
    sim = MeshSim(demo_topology(), SimConfig(radio="lora", seed=seed, **kw))
    msg = sim.send_message("Hiker", "Ranger", parts)
    led = NetworkLedger(msg)
    for ev in sim.events:
        led.on_event(ev)
    sim.listeners.append(led.on_event)
    if knockout:
        sim.run_until(15.0)
        sim.set_alive(knockout, False)
    sim.run()
    return sim, msg, led


# --- PHY ----------------------------------------------------------------------
def test_airtime_matches_semtech_calculator():
    # Semtech LoRa calculator: SF7, BW125, CR4/5, 8-symbol preamble, 20 B, explicit header, CRC on -> 56.58 ms
    assert lora.airtime_s(lora.LoRaPreset("x", 7, 125e3, 1, 8), 20) == pytest.approx(0.05658, abs=1e-5)
    # SF12/125 kHz uses low-data-rate optimisation (symbol 32.768 ms > 16 ms):
    # ceil((160-48+28+16)/(4*(12-2)))*5 + 8 = 28 payload symbols + 12.25 preamble = 40.25 symbols = 1318.9 ms
    assert lora.airtime_s(lora.LoRaPreset("x", 12, 125e3, 1, 8), 20) == pytest.approx(1.3189, abs=1e-3)


def test_preset_bitrates():
    assert lora.PRESETS["long_fast"].bitrate == pytest.approx(1074, abs=1)
    assert lora.PRESETS["short_turbo"].bitrate == pytest.approx(21875, abs=1)


def test_slower_presets_reach_farther():
    env = lora.RadioEnv()
    margins = {k: env.mean_snr(p, 3.0) - p.snr_floor_db for k, p in lora.PRESETS.items()}
    assert margins["long_slow"] > margins["long_fast"] > margins["medium_fast"] > margins["short_fast"] \
        > margins["short_turbo"]


# --- LoRa mesh behaviour -------------------------------------------------------
@pytest.fixture(scope="module")
def lora_runs(sos_payload):
    return [lora_trial(sos_payload.parts, s) for s in range(30)]


def test_lora_mode_delivers_and_shows_rf_effects(lora_runs, sos_payload):
    delivered = 0
    reasons = set()
    for sim, msg, _ in lora_runs:
        if sim.tx[msg].state == "delivered":
            delivered += 1
            got = sim.rx_parts[msg]
            assert all(got[k] == v for k, v in
                       {"transcript": sos_payload.parts[1], "audio": sos_payload.parts[3]}.items())
        reasons |= {e.detail["reason"] for e in sim.events if e.kind == "lost"}
        rx = [e for e in sim.events if e.kind == "received"]
        assert all("snr_db" in e.detail and "rssi_dbm" in e.detail for e in rx)
    assert delivered / len(lora_runs) >= 0.9
    assert {"collision", "weak signal"} <= reasons  # real RF failure modes actually happen


def test_lora_delivered_iff_e2e_ack(lora_runs):
    for sim, msg, _ in lora_runs:
        acked = any(e.kind == "e2e_ack_received" and e.msg_id == msg for e in sim.events)
        assert (sim.tx[msg].state == "delivered") == acked


def test_lora_knockout_reroutes(sos_payload):
    ok = 0
    for s in range(15):
        sim, msg, _ = lora_trial(sos_payload.parts, s, knockout="C")
        ok += sim.tx[msg].state == "delivered"
    assert ok >= 13


def test_split_horizon_never_bounces_back(lora_runs):
    for sim, msg, _ in lora_runs:
        got_from = {}
        for e in sim.events:
            if e.kind == "relayed" and e.msg_id == msg:
                got_from[(e.node, e.part, e.parity, e.index)] = e.peer
            if e.kind == "sent" and e.msg_id == msg and (e.node, e.part, e.parity, e.index) in got_from:
                # a relay's first send of a forwarded packet never goes straight back where it came from
                assert e.peer != got_from[(e.node, e.part, e.parity, e.index)] or e.peer in ("Hiker", "Ranger")


# --- ledger accounting ---------------------------------------------------------
def _check_ledger(sim, msg, led):
    snap = led.snapshot()
    for link in snap["links"]:
        assert link["frames"] == link["delivered"] + link["lost"], link
        assert link["acks_lost"] <= link["acks_sent"] <= link["delivered"]
    t = snap["totals"]
    assert t["frames_on_air"] == sum(1 for e in sim.events if e.msg_id == msg and e.kind in ("sent", "retried"))
    assert t["frames_lost"] == sum(1 for e in sim.events if e.msg_id == msg and e.kind == "lost" and e.part != "ack")
    assert t["dest_unique_chunks"] == sum(1 for e in sim.events if e.msg_id == msg and e.kind == "received")
    arrived = {c["label"] for c in snap["chunks"] if c["status"] == "arrived" and c["part"] not in ("nack", "e2e_ack")}
    received = {f"{e.part}{'-parity' if e.parity else ''}#{e.index}" for e in sim.events
                if e.msg_id == msg and e.kind == "received"}
    assert arrived == received


def test_ledger_balances_lora(lora_runs):
    for sim, msg, led in lora_runs:
        _check_ledger(sim, msg, led)


def test_ledger_balances_simple(sos_payload):
    for s in range(30):
        t = run_trial(sos_payload.parts, seed=s, knockout="C" if s % 3 == 0 else None)
        led = NetworkLedger(t.msg_id)
        for ev in t.sim.events:
            led.on_event(ev)
        _check_ledger(t.sim, t.msg_id, led)
