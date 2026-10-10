"""F. Transfer lab: experiments are honest summaries of real simulator runs."""
from mesh.lab import LabConfig, run_experiment, run_sweep, variant_catalog


def test_catalog_sizes():
    cat = {v["key"]: v for v in variant_catalog()}
    assert cat["audio"]["bytes"] <= 800 and cat["image"]["bytes"] <= 800
    assert cat["audio_raw"]["bytes"] == 40000 and cat["image_raw"]["bytes"] > 10000


def test_experiment_simple_model():
    r = run_experiment(LabConfig(variants=("text", "audio", "audio_raw"), trials=8))
    by = {v["key"]: v for v in r["variants"]}
    assert len(r["trials"]) == 24
    assert all(v["delivery_pct"] == 100 and v["intact_pct"] == 100 for v in by.values())
    # smaller payload arrives sooner; compression is the big win
    assert by["text"]["t_arrive"]["p50"] < by["audio"]["t_arrive"]["p50"] < by["audio_raw"]["t_arrive"]["p50"]
    assert by["audio_raw"]["t_arrive"]["p50"] > 10 * by["audio"]["t_arrive"]["p50"]
    for v in by.values():
        assert v["curve"][0] == 0 and v["curve"][-1] == 100          # progress curve ends complete
        assert all(a <= b for a, b in zip(v["curve"], v["curve"][1:]))  # and never goes backwards
        assert 10 <= v["frame_loss_pct"] <= 30                       # 20% configured loss
    for t in r["trials"]:
        assert t["frames"] >= t["frames_lost"] and (t["delivered"] or t["t_confirm"] is None)


def test_experiment_reports_failures_honestly():
    r = run_experiment(LabConfig(variants=("text",), trials=4, knockout="D", knockout_at=0.0, deadline_s=2000))
    v = r["variants"][0]
    assert v["delivery_pct"] == 0 and v["intact_pct"] == 0 and v["t_arrive"]["p50"] is None
    assert all(t["state"] != "delivered" for t in r["trials"])


def test_sweep_loss_gets_slower():
    sw = run_sweep(LabConfig(variants=("audio",)), "loss", trials=6)
    t = [p["variants"][0]["t_p50"] for p in sw["points"]]
    assert t[0] < t[-1] and len(t) == 6
