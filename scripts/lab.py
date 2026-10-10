"""Transfer lab from the terminal: compare speed and loss of text / audio / image.

    python scripts/lab.py                                   # text, audio, image, full SOS; simple model
    python scripts/lab.py --raw                             # + uncompressed audio and original photo
    python scripts/lab.py --radio lora --km 2.0 --trials 50
    python scripts/lab.py --sweep loss --csv out.csv --json out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesh.lab import SWEEPS, LabConfig, run_experiment, run_sweep  # noqa: E402
from mesh.sim import RATE_PRESETS  # noqa: E402


def fmt_s(s):
    if s is None:
        return "never"
    return f"{s:.1f} s" if s < 120 else f"{s / 60:.1f} min" if s < 7200 else f"{s / 3600:.1f} h"


def table(rows, cols):
    w = [max(len(h), *(len(str(r[k])) for r in rows)) for k, h in cols]
    print("  ".join(h.ljust(x) for (_, h), x in zip(cols, w)))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print("  ".join(str(r[k]).ljust(x) for (k, _), x in zip(cols, w)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="text,audio,image,sos")
    ap.add_argument("--raw", action="store_true", help="also send uncompressed audio and the original photo")
    ap.add_argument("--radio", choices=["simple", "lora"], default="simple")
    ap.add_argument("--loss", type=float, default=0.2)
    ap.add_argument("--rate", choices=list(RATE_PRESETS), default="long_range")
    ap.add_argument("--preset", default="long_fast")
    ap.add_argument("--km", type=float, default=1.8)
    ap.add_argument("--terrain", type=float, default=0.0)
    ap.add_argument("--knockout", help="relay that fails 15 s in (B, C, D or E)")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", choices=list(SWEEPS))
    ap.add_argument("--csv", help="write every trial to this CSV")
    ap.add_argument("--json", help="write the full result to this JSON")
    a = ap.parse_args()

    variants = a.variants.split(",") + (["audio_raw", "image_raw"] if a.raw else [])
    cfg = LabConfig(variants=tuple(variants), radio=a.radio, loss=a.loss, rate_bps=RATE_PRESETS[a.rate],
                    lora_preset=a.preset, km=a.km, terrain_db=a.terrain, trials=a.trials, seed=a.seed,
                    knockout=a.knockout)
    res = run_experiment(cfg, lambda f: print(f"\r  simulating {f:4.0%}", end="", file=sys.stderr))
    print(file=sys.stderr)

    print(f"\nSPEED AND LOSS - {a.trials} trials each, 4 hops, "
          + (f"LoRa {a.preset} {a.km} km hops" if a.radio == "lora" else f"{a.loss:.0%} loss/link @ {RATE_PRESETS[a.rate]:,.0f} bps"))
    rows = [{"label": v["label"], "size": f"{v['bytes']:,} B", "pkts": v["packets"],
             "first": fmt_s(v["t_first_p50"]), "p50": fmt_s(v["t_arrive"]["p50"]), "p90": fmt_s(v["t_arrive"]["p90"]),
             "intact": f"{v['intact_pct']:.0f}%", "lost": f"{v['frame_loss_pct']}%", "resends": v["resends"],
             "air": f"{v['air_bytes']:,} B", "x": f"{v['overhead_x']}x",
             "causes": ", ".join(f"{k} {n:.1f}" for k, n in v["lost_by"].items()) or "-"} for v in res["variants"]]
    table(rows, [("label", "content"), ("size", "size"), ("pkts", "packets"), ("first", "first chunk"),
                 ("p50", "has all (median)"), ("p90", "p90"), ("intact", "intact"), ("lost", "frames lost"),
                 ("resends", "resends"), ("air", "bytes on air"), ("x", "air/useful"), ("causes", "lost frames by cause")])

    if a.sweep:
        sw = run_sweep(cfg, a.sweep, max(3, a.trials // 3))
        print(f"\nSWEEP - median time until the ranger has everything, by {sw['label'].lower()}")
        keys = [v["key"] for v in res["variants"]]
        labels = {v["key"]: v["label"] for v in res["variants"]}
        srows = []
        for p in sw["points"]:
            r = {"x": f"{p['x']:g}{sw['unit']}"}
            for v in p["variants"]:
                r[v["key"]] = f"{fmt_s(v['t_p50'])} ({v['delivery_pct']:.0f}%)"
            srows.append(r)
        table(srows, [("x", sw["label"])] + [(k, labels[k]) for k in keys])
        print("(value = median time, brackets = delivered intact)")
        res["sweep"] = sw

    if a.csv:
        cols = [k for k in res["trials"][0] if k != "lost_by"]
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(res["trials"])
        print(f"\nwrote {len(res['trials'])} trials to {a.csv}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
