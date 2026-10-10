"""Quick check from the terminal: the CHRP track requirements as live tests.

    python scripts/quickcheck.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesh.quickcheck import run_quickcheck  # noqa: E402


def main() -> int:
    r = run_quickcheck(lambda f, title: print(f"\r  running {f:4.0%}  {title[:50]:50s}", end="", file=sys.stderr))
    print(file=sys.stderr)
    for c in r["checks"]:
        print(f"{'PASS' if c['passed'] else 'FAIL'}  {c['title']}: {c['value']}")
        print(f"      requirement: {c['requirement']}")
        print(f"      {c['detail']}".replace("→", "->").replace("·", "|"))
    print(f"\n{r['passed']}/{r['total']} passed in {r['seconds']} s (AI: {r['ai']})")
    return 0 if r["passed"] == r["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
