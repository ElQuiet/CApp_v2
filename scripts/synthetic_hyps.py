"""Buat data latih sintetis → output/hyps_synthetic.csv (format sama dengan benchmark.py).

Pakai:  .venv\\Scripts\\python scripts\\synthetic_hyps.py [jumlah]
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ca_core import analyze  # noqa: E402
from ca_core.synthetic import drop_image, variants  # noqa: E402


def one(args):
    i, theta, kw = args
    r = analyze(drop_image(theta, **kw))
    rows = []
    for j, f in enumerate(r.diagnostics.get("hyps", [])):
        rows.append(dict(name=f"syn_{i:03d}", sample="synthetic", theta_ref=theta, ref_ok=1, hyp=j,
                         chosen=int(j == r.diagnostics.get("chosen")), **f))
    return rows


def main(n: int):
    jobs = [(i, t, kw) for i, (t, kw) in enumerate(variants(n))]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        rows = [r for rs in ex.map(one, jobs) for r in rs]
    out = ROOT / "output" / "hyps_synthetic.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{n} tetesan sintetis, {len(rows)} hipotesis → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 80)
