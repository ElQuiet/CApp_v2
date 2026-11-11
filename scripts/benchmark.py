"""Bandingkan hasil app dengan nilai acuan ImageJ (data/reference_imagej.csv).

Pakai:
  .venv\\Scripts\\python scripts\\benchmark.py "C:\\...\\KARAKTERISASI"  [--tag nama] [--overlays]

Folder dataset juga bisa diberikan lewat variabel lingkungan CA_DATASET.
Hasil per foto → output/benchmark_<tag>.csv ; overlay (opsional) → output/bench_<tag>/
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ca_core import analyze  # noqa: E402

REF = ROOT / "data" / "reference_imagej.csv"
DIAG = ["support", "fit_rms", "edge_width", "edge_contrast", "sensitivity", "method_spread", "asymmetry",
        "base_width", "apex_height", "surface_visible", "clutter", "glare_frac", "baseline_angle"]


def run_one(args):
    path, overlay_dir, name = args
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    t0 = time.time()
    r = analyze(img)
    out = dict(ok=r.ok, theta=r.theta, theta_left=r.theta_left, theta_right=r.theta_right,
               confidence=r.confidence, method=r.primary_method, seconds=round(time.time() - t0, 2),
               issues="; ".join(i.title for i in r.issues), roi=r.roi_suggestion)
    out.update({f"d_{k}": r.diagnostics.get(k, np.nan) for k in DIAG})
    for m in r.methods:
        out[f"m_{m.name.split()[0]}"] = m.theta_mean
    out.update({f"c_{k}": v for k, v in r.diagnostics.get("cand", {}).items()})
    out["chosen"] = r.diagnostics.get("chosen", np.nan)
    out["geom_theta"] = r.diagnostics.get("geom_theta", r.theta)
    out["model_err"] = r.diagnostics.get("model_err", np.nan)
    out["hyps"] = r.diagnostics.get("hyps", [])
    if overlay_dir and r.overlay is not None:
        ov = r.overlay
        if r.roi_suggestion is not None:  # crop di sekitar tetesan agar mudah diperiksa
            s = r.scale
            x0, y0, x1, y1 = [int(v * s) for v in r.roi_suggestion]
            x0, x1 = sorted([x0, x1]); y0, y1 = sorted([y0, y1])
            pad = int(1.2 * max(x1 - x0, 60))
            h, w = ov.shape[:2]
            ov = ov[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
        cv2.imwrite(str(Path(overlay_dir) / f"{name}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return out


def summarize(df: pd.DataFrame, label: str = "") -> dict:
    d = df[df["ref_ok"] == 1]
    err = (d["theta"] - d["theta_ref"]).abs()
    det = d["ok"].astype(bool)
    s = dict(n=len(d), detected=f"{det.mean():.0%}",
             MAE=round(float(err[det].mean()), 2), median=round(float(err[det].median()), 2),
             within5=f"{(err <= 5).mean():.0%}", within10=f"{(err <= 10).mean():.0%}")
    if label:
        print(label, s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default=os.environ.get("CA_DATASET"))
    ap.add_argument("--tag", default="current")
    ap.add_argument("--overlays", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    if not a.dataset:
        sys.exit("Berikan path folder KARAKTERISASI (argumen atau CA_DATASET).")
    ds = Path(a.dataset)
    ref = pd.read_csv(REF)
    odir = None
    if a.overlays:
        odir = ROOT / "output" / f"bench_{a.tag}"
        odir.mkdir(parents=True, exist_ok=True)
    names = [f"{i:03d}_{r['sample']}_{r['stage']}_t{r['titik']}".replace(" ", "") for i, r in ref.iterrows()]
    jobs = [(str(ds / p), odir, n) for p, n in zip(ref["image"], names)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_one, jobs, chunksize=2))
    hyp_rows = []
    for i, r in enumerate(res):
        for j, f in enumerate(r.pop("hyps")):
            hyp_rows.append(dict(name=names[i], sample=ref.at[i, "sample"], theta_ref=ref.at[i, "theta_ref"],
                                 ref_ok=ref.at[i, "ref_ok"], hyp=j, chosen=int(j == r["chosen"]), **f))
    df = pd.concat([ref, pd.DataFrame(res)], axis=1)
    df.insert(0, "name", names)
    df["err"] = df["theta"] - df["theta_ref"]
    out = ROOT / "output" / f"benchmark_{a.tag}.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    pd.DataFrame(hyp_rows).to_csv(ROOT / "output" / f"hyps_{a.tag}.csv", index=False)
    print(f"{len(df)} foto, {time.time() - t0:.0f} s → {out.relative_to(ROOT)}")
    summarize(df, "SEMUA      ")
    for lo, hi in ((0, 40), (40, 70), (70, 180)):
        summarize(df[(df.theta_ref >= lo) & (df.theta_ref < hi)], f"acuan {lo:>3}–{hi:<3}°")
    for st in df["stage"].unique():
        summarize(df[df.stage == st], f"{st:<11}")


if __name__ == "__main__":
    main()
