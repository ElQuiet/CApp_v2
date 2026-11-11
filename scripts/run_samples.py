"""Jalankan analisis pada semua foto di samples/ dan simpan overlay ke output/."""
import glob
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ca_core import analyze  # noqa: E402

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(root, "output")
os.makedirs(out_dir, exist_ok=True)
files = sys.argv[1:] or sorted(glob.glob(os.path.join(root, "samples", "drop_*.jpg")))
for f in files:
    t0 = time.time()
    r = analyze(cv2.imread(f))
    dt = time.time() - t0
    name = os.path.basename(f)
    print(f"\n=== {name}  ({dt:.1f}s)  ok={r.ok}  θ={r.theta:.1f}  L={r.theta_left:.1f}  R={r.theta_right:.1f}"
          f"  ±{r.sensitivity:.1f}/px  w={r.base_width_px:.0f}px h={r.height_px:.0f}px  conf={r.confidence}")
    for m in r.methods:
        print(f"   {m.name:20s} L={m.theta_left:6.1f} R={m.theta_right:6.1f} mean={m.theta_mean:6.1f}")
    for i in r.issues:
        print(f"   [{i.level}/{i.source}] {i.title}")
    d = r.diagnostics
    if d:
        print("   diag:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()
                         if k in ("surface_visible", "clutter", "view_from_above", "edge_width", "edge_contrast",
                                  "glare_frac", "support", "fit_rms", "ellipse_rejected", "primary_method")})
    if r.message:
        print("   note:", r.message)
    if r.overlay is not None:
        cv2.imwrite(os.path.join(out_dir, "ov_" + name), r.overlay)
