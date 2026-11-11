"""Uji cepat fungsi perbandingan dan ekspor memakai foto di samples/."""
import glob
import io
import json
import os
import sys
import zipfile

import cv2
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ca_core import analyze  # noqa: E402
from ca_ui.report import (annotated_zip, compare_groups, dataset_zip, describe_comparison,  # noqa: E402
                          html_report, text_summary)

files = sorted(glob.glob(os.path.join(ROOT, "samples", "drop_*.jpg")))
rows = []
for i, f in enumerate(files):
    rows.append(dict(label="A" if i < 4 else "B", replicate=i % 4 + 1, name=os.path.basename(f),
                     res=analyze(cv2.imread(f)), settings=dict(mode="auto", roi=None, baseline=[], points=[]),
                     hash=f"h{i}", bytes=open(f, "rb").read()))
groups = [{"name": n, "values": [r["res"].theta for r in rows if r["label"] == n]} for n in ("A", "B")]
comp = compare_groups(groups)
texts = describe_comparison(comp)
print(comp.round(3).to_string())
print(texts)
summ = pd.DataFrame([{"Sampel": g["name"], "n": len(g["values"]), "θ rata-rata (°)": sum(g["values"]) / len(g["values"]),
                      "SD (°)": 1.0, "Min (°)": 0.0, "Maks (°)": 0.0} for g in groups])
out = os.path.join(ROOT, "output")
os.makedirs(out, exist_ok=True)
h = html_report("Uji laporan", summ, None, comp, None, texts, rows)
open(os.path.join(out, "laporan_uji.html"), "wb").write(h)
print("html KB", len(h) // 1024)
z = annotated_zip(rows)
print("zip KB", len(z) // 1024, zipfile.ZipFile(io.BytesIO(z)).namelist()[:3])
d = zipfile.ZipFile(io.BytesIO(dataset_zip(rows)))
print("dataset", d.namelist()[:2], len(json.loads(d.read("labels.json"))["items"]))
print(text_summary(summ, texts))
