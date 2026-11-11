"""Ekspor: gambar beranotasi (ZIP), laporan HTML mandiri, data latih, dan statistik perbandingan."""
from __future__ import annotations

import base64
import csv
import datetime as dt
import html
import io
import json
import math
import re
import zipfile

import cv2
import numpy as np
import pandas as pd
from scipy import stats


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "foto"


def _jpeg(img, q=90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes() if ok else b""


def annotated_zip(rows: list[dict]) -> bytes:
    """rows: {label, replicate, name, res}. Overlay disimpan sebagai JPEG, plus tabel ringkas."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        table = io.StringIO()
        # csv.writer memberi tanda kutip pada nama sampel yang memuat koma (mis. "Entri 0,3")
        w = csv.writer(table, lineterminator="\n")
        w.writerow(["file", "sampel", "ulangan", "theta_kiri", "theta_kanan", "theta", "keyakinan"])
        for r in rows:
            res = r["res"]
            if res.overlay is None:
                continue
            fn = f"{safe_name(r['label'])}_{int(r['replicate']):02d}_{safe_name(r['name'].rsplit('.', 1)[0])}.jpg"
            z.writestr(fn, _jpeg(res.overlay))
            w.writerow([fn, r["label"], r["replicate"], f"{res.theta_left:.2f}", f"{res.theta_right:.2f}",
                        f"{res.theta:.2f}", res.confidence])
        z.writestr("ringkasan.csv", table.getvalue().encode("utf-8-sig"))
    return buf.getvalue()


def dataset_zip(rows: list[dict]) -> bytes:
    """Foto asli + label (koreksi user dan hasil) untuk evaluasi/penyempurnaan algoritma."""
    buf = io.BytesIO()
    labels = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for r in rows:
            res, s = r["res"], r["settings"]
            fn = f"images/{r['hash']}_{safe_name(r['name'])}"
            z.writestr(fn, r["bytes"])
            labels.append({
                "file": fn, "original_name": r["name"], "sample": r["label"], "replicate": r["replicate"],
                "corrected": s["mode"] != "auto" or bool(s["roi"] or s["baseline"] or s["points"]),
                "correction": {"roi": s["roi"], "baseline": s["baseline"], "edge_points": s["points"]},
                "result": {"ok": res.ok, "theta": _num(res.theta), "theta_left": _num(res.theta_left),
                           "theta_right": _num(res.theta_right), "confidence": res.confidence,
                           "method": res.primary_method, "baseline": res.baseline_pts,
                           "contacts": res.contact_pts},
                "issues": [{"level": i.level, "source": i.source, "title": i.title} for i in res.issues],
                "reference_theta_imagej": None,  # isi bila ada nilai acuan
            })
        z.writestr("labels.json", json.dumps({"version": 1, "created": dt.datetime.now().isoformat(timespec="seconds"),
                                              "items": labels}, indent=2, default=_json_default))
        z.writestr("README.txt", (
            "Data latih Contact Angle Analyzer.\n"
            "labels.json berisi koreksi user (koordinat piksel foto asli) dan hasil analisis tiap foto.\n"
            "Isi 'reference_theta_imagej' dengan nilai acuan ImageJ bila tersedia untuk validasi.\n").encode())
    return buf.getvalue()


def _num(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def compare_groups(groups: list[dict]) -> pd.DataFrame:
    """groups: {name, values}. Statistik tiap kelompok + selisih & uji Welch terhadap kelompok pertama."""
    rows = []
    ref = groups[0] if groups else None
    for g in groups:
        v = np.asarray([x for x in g["values"] if np.isfinite(x)], float)
        row = {"Kelompok": g["name"], "n": len(v), "Rata-rata (°)": v.mean() if len(v) else np.nan,
               "SD (°)": v.std(ddof=1) if len(v) > 1 else 0.0 if len(v) else np.nan,
               "Δ (°)": np.nan, "p": np.nan}
        if ref is not None and g is not ref:
            rv = np.asarray([x for x in ref["values"] if np.isfinite(x)], float)
            if len(v) and len(rv):
                row["Δ (°)"] = v.mean() - rv.mean()
            if len(v) >= 2 and len(rv) >= 2:
                row["p"] = float(stats.ttest_ind(v, rv, equal_var=False).pvalue)
        rows.append(row)
    return pd.DataFrame(rows)


def describe_comparison(df: pd.DataFrame) -> list[str]:
    out = []
    if len(df) < 2:
        return out
    ref = df.iloc[0]
    for _, r in df.iloc[1:].iterrows():
        if not np.isfinite(r["Δ (°)"]):
            continue
        arah = "lebih tinggi" if r["Δ (°)"] > 0 else "lebih rendah"
        s = f"{r['Kelompok']} {abs(r['Δ (°)']):.1f}° {arah} dari {ref['Kelompok']}"
        if np.isfinite(r["p"]):
            sig = "berbeda signifikan" if r["p"] < 0.05 else "tidak berbeda signifikan"
            s += f" (p = {r['p']:.3g}, {sig} pada α = 0,05)"
        else:
            s += " (uji statistik butuh minimal 2 foto per kelompok)"
        out.append(s + ".")
    return out


def html_report(title: str, summary: pd.DataFrame, chart_png: bytes | None, comp: pd.DataFrame | None,
                comp_png: bytes | None, comp_text: list[str], rows: list[dict]) -> bytes:
    """Laporan satu file HTML (gambar tertanam) yang bisa dikirim lewat email/chat dan dibuka di browser."""
    def img_tag(png, alt, mime="image/png"):
        return f'<img alt="{html.escape(alt)}" src="data:{mime};base64,{base64.b64encode(png).decode()}">' if png else ""

    def table(df, fmt):
        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(fmt(c, v))}</td>" for c, v in r.items()) + "</tr>"
                       for _, r in df.iterrows())
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    def f(c, v):
        if isinstance(v, (float, np.floating)):
            if not math.isfinite(v):
                return "–"
            return f"{v:.3g}" if c == "p" else f"{v:.2f}"
        return str(v)

    cards = []
    for r in rows:
        res = r["res"]
        if res.overlay is None:
            continue
        thumb = res.overlay
        if res.roi_suggestion is not None:
            s = res.scale
            x0, y0, x1, y1 = [int(v * s) for v in res.roi_suggestion]
            x0, x1 = sorted([x0, x1]); y0, y1 = sorted([y0, y1])
            pad = int(0.6 * max(x1 - x0, 20))
            h, w = thumb.shape[:2]
            thumb = thumb[max(0, y0 - pad):min(h, y1 + pad // 2), max(0, x0 - pad):min(w, x1 + pad)]
        if thumb.size == 0:
            thumb = res.overlay
        tw = 360
        thumb = cv2.resize(thumb, (tw, max(1, int(tw * thumb.shape[0] / thumb.shape[1]))), interpolation=cv2.INTER_AREA)
        theta = "–" if not res.ok else f"{res.theta:.1f}°"
        cards.append(f'<figure>{img_tag(_jpeg(thumb, 82), r["name"], "image/jpeg")}<figcaption><b>{theta}</b> '
                     f'{html.escape(r["label"])}, ulangan {r["replicate"]}<br><span>{html.escape(r["name"])} · '
                     f'keyakinan {res.confidence}%</span></figcaption></figure>')

    comp_html = ""
    if comp is not None and len(comp):
        comp_html = (f"<h2>Perbandingan</h2>{img_tag(comp_png, 'Grafik perbandingan')}"
                     + "".join(f"<p>{html.escape(t)}</p>" for t in comp_text) + table(comp, f))
    doc = f"""<!doctype html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>
body {{ font-family: "IBM Plex Sans", system-ui, sans-serif; color: #1C2522; background: #F5F7F6; margin: 0; }}
main {{ max-width: 1000px; margin: 0 auto; padding: 2rem 1rem 4rem; }}
h1 {{ font-weight: 600; margin: 0 0 .2rem; }} h2 {{ font-weight: 600; margin: 2.2rem 0 .8rem; }}
.meta {{ color: #66726D; margin: 0 0 1.5rem; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; margin: .6rem 0; background: #fff; }}
th {{ text-align: left; font-weight: 500; color: #66726D; }}
th, td {{ padding: .5rem .6rem; border-bottom: 1px solid #DCE2DF; font-size: .92rem; }}
img {{ max-width: 100%; height: auto; display: block; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }}
figure {{ margin: 0; background: #fff; border: 1px solid #DCE2DF; border-radius: 6px; overflow: hidden; }}
figcaption {{ padding: .5rem .6rem .6rem; font-size: .85rem; }} figcaption span {{ color: #66726D; font-size: .78rem; }}
p {{ max-width: 72ch; line-height: 1.5; }}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="meta">Dibuat {dt.datetime.now().strftime("%d-%m-%Y %H:%M")} dengan Contact Angle Analyzer (sessile drop).</p>
<h2>Ringkasan per sampel</h2>{table(summary, f)}{img_tag(chart_png, "Grafik sudut kontak per sampel")}
{comp_html}
<h2>Foto beranotasi</h2><div class="grid">{''.join(cards)}</div>
</main></body></html>"""
    return doc.encode("utf-8")


def text_summary(summary: pd.DataFrame, comp_text: list[str]) -> str:
    lines = ["Hasil contact angle (sessile drop)"]
    for _, r in summary.iterrows():
        lines.append(f"- {r['Sampel']}: {r['θ rata-rata (°)']:.1f}° ± {r['SD (°)']:.1f}° (n = {int(r['n'])})")
    if comp_text:
        lines.append("")
        lines += comp_text
    return "\n".join(lines)
