"""Kandidat pengukuran sudut + ciri geometri dari satu kontur tetesan.

Sudut kontak dari foto bergantung pada keputusan yang ambigu: di mana baseline (tepi depan pelat,
garis pertemuan tetesan–pantulan, atau ujung kontur yang masih terlihat) dan bagian kontur mana yang
dipakai (seluruhnya, atau hanya busur atas karena kaki tetesan gelap/berbayang). Modul ini menghitung
beberapa variasi yang masuk akal sekaligus. Model terlatih (lihat learned.py) memilih/menggabungkannya
berdasarkan ciri geometri, dengan acuan pengukuran manual ImageJ.

Semua koordinat pada citra lurus (baseline kira-kira horizontal), y ke bawah.
"""
from __future__ import annotations

import numpy as np

from .geometry import Line, contact_from_curve, fit_circle, fit_ellipse_axis, spherical_cap_angle, tangent_poly

CAND_NAMES = ["ell", "cir", "tan", "hw", "up_ell", "up_cir", "end_ell", "end_hw", "ext_ell", "ext_cir", "ext_hw"]
FEAT_NAMES = ["closed", "aspect", "wmax_ratio", "gap", "ext_drop", "support", "fit_rms", "edge_width",
              "edge_contrast", "asym", "sens", "log_width", "surface", "clutter", "glare", "ell_rejected"]


def _theta(curve, line):
    if curve is None:
        return np.nan, None
    c = contact_from_curve(curve, line)
    if c is None or c.closed_above:
        return np.nan, c
    return c.theta_mean, c


def _line_through(p, q) -> Line:
    m = (q[1] - p[1]) / (q[0] - p[0]) if abs(q[0] - p[0]) > 1e-9 else 0.0
    return Line(float(p[1] - m * p[0]), float(m))


def compute(pts: np.ndarray, line: Line, primary, cres, ctx: dict) -> dict:
    """pts: titik tepi tetesan; primary/cres: kurva utama & hasil kontaknya (boleh None/tertutup)."""
    out = {k: np.nan for k in CAND_NAMES + FEAT_NAMES}
    pts = np.asarray(pts, float)
    s = line.height(pts)
    pts = pts[s > 0.5]
    s = s[s > 0.5]
    if len(pts) < 8:
        return out
    e = line.direction
    u = pts @ e
    H = float(s.max())
    closed = bool(cres is None or cres.closed_above)
    out["closed"] = float(closed)

    # --- metode standar terhadap baseline yang terdeteksi
    out["ell"], c_ell = _theta(fit_ellipse_axis(pts), line)
    out["cir"], _ = _theta(fit_circle(pts), line)
    if not closed:
        tl, tr = tangent_poly(pts, line, cres.p_left, cres.p_right)
        out["tan"] = float(np.nanmean([tl, tr])) if np.isfinite([tl, tr]).any() else np.nan
        out["hw"] = spherical_cap_angle(cres.apex_height, cres.base_width)
        W = cres.base_width
    else:
        W = float(np.ptp(u))

    # --- hanya busur atas (abaikan kaki tetesan yang sering gelap/berbayang/terpotong)
    up = pts[s > 0.25 * H]
    if len(up) >= 8:
        out["up_ell"], _ = _theta(fit_ellipse_axis(up), line)
        out["up_cir"], _ = _theta(fit_circle(up), line)

    # --- baseline di ujung kontur yang masih terlihat (kontur berhenti di atas tepi pelat)
    mid = np.median(u)
    sl, sr = s[u < mid], s[u >= mid]
    if len(sl) and len(sr):
        gap = float(0.5 * (sl.min() + sr.min()))
        out["gap"] = gap / H
        if gap > 0.5:
            l_end = Line(line.c0 - gap * line.norm, line.m)
            out["end_ell"], c_end = _theta(fit_ellipse_axis(pts), l_end)
            if c_end is not None and not c_end.closed_above:
                out["end_hw"] = spherical_cap_angle(c_end.apex_height, c_end.base_width)
        else:
            out["end_ell"], out["end_hw"] = out["ell"], out["hw"]

    # --- baseline melalui ujung kiri & kanan siluet (kamera dari atas: tapak/pantulan ikut terlihat)
    pl, pr = pts[np.argmin(u)], pts[np.argmax(u)]
    l_ext = _line_through(pl, pr)
    s_ext = l_ext.height(pts)
    out["ext_drop"] = float(np.mean(line.height(np.array([pl, pr])))) / H  # tinggi ujung di atas baseline
    above = pts[s_ext > 0.5]
    if len(above) >= 8:
        out["ext_ell"], _ = _theta(fit_ellipse_axis(above), l_ext)
        out["ext_cir"], _ = _theta(fit_circle(above), l_ext)
        out["ext_hw"] = spherical_cap_angle(float(s_ext.max()), float(np.hypot(*(pr - pl))))

    out["aspect"] = H / max(W, 1e-6)
    out["wmax_ratio"] = float(np.ptp(u)) / max(W, 1e-6)
    out["log_width"] = float(np.log(max(W, 1.0)))
    for k_out, k_ctx in (("support", "support"), ("fit_rms", "fit_rms"), ("edge_width", "edge_width"),
                         ("edge_contrast", "edge_contrast"), ("asym", "asymmetry"), ("sens", "sensitivity"),
                         ("clutter", "clutter"), ("glare", "glare_frac")):
        v = ctx.get(k_ctx, np.nan)
        out[k_out] = float(v) if v is not None and np.isfinite(v) else np.nan
    out["surface"] = float(ctx.get("surface_visible", 0.0) or 0.0) / max(H, 1e-6)
    out["ell_rejected"] = float(bool(ctx.get("ellipse_rejected")))
    return out
