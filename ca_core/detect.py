"""Deteksi profil tetesan di atas baseline.

Strategi (dirancang agar tahan terhadap latar bertekstur, kilau, dan tepi lemah):
1. Edge map (Canny adaptif) pada pita di atas baseline.
2. Tiap rantai tepi yang halus & cembung ke atas → hipotesis lingkaran yang memotong baseline.
3. Hipotesis dinilai dari dukungan tepi di SEPANJANG busur (dibagi beberapa sektor),
   sehingga garis tekstur yang hanya mendukung sebagian busur besar tidak menang.
4. Hipotesis terbaik diperhalus berulang (fit elips bersumbu vertikal ke tepi di dekat kurva)
   lalu "didorong" ke tepi terluar: siluet tetesan = tepi terluar (kilau/hotspot ada di dalam).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import Line, ParamCurve, curve_distance, fit_circle_fast, fit_ellipse_axis


@dataclass
class DropDetection:
    points: np.ndarray  # titik profil (N, 2) pada citra yang diluruskan
    curve: ParamCurve
    support: float  # fraksi busur yang didukung tepi (0..1)
    edge_map: np.ndarray = field(repr=False, default=None)
    band_top: int = 0
    notes: list[str] = field(default_factory=list)
    score: float = 0.0


def _edge_map(gray: np.ndarray, line: Line, top: int, x0: int, x1: int):
    h, w = gray.shape
    bottom = int(min(h, np.ceil(max(line.y_at(x0), line.y_at(x1))) + 4))
    band = gray[top:bottom, x0:x1]
    scale = max(1.0, w / 1280.0)
    b = cv2.GaussianBlur(band, (0, 0), 1.2 * scale)
    b = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(b)
    gx = cv2.Sobel(b.astype(np.float32), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(b.astype(np.float32), cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    hi = max(float(np.percentile(mag, 96)), 1.0)
    e = cv2.Canny(b, 0.4 * hi, hi, L2gradient=True)
    full = np.zeros((h, w), np.uint8)
    full[top:bottom, x0:x1] = e
    # buang tepi baseline itu sendiri & semua yang di bawahnya
    yy, xx = np.nonzero(full)
    below = (line.c0 + line.m * xx - yy) < 2.0
    full[yy[below], xx[below]] = 0
    return full


def _arc(curve: ParamCurve, line: Line, step: float = 1.0):
    """Titik busur di atas baseline, berurutan dari satu titik kontak ke titik kontak lain."""
    per = np.pi * (3 * (curve.a + curve.b) - np.sqrt(max((3 * curve.a + curve.b) * (curve.a + 3 * curve.b), 0)))
    n = int(np.clip(per / step, 64, 20000))
    pts = curve.polyline(n)
    s = line.height(pts)
    above = s > 1.5
    if not above.all() and above.any():
        k = int(np.argmin(above))  # sebuah titik di bawah baseline → putar agar busur kontinu
        pts, above = np.roll(pts, -k, 0), np.roll(above, -k)
    return pts[above]


def _support(curve: ParamCurve, line: Line, dist: np.ndarray, tol: float = 1.5, nsec: int = 8):
    """(fraksi didukung, skor). Skor = panjang busur × rata-rata sektor terlemah."""
    pts = _arc(curve, line)
    if len(pts) < 16:
        return 0.0, 0.0
    h, w = dist.shape
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    ok = np.zeros(len(pts), bool)
    p = pts[inside].astype(int)
    ok[inside] = dist[p[:, 1], p[:, 0]] <= tol
    frac = float(ok.mean())
    secs = np.sort([c.mean() for c in np.array_split(ok, nsec)])
    weak = float(secs[: nsec // 2].mean())
    q = 0.3 * frac + 0.7 * weak
    if (line.height(curve.polyline(360)) > 0).all():
        q *= 0.75  # kurva tertutup di atas baseline (tidak menyentuh substrat): kurang mungkin
    # kualitas dipangkatkan: busur yang konsisten > busur panjang yang hanya sebagian didukung
    return frac, len(pts) * q**3


def _valid_drop_curve(c: ParamCurve, line: Line, shape, min_h: float) -> bool:
    if c is None or not np.isfinite([c.cx, c.cy, c.a, c.b]).all():
        return False
    h, w = shape
    if max(c.a, c.b) > 2.5 * w or min(c.a, c.b) < 3:
        return False
    if max(c.a, c.b) / min(c.a, c.b) > 8:
        return False
    pts = c.polyline(720)
    s = line.height(pts)
    if s.max() < min_h:  # terlalu pendek untuk dianggap tetesan
        return False
    if s.min() > 0:  # tertutup sepenuhnya di atas baseline: tetap valid (dicek di quality)
        return True
    above = pts[s > 0]
    if above[:, 0].min() < -2 or above[:, 0].max() > w + 2 or above[:, 1].min() < -2:
        return False
    # busur harus cukup melengkung (bukan garis tekstur hampir lurus)
    base = above[:, 0].max() - above[:, 0].min()
    if s.max() < 0.03 * base:
        return False
    return True


def _refine(curve: ParamCurve, line: Line, edge_pts: np.ndarray, shape, dist: np.ndarray,
            iters=(6, 5, 4, 3, 2.5, 2)):
    """Perhalus kurva berulang; simpan kurva dengan skor dukungan terbaik (anti melenceng)."""
    cur = curve
    best_sc, best = _support(curve, line, dist)[1], curve
    s = line.height(edge_pts)
    # titik sangat dekat baseline sering berupa tekstur permukaan → jangan dipakai untuk fit
    cand_all = edge_pts[s > 3.0]
    if len(cand_all) < 8:
        return best
    size = max(min(curve.a, curve.b), 5.0)
    for tol in iters:
        tol = min(tol, max(2.0, 0.2 * size))
        # hanya titik di sekitar kurva saat ini (percepat)
        x0, x1 = cur.cx - cur.a - 15, cur.cx + cur.a + 15
        y0 = cur.cy - cur.b - 15
        cand = cand_all[(cand_all[:, 0] > x0) & (cand_all[:, 0] < x1) & (cand_all[:, 1] > y0)]
        if len(cand) < 8:
            break
        d = curve_distance(cur, cand)
        sel = cand[d < tol]
        if len(sel) < 8:
            break
        nxt = fit_ellipse_axis(sel)
        if not _valid_drop_curve(nxt, line, shape, 2):
            nxt = fit_circle_fast(sel)
        if not _valid_drop_curve(nxt, line, shape, 2):
            break
        cur = nxt
        sc = _support(cur, line, dist)[1]
        if sc > best_sc:
            best_sc, best = sc, cur
    return best


def _push_outward(curve: ParamCurve, line: Line, dist: np.ndarray, base_frac: float):
    """Pilih tepi luar pertama yang koheren (siluet), bukan tepi kilau di dalam.

    Pita kilau menghasilkan dua tepi sejajar: dalam (kurva saat ini) dan luar. Ciri tepi luar:
    dukungan turun (celah di dalam pita) lalu naik lagi. Tekstur jauh di luar diabaikan karena
    hanya puncak PERTAMA setelah celah yang diambil.
    """
    d_max = max(2.0, min(10.0, 0.2 * min(curve.a, curve.b)))
    ds = np.arange(1.0, d_max + 0.5, 1.0)
    sup = [(_support(curve.grown(d), line, dist, tol=1.0)[0]) for d in ds]
    dipped = False
    for i, d in enumerate(ds):
        if sup[i] < 0.55 * base_frac:
            dipped = True
        elif dipped and sup[i] >= max(0.75 * base_frac, 0.5):
            while i + 1 < len(ds) and sup[i + 1] > sup[i]:
                i += 1
            return float(ds[i])
    return 0.0


def snap_profile(gray: np.ndarray, curve: ParamCurve, line: Line, search: float = 2.5) -> np.ndarray:
    """Profil sub-piksel: dari kurva panduan, cari gradien maksimum di sepanjang normal."""
    from scipy.ndimage import map_coordinates

    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    per = np.pi * (3 * (curve.a + curve.b) - np.sqrt((3 * curve.a + curve.b) * (curve.a + 3 * curve.b)))
    t = np.linspace(0, 2 * np.pi, int(np.clip(per, 200, 20000)), endpoint=False)
    pts, der = curve.point(t), curve.deriv(t)
    keep = line.height(pts) > 0.8
    pts, der = pts[keep], der[keep]
    if len(pts) < 8:
        return pts
    nrm = np.column_stack([der[:, 1], -der[:, 0]])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    offs = np.arange(-search, search + 1e-6, 0.25)
    xs = pts[:, 0:1] + nrm[:, 0:1] * offs
    ys = pts[:, 1:2] + nrm[:, 1:2] * offs
    prof = map_coordinates(mag, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(xs.shape)
    k = prof.argmax(1)
    inner = (k > 0) & (k < len(offs) - 1)
    kk = np.clip(k, 1, len(offs) - 2)
    r = np.arange(len(k))
    a, b, c = prof[r, kk - 1], prof[r, kk], prof[r, kk + 1]
    den = a - 2 * b + c
    sub = np.divide(0.5 * (a - c), den, out=np.zeros_like(den), where=np.abs(den) > 1e-9)
    off = offs[kk] + np.clip(sub, -0.5, 0.5) * 0.25
    out = pts + nrm * off[:, None]
    peak = prof[r, k]
    ok = inner & (peak >= 0.3 * np.median(peak)) & (line.height(out) > 0.8)
    return out[ok]


def _hypotheses(edges: np.ndarray, line: Line, shape, min_h: float) -> list[ParamCurve]:
    """Hipotesis lingkaran dari potongan-potongan rantai tepi yang ditelusuri berurutan."""
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    hyps: list[ParamCurve] = []
    keys = set()
    for cnt in contours:
        cnt = cnt[:, 0, :].astype(float)
        n = len(cnt)
        if n < 24:
            continue
        for win in (30, 60, 120, 240, 480):
            if win > n:
                break
            stride = max(win // 3, 8)
            for i in range(0, n - win + 1, stride):
                seg = cnt[i:i + win]
                if np.ptp(seg[:, 0]) < 6 and np.ptp(seg[:, 1]) < 6:
                    continue
                c = fit_circle_fast(seg)
                if c is None:
                    continue
                res = np.abs(np.hypot(seg[:, 0] - c.cx, seg[:, 1] - c.cy) - c.a)
                if np.sqrt(np.mean(res**2)) > 1.2:
                    continue
                if c.cy < seg[:, 1].mean():  # harus cembung ke atas
                    continue
                if abs(line.height(np.array([[c.cx, c.cy]]))[0]) > c.a * 1.02:
                    continue  # tidak memotong baseline
                key = (round(c.cx / 4), round(c.cy / 4), round(c.a / 4))
                if key in keys:
                    continue
                if _valid_drop_curve(c, line, shape, min_h):
                    keys.add(key)
                    hyps.append(c)
    return hyps


def detect_drop(gray_rot: np.ndarray, line: Line, roi: tuple[int, int, int, int] | None = None) -> DropDetection | None:
    """Cari tetesan di atas baseline. roi = (x0, y0, x1, y1) opsional pada citra lurus."""
    found = detect_drops(gray_rot, line, roi, k=1)
    return found[0] if found else None


def _distinct(c: ParamCurve, chosen: list[ParamCurve]) -> bool:
    """Kurva c berada di lokasi lain (bukan variasi dari tetesan yang sudah dipilih)."""
    for o in chosen:
        size = max(o.a, c.a)
        if np.hypot(c.cx - o.cx, c.cy - o.cy) < 0.6 * size and 0.5 < c.a / max(o.a, 1e-6) < 2.0:
            return False
    return True


def detect_drops(gray_rot: np.ndarray, line: Line, roi: tuple[int, int, int, int] | None = None,
                 k: int = 1) -> list[DropDetection]:
    """Hingga k hipotesis tetesan di lokasi berbeda, skor tertinggi lebih dulu."""
    h, w = gray_rot.shape
    yb = float(line.y_at(w / 2))
    if roi is not None:
        x0, top, x1, _ = [int(v) for v in roi]
        x0, x1 = max(0, x0), min(w, x1)
        top = max(0, top)
    else:
        x0, x1 = 0, w
        top = int(max(0, yb - 0.5 * h))
    edges = _edge_map(gray_rot, line, top, x0, x1)
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    ys, xs = np.nonzero(edges)
    edge_pts = np.column_stack([xs, ys]).astype(float)
    if len(edge_pts) < 20:
        return []

    min_h = max(3.0, 0.004 * h)
    hyps = _hypotheses(edges, line, (h, w), min_h)
    if not hyps:
        return []

    scored = sorted(((_support(c, line, dist)[1], c) for c in hyps), key=lambda t: t[0], reverse=True)
    refined = []
    for _, c in scored[:max(10, 4 * k)]:
        r = _refine(c, line, edge_pts, (h, w), dist)
        frac, sc = _support(r, line, dist)
        refined.append((sc, frac, r))
    refined.sort(key=lambda t: t[0], reverse=True)
    out, chosen = [], []
    for score, frac, curve in refined:
        if len(out) >= k or score <= 0:
            break
        if not _distinct(curve, chosen):
            continue
        chosen.append(curve)
        det = _finish(gray_rot, line, curve, frac, score, edges, edge_pts, dist, top, (h, w))
        if det is not None:
            out.append(det)
    return out


def _finish(gray_rot, line, curve, frac, score, edges, edge_pts, dist, top, shape) -> DropDetection | None:
    h, w = shape
    notes = []
    d_out = _push_outward(curve, line, dist, frac)
    if d_out > 0:
        curve2 = _refine(curve.grown(d_out), line, edge_pts, (h, w), dist, iters=(3, 2.5, 2))
        frac2, sc2 = _support(curve2, line, dist)
        if frac2 >= 0.6 * frac:
            curve, frac, score = curve2, frac2, sc2
            notes.append(f"Kontur digeser {d_out:.0f} px ke tepi terluar (siluet).")

    prof = snap_profile(gray_rot, curve, line)
    if len(prof) < 12:  # fallback: titik tepi Canny di dekat kurva
        s = line.height(edge_pts)
        d = curve_distance(curve, edge_pts)
        prof = edge_pts[(d < 2.0) & (s > 0.5)]
    if len(prof) < 8:
        return None
    return DropDetection(prof, curve, frac, edges, top, notes, score)
