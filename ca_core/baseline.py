"""Deteksi baseline (permukaan substrat).

Baseline dicari sebagai garis lurus panjang dengan transisi terang (atas) → gelap
(bawah) yang paling kuat, lewat proyeksi gradien pada beberapa sudut kemiringan.
Kasus sebaliknya (substrat terang, latar gelap) juga dicoba.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Line


@dataclass
class BaselineCandidate:
    angle_deg: float  # kemiringan baseline di citra asli (derajat, + = turun ke kanan)
    y_center: float  # posisi y baseline di tengah citra (koordinat asli)
    score: float
    polarity: int  # +1: terang di atas, -1: gelap di atas


def find_baseline_candidates(gray: np.ndarray, max_angle: float = 10.0, n: int = 5,
                             y_range: tuple[float, float] | None = None) -> list[BaselineCandidate]:
    h, w = gray.shape
    s = min(1.0, 640.0 / w)
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    hh, ww = g.shape
    ones = np.ones_like(g)
    cands: list[BaselineCandidate] = []
    for polarity in (1, -1):
        resp = np.clip(-polarity * gy, 0, None)
        for ang in np.arange(-max_angle, max_angle + 1e-6, 0.25):
            # Putar citra sebesar `ang` sehingga garis dengan kemiringan `ang` menjadi horizontal
            M = cv2.getRotationMatrix2D((ww / 2, hh / 2), ang, 1.0)
            r = cv2.warpAffine(resp, M, (ww, hh), flags=cv2.INTER_LINEAR, borderValue=0)
            valid = cv2.warpAffine(ones, M, (ww, hh), borderValue=0)
            cnt = valid.sum(1)
            prof = r.sum(1) / np.maximum(cnt, 1)
            # Garis harus melintang sebagian besar lebar citra
            prof[cnt < 0.5 * ww] = 0
            # Skor dikalikan konsistensi: berapa banyak kolom yang ikut merespons
            for y in np.argsort(prof)[::-1][:4]:
                row = r[y][valid[y] > 0.5]
                frac = float((row > 0.25 * max(row.max(), 1e-6)).mean()) if row.size else 0
                yc = (y - hh / 2) / s + h / 2  # rotasi di pusat → y tengah sama
                if y_range and not (y_range[0] <= yc <= y_range[1]):
                    continue
                cands.append(BaselineCandidate(float(ang), yc, float(prof[y]) * (0.5 + frac), polarity))
    cands.sort(key=lambda c: c.score, reverse=True)
    # Buang duplikat (garis hampir sama)
    out: list[BaselineCandidate] = []
    for c in cands:
        if all(abs(c.y_center - o.y_center) > 0.02 * h or abs(c.angle_deg - o.angle_deg) > 2 for o in out):
            out.append(c)
        if len(out) >= n:
            break
    return out


def rotation_for(angle_deg: float, shape) -> np.ndarray:
    """Matriks rotasi yang membuat garis dengan kemiringan angle_deg menjadi horizontal."""
    h, w = shape[:2]
    return cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)


def refine_local(gray_rot: np.ndarray, line: Line, x_lo: float, x_hi: float, polarity: int = 1,
                 search: int = 6, exclude: tuple[float, float] | None = None) -> Line:
    """Perhalus baseline secara lokal (sub-piksel) di sekitar tetesan.

    Kolom di dalam `exclude` (area tetesan) diabaikan agar tetesan tidak menarik garis.
    """
    h, w = gray_rot.shape
    g = cv2.GaussianBlur(gray_rot.astype(np.float32), (0, 0), 1.0)
    gy = -polarity * cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    xs, ys, ws = [], [], []
    for x in range(max(1, int(x_lo)), min(w - 1, int(x_hi))):
        if exclude and exclude[0] <= x <= exclude[1]:
            continue
        y0 = int(round(line.y_at(x)))
        a, b = max(1, y0 - search), min(h - 2, y0 + search)
        if b <= a:
            continue
        col = gy[a:b + 1, x]
        k = int(np.argmax(col))
        if col[k] <= 0:
            continue
        # interpolasi parabola sub-piksel
        if 0 < k < len(col) - 1:
            d = col[k - 1] - 2 * col[k] + col[k + 1]
            off = 0.5 * (col[k - 1] - col[k + 1]) / d if d != 0 else 0.0
        else:
            off = 0.0
        xs.append(x)
        ys.append(a + k + off)
        ws.append(col[k])
    if len(xs) < 10:
        return line
    xs, ys, ws = map(np.asarray, (xs, ys, ws))
    # fit garis robust (iteratively reweighted)
    m, c0 = line.m, line.c0
    for _ in range(5):
        r = ys - (c0 + m * xs)
        sig = max(1.4826 * np.median(np.abs(r - np.median(r))), 0.3)
        wt = ws * (np.abs(r) < 2.5 * sig)
        if wt.sum() <= 0:
            break
        A = np.column_stack([np.ones_like(xs), xs]) * np.sqrt(wt)[:, None]
        c0, m = np.linalg.lstsq(A, ys * np.sqrt(wt), rcond=None)[0]
    if abs(m) > 0.1:  # terlalu jauh dari garis awal, abaikan
        return line
    return Line(float(c0), float(m))


def refine_global(gray_rot: np.ndarray, line: Line, polarity: int = 1, search: int = 4) -> Line:
    """Perhalus posisi baseline (sub-piksel) dari profil baris rata-rata seluruh lebar."""
    g = cv2.GaussianBlur(gray_rot.astype(np.float32), (0, 0), 1.0)
    gy = -polarity * cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    y0 = int(round(line.c0))
    a, b = max(1, y0 - search), min(g.shape[0] - 2, y0 + search)
    prof = gy[a:b + 1].mean(1)
    k = int(np.argmax(prof))
    off = 0.0
    if 0 < k < len(prof) - 1:
        d = prof[k - 1] - 2 * prof[k] + prof[k + 1]
        off = 0.5 * (prof[k - 1] - prof[k + 1]) / d if d != 0 else 0.0
    return Line(float(a + k + off), 0.0)
