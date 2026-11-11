"""Geometri: baseline, fitting kurva (lingkaran/elips/polinomial) dan perhitungan sudut kontak.

Semua koordinat memakai konvensi gambar (x ke kanan, y ke bawah) pada citra yang
sudah diluruskan (baseline kira-kira horizontal).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import brentq, least_squares


@dataclass
class Line:
    """Baseline: y = c0 + m * x."""

    c0: float
    m: float

    @property
    def norm(self) -> float:
        return float(np.hypot(1.0, self.m))

    @property
    def direction(self) -> np.ndarray:
        return np.array([1.0, self.m]) / self.norm

    def y_at(self, x):
        return self.c0 + self.m * np.asarray(x, dtype=float)

    def height(self, pts) -> np.ndarray:
        """Jarak bertanda dari baseline, positif = di atas baseline."""
        pts = np.atleast_2d(pts)
        return (self.c0 + self.m * pts[:, 0] - pts[:, 1]) / self.norm

    def shifted(self, dy: float) -> "Line":
        return Line(self.c0 + dy, self.m)


@dataclass
class ParamCurve:
    """Elips/lingkaran parametrik: x = cx + a cos t cosφ - b sin t sinφ."""

    cx: float
    cy: float
    a: float
    b: float
    phi: float = 0.0  # radian
    kind: str = "ellipse"

    def point(self, t):
        t = np.asarray(t, dtype=float)
        c, s = np.cos(self.phi), np.sin(self.phi)
        x = self.cx + self.a * np.cos(t) * c - self.b * np.sin(t) * s
        y = self.cy + self.a * np.cos(t) * s + self.b * np.sin(t) * c
        return np.stack([x, y], axis=-1)

    def deriv(self, t):
        t = np.asarray(t, dtype=float)
        c, s = np.cos(self.phi), np.sin(self.phi)
        dx = -self.a * np.sin(t) * c - self.b * np.cos(t) * s
        dy = -self.a * np.sin(t) * s + self.b * np.cos(t) * c
        return np.stack([dx, dy], axis=-1)

    def grown(self, d: float) -> "ParamCurve":
        return ParamCurve(self.cx, self.cy, self.a + d, self.b + d, self.phi, self.kind)

    def polyline(self, n: int = 2000) -> np.ndarray:
        return self.point(np.linspace(0, 2 * np.pi, n, endpoint=False))


def fit_circle_fast(pts: np.ndarray) -> ParamCurve | None:
    """Fit lingkaran aljabar (Kasa) tanpa penyempurnaan, untuk membuat hipotesis."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return None
    m = pts.mean(0)
    x, y = (pts - m).T
    A = np.column_stack([x, y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx**2 + cy**2
    if not np.isfinite(r2) or r2 <= 0:
        return None
    return ParamCurve(cx + m[0], cy + m[1], np.sqrt(r2), np.sqrt(r2), 0.0, "circle")


def fit_ellipse_axis(pts: np.ndarray) -> ParamCurve | None:
    """Fit elips dengan sumbu sejajar sumbu citra (sumbu simetri tetesan vertikal).

    A x² + C y² + D x + E y + F = 0 ; lebih stabil daripada elips bebas untuk busur pendek.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 5:
        return None
    m = pts.mean(0)
    sc = max(pts.std(0).max(), 1e-6)
    x, y = ((pts - m) / sc).T
    D = np.column_stack([x**2, y**2, x, y, np.ones_like(x)])
    try:
        _, _, vt = np.linalg.svd(D, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    A, C, Dd, E, F = vt[-1]
    if A * C <= 0:
        return None
    cx, cy = -Dd / (2 * A), -E / (2 * C)
    rhs = Dd**2 / (4 * A) + E**2 / (4 * C) - F
    a2, b2 = rhs / A, rhs / C
    if a2 <= 0 or b2 <= 0:
        return None
    return ParamCurve(cx * sc + m[0], cy * sc + m[1], np.sqrt(a2) * sc, np.sqrt(b2) * sc, 0.0, "ellipse")


def fit_circle(pts: np.ndarray) -> ParamCurve | None:
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return None
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    bvec = x**2 + y**2
    try:
        sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx**2 + cy**2
    if not np.isfinite(r2) or r2 <= 0:
        return None
    r = np.sqrt(r2)
    # Penyempurnaan geometrik (Kasa bias untuk busur pendek)
    res = least_squares(lambda p: np.hypot(x - p[0], y - p[1]) - p[2], [cx, cy, r], method="lm")
    cx, cy, r = res.x
    if not np.isfinite(r) or r <= 0:
        return None
    return ParamCurve(cx, cy, abs(r), abs(r), 0.0, "circle")


def fit_ellipse(pts: np.ndarray) -> ParamCurve | None:
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 6:
        return None
    try:
        (cx, cy), (w, h), ang = cv2.fitEllipseDirect(pts)
    except cv2.error:
        return None
    if not (np.isfinite(w) and np.isfinite(h)) or w <= 0 or h <= 0:
        return None
    return ParamCurve(cx, cy, w / 2, h / 2, np.deg2rad(ang), "ellipse")


def curve_distance(curve: ParamCurve, pts: np.ndarray) -> np.ndarray:
    """Jarak aproksimasi titik ke elips (jarak Sampson): |F| / |∇F|.

    Akurat untuk titik di dekat kurva (yang dipakai untuk ambang toleransi) dan jauh lebih cepat
    daripada pencarian tetangga terdekat pada polyline.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return np.zeros(0)
    c, s_ = np.cos(curve.phi), np.sin(curve.phi)
    dx, dy = pts[:, 0] - curve.cx, pts[:, 1] - curve.cy
    u = dx * c + dy * s_
    v = -dx * s_ + dy * c
    a2, b2 = curve.a ** 2, curve.b ** 2
    f = u * u / a2 + v * v / b2 - 1.0
    g = 2.0 * np.hypot(u / a2, v / b2)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.abs(f) / g
    return np.where(np.isfinite(d), d, np.inf)


@dataclass
class ContactResult:
    theta_left: float
    theta_right: float
    p_left: np.ndarray
    p_right: np.ndarray
    apex_height: float
    base_width: float
    closed_above: bool = False  # kurva tidak memotong baseline sama sekali

    @property
    def theta_mean(self) -> float:
        return 0.5 * (self.theta_left + self.theta_right)


def _angle_between(u, v) -> float:
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def contact_from_curve(curve: ParamCurve, line: Line) -> ContactResult | None:
    """Potong kurva dengan baseline, hitung sudut kontak dari garis tangen."""
    n = 7200
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    s = line.height(curve.point(t))
    above = s > 0
    if above.all():
        apex = float(s.max())
        return ContactResult(np.nan, np.nan, np.full(2, np.nan), np.full(2, np.nan), apex, np.nan, True)
    if not above.any():
        return None
    # Run kontinu (sirkular) di atas baseline yang memuat titik tertinggi
    k = int(np.argmax(s))
    i = k
    while above[(i - 1) % n]:
        i -= 1
    j = k
    while above[(j + 1) % n]:
        j += 1

    def h_of(tt):
        return float(line.height(curve.point(tt))[0])

    step = 2 * np.pi / n
    t_start = brentq(h_of, (i - 1) * step, i * step)  # masuk ke atas baseline
    t_end = brentq(h_of, j * step, (j + 1) * step)  # keluar dari atas baseline
    p_start, p_end = curve.point(t_start), curve.point(t_end)
    v_start = curve.deriv(t_start)  # t naik → masuk ke busur
    v_end = -curve.deriv(t_end)  # t turun → masuk ke busur
    e = line.direction
    if np.dot(p_start, e) <= np.dot(p_end, e):
        pl, vl, pr, vr = p_start, v_start, p_end, v_end
    else:
        pl, vl, pr, vr = p_end, v_end, p_start, v_start
    th_l = _angle_between(e, vl)
    th_r = _angle_between(-e, vr)
    return ContactResult(th_l, th_r, pl, pr, float(s.max()), float(np.linalg.norm(pr - pl)))


def tangent_poly(pts: np.ndarray, line: Line, p_left, p_right, frac: float = 0.3):
    """Metode tangen lokal: fit posisi-vs-tinggi (polinomial) di dekat tiap titik kontak.

    Mengembalikan (theta_left, theta_right); NaN bila titik tidak cukup/tidak andal.
    """
    pts = np.asarray(pts, float)
    s = line.height(pts)
    e = line.direction
    u = pts @ e  # posisi sepanjang baseline
    h_max = s.max() if len(s) else 0
    lim = max(frac * h_max, 5.0)
    ul, ur = np.dot(p_left, e), np.dot(p_right, e)
    mid = 0.5 * (ul + ur)
    out = []
    for side, uc in ((-1, ul), (1, ur)):
        sel = (s >= 0.5) & (s <= lim) & ((u - mid) * side > 0)
        if sel.sum() < 8 or np.ptp(s[sel]) < 3:
            out.append(np.nan)
            continue
        uu, ss = u[sel], s[sel]
        deg = 2 if (sel.sum() >= 20 and np.ptp(ss) >= 6) else 1
        wts = np.ones_like(ss)
        coef = None
        for _ in range(4):  # robust: buang outlier secara iteratif
            coef = np.polyfit(ss, uu, deg, w=wts)
            r = uu - np.polyval(coef, ss)
            sig = max(1.4826 * np.median(np.abs(r)), 0.3)
            wts = (np.abs(r) < 2.5 * sig).astype(float) + 1e-6
        slope = np.polyval(np.polyder(coef), 0.0)  # du/ds di baseline
        # titik kontak hasil polinomial harus dekat titik kontak kurva utama
        if abs(np.polyval(coef, 0.0) - uc) > max(4.0, 0.05 * abs(ur - ul)):
            out.append(np.nan)
            continue
        inward = slope if side < 0 else -slope
        out.append(float(np.degrees(np.arctan2(1.0, inward))))
    return out[0], out[1]


def spherical_cap_angle(height: float, width: float) -> float:
    if not (width > 0 and height > 0):
        return np.nan
    return float(np.degrees(2 * np.arctan(2 * height / width)))
