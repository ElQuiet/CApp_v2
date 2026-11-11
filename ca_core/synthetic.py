"""Foto tetesan sintetis dengan sudut kontak yang diketahui pasti.

Dipakai untuk (1) uji akurasi dan (2) augmentasi data latih: model belajar bahwa pada tetesan yang
bersih, pengukuran geometri sudah benar, sehingga koreksi hanya diberikan bila foto memang bermasalah
(pantulan, bayangan, kaki tetesan gelap) seperti pada foto lab yang dinilai di ImageJ.
"""
from __future__ import annotations

import cv2
import numpy as np


def drop_image(theta_deg: float, radius: float = 260, size=(1200, 1800), ss: int = 4, blur: float = 1.0,
               noise: float = 2.0, bg: int = 225, drop: int = 40, substrate: int = 55, reflection: float = 0.0,
               tilt: float = 0.0, seed: int = 0) -> np.ndarray:
    """Siluet tutup bola gelap pada backlight terang, di atas substrat.

    reflection : 0..1, kekuatan pantulan tetesan pada permukaan pelat (di bawah baseline)
    tilt       : kemiringan foto (derajat)
    """
    h, w = size
    yb, cx = 0.62 * h, w / 2
    cy = yb + radius * np.cos(np.radians(theta_deg))  # pusat lingkaran (y ke bawah)
    big = np.full((h * ss, w * ss), bg, np.uint8)
    cv2.circle(big, (int(cx * ss), int(cy * ss)), int(radius * ss), drop, -1, cv2.LINE_AA)
    big[int(yb * ss):] = substrate
    if reflection > 0:  # bayangan cermin tetesan, redup, tepat di bawah baseline
        mirror = np.full_like(big, substrate)
        cv2.circle(mirror, (int(cx * ss), int((2 * yb - cy) * ss)), int(radius * ss),
                   int(substrate + reflection * (drop - substrate)), -1, cv2.LINE_AA)
        big[int(yb * ss):] = mirror[int(yb * ss):]
    img = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    if tilt:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), tilt, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    img = cv2.GaussianBlur(img, (0, 0), blur)
    img += np.random.default_rng(seed).normal(0, noise, img.shape)
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def variants(n: int = 80, seed: int = 0):
    """(theta, kwargs) bervariasi untuk augmentasi."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        theta = float(rng.uniform(10, 150))
        kw = dict(radius=float(rng.uniform(60, 320)), blur=float(rng.uniform(0.7, 2.5)),
                  noise=float(rng.uniform(1, 6)), bg=int(rng.integers(170, 240)), drop=int(rng.integers(20, 90)),
                  substrate=int(rng.integers(30, 90)), reflection=float(rng.choice([0, 0, 0.3, 0.6])),
                  tilt=float(rng.uniform(-3, 3)), seed=i)
        # tetesan sudut rendah perlu jari-jari besar agar tingginya beberapa piksel ke atas
        if theta < 30:
            kw["radius"] = float(rng.uniform(250, 600))
        yield theta, kw
