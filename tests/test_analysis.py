"""Uji regresi & akurasi.  Jalankan:  .venv\\Scripts\\python -m pytest

- Foto contoh: sudut dikunci ke nilai saat ini, agar perubahan algoritma tidak diam-diam
  menggeser hasil. Bila perubahan memang disengaja (dan lebih akurat), perbarui EXPECTED.
- Tetesan sintetis: sudut sebenarnya diketahui pasti, jadi mengukur galat absolut.
"""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from ca_core import analyze

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"

# nilai θ (rata-rata kiri & kanan) dari scripts/run_samples.py, 2026-11-11
EXPECTED = {
    "drop_3.jpg": 98.4,
    "drop_4.jpg": 32.2,
    "drop_5.jpg": 88.7,
    "drop_6.jpg": 32.3,
    "drop_7.jpg": 92.3,
    "drop_8.jpg": 27.4,
    "drop_9.jpg": 43.2,
}


@pytest.mark.parametrize("name,theta", EXPECTED.items())
def test_sample_regression(name, theta):
    res = analyze(cv2.imread(str(SAMPLES / name)))
    assert res.ok, res.message
    assert res.theta == pytest.approx(theta, abs=0.3)


def synthetic_drop(theta_deg: float, radius: float = 260, size=(1200, 1800), ss: int = 4) -> np.ndarray:
    """Siluet tetesan (tutup bola) gelap pada backlight terang, di atas substrat gelap."""
    h, w = size
    yb, cx = 0.62 * h, w / 2
    cy = yb + radius * np.cos(np.radians(theta_deg))  # pusat lingkaran (y ke bawah)
    big = np.full((h * ss, w * ss), 225, np.uint8)
    cv2.circle(big, (int(cx * ss), int(cy * ss)), int(radius * ss), 40, -1, cv2.LINE_AA)
    big[int(yb * ss):] = 55  # substrat
    img = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    noise = np.random.default_rng(0).normal(0, 2.0, img.shape)
    return cv2.cvtColor(np.clip(img + noise, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("theta", [40, 60, 75, 90, 110, 130])
def test_synthetic_accuracy(theta):
    res = analyze(synthetic_drop(theta))
    assert res.ok, res.message
    assert res.theta == pytest.approx(theta, abs=2.0)  # target validasi README: ±2°


def test_manual_points_collinear_does_not_crash():
    img = synthetic_drop(90)
    res = analyze(img, manual_points=[(100, 500), (200, 500), (300, 500), (400, 500)])
    assert not res.ok
    assert "Analisis gagal" not in res.message


def test_annotated_zip_csv_handles_commas():
    from ca_ui.report import annotated_zip

    res = analyze(cv2.imread(str(SAMPLES / "drop_5.jpg")))
    z = zipfile.ZipFile(io.BytesIO(annotated_zip(
        [dict(label="Entri 0,3", replicate=1, name="drop_5.jpg", res=res)])))
    rows = list(csv.reader(io.StringIO(z.read("ringkasan.csv").decode("utf-8-sig"))))
    assert rows[0][1] == "sampel"
    assert rows[1][1] == "Entri 0,3"
    assert len(rows[1]) == len(rows[0])


def test_lru_cache_evicts_least_recently_used():
    from ca_ui import LRUCache

    c = LRUCache(2)
    c["a"], c["b"] = 1, 2
    assert c.get("a") == 1  # "a" baru dipakai → "b" yang paling lama
    c["c"] = 3
    assert "b" not in c and "a" in c and "c" in c
    assert len(c) == 2
