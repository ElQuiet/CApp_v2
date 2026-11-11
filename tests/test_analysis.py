"""Uji regresi & akurasi.  Jalankan:  .venv\\Scripts\\python -m pytest

- Foto contoh: sudut dikunci ke nilai saat ini, agar perubahan algoritma/model tidak diam-diam
  menggeser hasil. Bila perubahan memang disengaja (mis. model dilatih ulang), perbarui EXPECTED.
- Tetesan sintetis: sudut sebenarnya diketahui pasti, jadi mengukur galat absolut. Seed di sini
  berbeda dari data augmentasi saat melatih (ca_core.synthetic.variants), jadi ini data baru bagi model.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from ca_core import analyze
from ca_core.synthetic import drop_image

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"

# θ hasil app (model terlatih ImageJ), 2026-11-11
EXPECTED = {
    "drop_3.jpg": 88.2,
    "drop_4.jpg": 23.5,
    "drop_5.jpg": 82.5,
    "drop_6.jpg": 25.7,
    "drop_7.jpg": 85.2,
    "drop_8.jpg": 24.2,
    "drop_9.jpg": 31.8,
}


@pytest.mark.parametrize("name,theta", EXPECTED.items())
def test_sample_regression(name, theta):
    res = analyze(cv2.imread(str(SAMPLES / name)))
    assert res.ok, res.message
    assert res.theta == pytest.approx(theta, abs=0.3)


@pytest.mark.parametrize("theta", [20, 40, 60, 75, 90, 110, 130, 145])
def test_synthetic_accuracy(theta):
    img = drop_image(theta, radius=500 if theta < 30 else 260, blur=1.2, noise=2.5, seed=1000 + theta)
    res = analyze(img)
    assert res.ok, res.message
    # per kasus ±3° (noise/blur acak); akurasi rata-rata dijaga test_synthetic_batch_not_worse_than_geometry
    assert res.theta == pytest.approx(theta, abs=3.0)
    assert res.diagnostics.get("geom_theta", res.theta) == pytest.approx(theta, abs=3.0)


def test_synthetic_batch_not_worse_than_geometry():
    """Pada tetesan bersih (bukan data latih), model tidak boleh memperburuk geometri."""
    from ca_core.synthetic import variants

    em, eg = [], []
    for theta, kw in variants(16, seed=4242):
        res = analyze(drop_image(theta, **kw))
        em.append(abs(res.theta - theta))
        eg.append(abs(res.diagnostics.get("geom_theta", res.theta) - theta))
    em, eg = np.array(em), np.array(eg)
    assert np.mean(em) <= 1.5
    assert np.mean(em <= 2.0) >= 0.85
    assert np.mean(em) <= np.mean(eg) + 0.5


def test_synthetic_with_reflection_and_tilt():
    res = analyze(drop_image(80, reflection=0.5, tilt=2.0, seed=2024))
    assert res.ok, res.message
    assert res.theta == pytest.approx(80, abs=3.0)


def test_manual_points_collinear_does_not_crash():
    res = analyze(drop_image(90), manual_points=[(100, 500), (200, 500), (300, 500), (400, 500)])
    assert not res.ok
    assert "Analisis gagal" not in res.message


def test_manual_baseline_is_not_overridden_by_model():
    img = drop_image(90)
    res = analyze(img, baseline=((640, 744), (1160, 744)))
    assert res.ok
    assert res.primary_method != "Model (ImageJ)"


def test_models_present_and_bounded():
    from ca_core import learned

    for name in ("ranker", "angle", "error"):
        assert learned.get(name) is not None, f"model {name} tidak ada di ca_core/model"
    meta = learned.get("angle").meta
    assert meta["anchor"] == "a_med" and meta["max_correction"] <= 30
    # koreksi selalu dibatasi terhadap pengukuran geometri, apa pun cirinya
    row = {f: 0.0 for f in learned.get("angle").features} | {"a_med": 100.0}
    assert abs(learned.angle(row) - 100.0) <= meta["max_correction"] + 1e-9


def test_model_json_is_plain_data():
    for p in (ROOT / "ca_core" / "model").glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        assert set(d) >= {"kind", "features", "init", "learning_rate", "trees"}


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


def test_result_is_picklable_for_worker_processes():
    import pickle

    res = analyze(drop_image(70))
    assert pickle.loads(pickle.dumps(res)).theta == pytest.approx(res.theta)
    assert np.isfinite(res.theta)
