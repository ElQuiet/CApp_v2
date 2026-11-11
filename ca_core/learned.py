"""Model terlatih dari pengukuran manual ImageJ (lihat scripts/train_model.py).

Tiga model gradient boosting kecil, disimpan sebagai pohon keputusan di ca_core/model/*.json
(cukup numpy untuk memakainya; scikit-learn hanya dibutuhkan saat melatih):

- ranker : dari beberapa kandidat lokasi tetesan, mana yang benar-benar tetesan (bukan kilau/kotoran);
- angle  : sudut kontak dari kandidat-kandidat pengukuran geometri (candidates.py) + ciri bentuk;
- error  : perkiraan besar selisih terhadap ImageJ → skor keyakinan yang adaptif.

Bila file model tidak ada, app tetap berjalan dengan metode geometri saja.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from .candidates import CAND_NAMES, FEAT_NAMES

MODEL_DIR = Path(__file__).parent / "model"
MISSING = -1.0  # pengganti NaN (sama dengan saat melatih)


def hyp_features(res, rank: int, score_ratio: float, img_shape) -> dict:
    """Ciri satu hipotesis tetesan (hasil analisis resolusi penuh), tidak bergantung ukuran foto."""
    c = res.diagnostics.get("cand", {}) if res is not None else {}
    f = {f"c_{k}": float(c.get(k, np.nan)) for k in CAND_NAMES + FEAT_NAMES}
    angles = np.array([c.get(k, np.nan) for k in CAND_NAMES], float)
    angles = np.clip(angles[np.isfinite(angles)], 0, 180)
    f["a_med"] = float(np.median(angles)) if len(angles) else np.nan
    f["a_min"] = float(angles.min()) if len(angles) else np.nan
    f["a_max"] = float(angles.max()) if len(angles) else np.nan
    f["a_std"] = float(angles.std()) if len(angles) > 1 else np.nan
    f["rank"] = float(rank)
    f["score_ratio"] = float(score_ratio)
    H, W = img_shape[:2]
    f["ok"] = float(bool(res is not None and res.ok))
    f["conf"] = float(res.confidence) if res is not None else np.nan
    f["w_rel"] = float(res.base_width_px / W) if res is not None and np.isfinite(res.base_width_px) else np.nan
    f["h_rel"] = float(res.height_px / W) if res is not None and np.isfinite(res.height_px) else np.nan
    if res is not None and res.roi_suggestion is not None:
        x0, y0, x1, y1 = res.roi_suggestion
        f["x_rel"] = abs(0.5 * (x0 + x1) / W - 0.5)
        f["y_rel"] = 0.5 * (y0 + y1) / H
    else:
        f["x_rel"] = f["y_rel"] = np.nan
    return f


class TreeModel:
    """Gradient boosting (pohon regresi) hasil ekspor scikit-learn."""

    def __init__(self, d: dict):
        self.kind = d["kind"]  # "regressor" | "classifier"
        self.features = d["features"]
        self.init = float(d["init"])
        self.lr = float(d["learning_rate"])
        self.trees = [{k: np.asarray(v) for k, v in t.items()} for t in d["trees"]]
        self.meta = d.get("meta", {})

    def _matrix(self, rows: list[dict]) -> np.ndarray:
        X = np.array([[r.get(k, np.nan) for k in self.features] for r in rows], float)
        X = np.where(np.isfinite(X), X, MISSING)
        return X.astype(np.float32).astype(float)  # scikit-learn membandingkan ciri dalam float32

    def raw(self, rows: list[dict]) -> np.ndarray:
        X = self._matrix(rows)
        out = np.full(len(X), self.init)
        idx = np.arange(len(X))
        for t in self.trees:
            node = np.zeros(len(X), int)
            for _ in range(64):  # kedalaman pohon kecil (≤ 4)
                leaf = t["left"][node] < 0
                if leaf.all():
                    break
                go = ~leaf
                f, th = t["feature"][node[go]], t["threshold"][node[go]]
                left = X[idx[go], f] <= th
                node[go] = np.where(left, t["left"][node[go]], t["right"][node[go]])
            out += self.lr * t["value"][node]
        return out

    def predict(self, rows: list[dict]) -> np.ndarray:
        r = self.raw(rows)
        return 1.0 / (1.0 + np.exp(-r)) if self.kind == "classifier" else r


@lru_cache(maxsize=1)
def _models() -> dict:
    out = {}
    for name in ("ranker", "angle", "error"):
        p = MODEL_DIR / f"{name}.json"
        if p.exists():
            out[name] = TreeModel(json.loads(p.read_text(encoding="utf-8")))
    return out


def get(name: str) -> TreeModel | None:
    return _models().get(name)


def rank(rows: list[dict]) -> np.ndarray | None:
    m = get("ranker")
    return None if m is None or not rows else m.predict(rows)


def angle(row: dict) -> float:
    m = get("angle")
    if m is None:
        return math.nan
    pred = float(m.predict([row])[0])
    anchor = m.meta.get("anchor")
    if anchor:  # model mempelajari koreksi (terbatas) terhadap pengukuran geometri
        base = row.get(anchor, math.nan)
        if not np.isfinite(base):
            return math.nan
        lim = float(m.meta.get("max_correction", 25.0))
        pred = base + float(np.clip(pred, -lim, lim))
    return float(np.clip(pred, 0.0, 180.0))


def expected_error(row: dict) -> float:
    m = get("error")
    return math.nan if m is None else float(max(0.5, m.predict([row])[0]))
