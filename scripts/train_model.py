"""Latih model pemilih tetesan, model sudut, dan model kesalahan dari hasil benchmark.

Alur:
  1. scripts/build_reference.py   → data/reference_imagej.csv   (nilai ImageJ per foto)
  2. scripts/benchmark.py --tag X  → output/hyps_X.csv           (semua kandidat tetesan + ciri per foto)
  3. scripts/train_model.py X      → ca_core/model/{ranker,angle,error}.json + laporan validasi

Validasi: leave-one-sample-out. Setiap sampel (mis. "49.G1-L", semua tahapnya) diprediksi oleh model yang
dilatih TANPA foto sampel itu, sehingga angka akurasi mencerminkan foto baru, bukan hafalan.
Butuh scikit-learn (hanya untuk melatih):  .venv\\Scripts\\python -m pip install scikit-learn
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ca_core.learned import MISSING, TreeModel  # noqa: E402

ID_COLS = ["name", "sample", "theta_ref", "ref_ok", "hyp", "chosen"]
GB = dict(learning_rate=0.05, n_estimators=150, max_depth=3, subsample=0.8, random_state=0)


def matrix(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    X = df[feats].to_numpy(float)
    return np.ascontiguousarray(np.where(np.isfinite(X), X, MISSING))


def label_hyps(h: pd.DataFrame) -> pd.DataFrame:
    """good = hipotesis yang (menurut sudutnya) adalah tetesan yang diukur di ImageJ."""
    h = h.copy()
    h["herr"] = (h["a_med"] - h["theta_ref"]).abs().fillna(999)
    best = h.groupby("name")["herr"].transform("min")
    h["good"] = ((h["herr"] <= np.maximum(10, best + 4)) & (best <= 20)).astype(int)
    return h


def can_predict(row) -> bool:
    if row["ok"] > 0:
        return True
    return np.isfinite(row.get("c_ext_hw", np.nan)) or np.isfinite(row.get("c_ext_cir", np.nan))


ANCHOR = "a_med"  # model sudut mempelajari KOREKSI terhadap median kandidat geometri
MAX_CORR = 25.0  # batas koreksi (derajat): tanpa alasan kuat, hasil tetap mengikuti geometri


def fit_models(h: pd.DataFrame, feats: list[str], syn: pd.DataFrame | None = None):
    ranker = GradientBoostingClassifier(**GB).fit(matrix(h, feats), h["good"])
    # pengukuran dari hipotesis yang memang tetesan yang diukur di ImageJ (+ tetesan sintetis bersih)
    tgt = pd.concat([h[h["good"] == 1], syn[syn["good"] == 1] if syn is not None else None])
    tgt = tgt[tgt[ANCHOR].notna()]
    angle = GradientBoostingRegressor(loss="huber", **GB).fit(matrix(tgt, feats), tgt["theta_ref"] - tgt[ANCHOR])
    return ranker, angle


def predict_angle(angle, rows: pd.DataFrame, feats) -> np.ndarray:
    corr = np.clip(angle.predict(matrix(rows, feats)), -MAX_CORR, MAX_CORR)
    return np.clip(rows[ANCHOR].to_numpy(float) + corr, 0, 180)


def select(h: pd.DataFrame, ranker, feats) -> pd.DataFrame:
    """Per foto: hipotesis dengan peluang tertinggi (atau satu-satunya), seperti saat app berjalan."""
    h = h.copy()
    h["p"] = ranker.predict_proba(matrix(h, feats))[:, 1]
    n = h.groupby("name")["hyp"].transform("size")
    h.loc[n == 1, "p"] = 1.0
    return h.sort_values(["name", "p"], ascending=[True, False]).groupby("name").head(1)


def report(name: str, pred: pd.Series, ref: pd.Series) -> dict:
    e = (pred - ref).abs()
    ok = e.notna()
    lo, hi = ref < 40, ref >= 70
    r = dict(n=int(len(ref)), measured=f"{ok.mean():.0%}", MAE=round(float(e[ok].mean()), 2),
             median=round(float(e[ok].median()), 2), within5=f"{(e <= 5).mean():.0%}",
             within10=f"{(e <= 10).mean():.0%}", MAE_low=round(float(e[lo & ok].mean()), 2),
             MAE_high=round(float(e[hi & ok].mean()), 2))
    print(f"{name:34s}", r)
    return r


def export(model, feats: list[str], kind: str, meta: dict, X_check: np.ndarray) -> dict:
    trees = []
    for est in model.estimators_[:, 0]:
        t = est.tree_
        trees.append(dict(left=t.children_left.tolist(), right=t.children_right.tolist(),
                          feature=np.maximum(t.feature, 0).tolist(), threshold=t.threshold.tolist(),
                          value=t.value[:, 0, 0].tolist()))
    init = float(model._raw_predict_init(X_check[:1])[0, 0])
    d = dict(kind=kind, features=feats, init=init, learning_rate=model.learning_rate, trees=trees, meta=meta)
    # pastikan inferensi numpy (ca_core.learned) identik dengan scikit-learn
    rows = [dict(zip(feats, x)) for x in X_check]
    ours = TreeModel(json.loads(json.dumps(d))).raw(rows)
    ref = model._raw_predict(X_check.astype(np.float32))[:, 0]
    assert np.allclose(ours, ref, atol=1e-4), f"ekspor {kind} tidak cocok: {np.abs(ours - ref).max()}"
    return d


def main(tag: str):
    h = pd.read_csv(ROOT / "output" / f"hyps_{tag}.csv")
    bench = pd.read_csv(ROOT / "output" / f"benchmark_{tag}.csv")
    bench = bench[bench["ref_ok"] == 1]
    h = label_hyps(h[h["ref_ok"] == 1])
    syn_path = ROOT / "output" / "hyps_synthetic.csv"
    syn = label_hyps(pd.read_csv(syn_path)) if syn_path.exists() else None
    feats = [c for c in h.columns if c not in ID_COLS + ["herr", "good"]]
    names = bench["name"].tolist()
    ref = bench.set_index("name")["theta_ref"]
    print(f"{len(names)} foto berlabel, {len(h)} hipotesis, {h.good.sum()} hipotesis 'tetesan benar', "
          f"{len(feats)} ciri, {h['sample'].nunique()} sampel"
          + (f"; + {syn['name'].nunique()} tetesan sintetis" if syn is not None else "") + "\n")

    # ---- validasi silang leave-one-sample-out (ranker + model sudut dilatih ulang tiap lipatan)
    oof = pd.Series(np.nan, index=names)
    oof_sel = {}
    for tr, te in LeaveOneGroupOut().split(h, groups=h["sample"]):
        htr, hte = h.iloc[tr], h.iloc[te]
        ranker, angle = fit_models(htr, feats, syn)
        sel = select(hte, ranker, feats)
        sel = sel[sel.apply(can_predict, axis=1)]
        for (_, r), th in zip(sel.iterrows(), predict_angle(angle, sel, feats) if len(sel) else []):
            oof_sel[r["name"]] = r
            oof[r["name"]] = float(th)
    if syn is not None:  # tetesan sintetis diuji dengan model yang tidak pernah melihat data sintetis
        ranker, angle = fit_models(h, feats)
        s_sel = select(syn, ranker, feats)
        s_sel = s_sel[s_sel.apply(can_predict, axis=1)]
        s_pred = pd.Series(predict_angle(angle, s_sel, feats), index=s_sel["name"].values)
        e_geo = (s_sel["a_med"] - s_sel["theta_ref"]).abs()
        e_mod = (s_pred.values - s_sel["theta_ref"]).__abs__()
        print(f"Tetesan sintetis (sudut pasti), model TANPA augmentasi: selisih geometri {e_geo.mean():.1f}°, "
              f"model {e_mod.mean():.1f}°")
        # selisih out-of-fold model DENGAN augmentasi (5 lipatan sintetis), untuk model kesalahan
        syn_names = syn["name"].unique()
        folds = np.arange(len(syn_names)) % 5
        e_aug = []
        for f in range(5):
            hold = set(syn_names[folds == f])
            ranker, angle = fit_models(h, feats, syn[~syn["name"].isin(hold)])
            s_sel = select(syn[syn["name"].isin(hold)], ranker, feats)
            s_sel = s_sel[s_sel.apply(can_predict, axis=1)]
            for (_, r), th in zip(s_sel.iterrows(), predict_angle(angle, s_sel, feats) if len(s_sel) else []):
                oof_sel[r["name"]] = r
                oof[r["name"]] = float(th)
                e_aug.append(abs(th - r["theta_ref"]))
        print(f"Tetesan sintetis, model DENGAN augmentasi (out-of-fold): selisih {np.mean(e_aug):.1f}°\n")

    print("Perbandingan dengan ImageJ (validasi silang per sampel):")
    base = report("app tanpa model (deteksi + elips)", bench.set_index("name")["geom_theta"].reindex(names), ref[names])
    first = h[h["hyp"] == 0].set_index("name")["a_med"].reindex(names)
    report("median kandidat, hipotesis pertama", first, ref[names])
    cvr = report("MODEL (pemilih + koreksi sudut)", oof.reindex(names), ref[names])
    smp = bench.assign(pred=oof.reindex(bench["name"]).values)
    per = smp.groupby(["sample", "stage"]).agg(ref=("theta_ref", "mean"), pred=("pred", "mean"),
                                               geo=("geom_theta", "mean"))
    print(f"\nRata-rata per sampel & tahap ({len(per)} kelompok): selisih rata-rata model "
          f"{(per.pred - per.ref).abs().mean():.1f}°, tanpa model {(per.geo - per.ref).abs().mean():.1f}°")

    # ---- model kesalahan: dilatih pada selisih out-of-fold (jujur), untuk skor keyakinan
    sel_df = pd.DataFrame(oof_sel.values()).reset_index(drop=True)
    sel_df["abs_err"] = (oof.reindex(sel_df["name"]).values - sel_df["theta_ref"]).__abs__()
    sel_df = sel_df[sel_df["abs_err"].notna()]
    err_cv = pd.Series(np.nan, index=sel_df.index)
    for tr, te in LeaveOneGroupOut().split(sel_df, groups=sel_df["sample"]):
        m = GradientBoostingRegressor(loss="huber", **GB).fit(matrix(sel_df.iloc[tr], feats), sel_df["abs_err"].iloc[tr])
        err_cv.iloc[te] = m.predict(matrix(sel_df.iloc[te], feats))
    lab = sel_df["sample"] != "synthetic"
    rho = err_cv[lab].corr(sel_df.loc[lab, "abs_err"], method="spearman")
    q = pd.qcut(err_cv[lab], 3, labels=["yakin", "sedang", "ragu"])
    print(f"\nModel kesalahan (foto lab): korelasi peringkat prediksi vs selisih sebenarnya = {rho:.2f}")
    print(sel_df[lab].groupby(q.values, observed=True)["abs_err"].agg(["count", "mean", "median"]).round(1).to_string())

    # ---- model akhir pada semua data
    ranker, angle = fit_models(h, feats, syn)
    error = GradientBoostingRegressor(loss="huber", **GB).fit(matrix(sel_df, feats), sel_df["abs_err"])
    meta = dict(trained=dt.date.today().isoformat(), n_photos=len(names), n_samples=int(h["sample"].nunique()),
                n_synthetic=0 if syn is None else int(syn["name"].nunique()),
                cv=cvr, baseline=base, anchor=ANCHOR, max_correction=MAX_CORR,
                source="ImageJ (Contact Angle Theta E / LBADSA), folder KARAKTERISASI")
    out = ROOT / "ca_core" / "model"
    out.mkdir(exist_ok=True)
    Xh = matrix(h, feats)
    for name, model, kind in (("ranker", ranker, "classifier"), ("angle", angle, "regressor"),
                              ("error", error, "regressor")):
        d = export(model, feats, kind, meta, Xh)
        (out / f"{name}.json").write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
    imp = pd.Series(angle.feature_importances_, feats).sort_values(ascending=False).head(8)
    print("\nCiri terpenting model sudut:", ", ".join(f"{k} {v:.2f}" for k, v in imp.items()))
    print(f"Model disimpan di {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "latest")
