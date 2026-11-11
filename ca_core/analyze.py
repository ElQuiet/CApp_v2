"""Pipeline utama: foto → baseline → kontur tetesan → sudut kontak + pemeriksaan kualitas."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import candidates as cand_mod
from . import learned
from .baseline import find_baseline_candidates, refine_global, refine_local, rotation_for
from .detect import DropDetection, detect_drops, snap_profile
from .geometry import (ContactResult, Line, ParamCurve, contact_from_curve, curve_distance, fit_circle,
                       fit_ellipse, fit_ellipse_axis, spherical_cap_angle, tangent_poly)
from .quality import Issue, confidence, edge_profile_stats, run_checks


@dataclass
class MethodResult:
    name: str
    theta_left: float
    theta_right: float

    @property
    def theta_mean(self) -> float:
        vals = [v for v in (self.theta_left, self.theta_right) if np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan


@dataclass
class AnalysisResult:
    ok: bool
    theta: float = np.nan  # hasil utama (rata-rata kiri & kanan, metode elips)
    theta_left: float = np.nan
    theta_right: float = np.nan
    sensitivity: float = np.nan  # ±derajat per 1 px pergeseran baseline
    methods: list[MethodResult] = field(default_factory=list)
    base_width_px: float = np.nan
    height_px: float = np.nan
    issues: list[Issue] = field(default_factory=list)
    confidence: int = 0
    overlay: np.ndarray | None = None
    # data untuk koreksi manual di UI (koordinat citra asli)
    baseline_pts: tuple | None = None
    contact_pts: tuple | None = None
    roi_suggestion: tuple | None = None
    message: str = ""
    primary_method: str = "Elips"
    diagnostics: dict = field(default_factory=dict)
    scale: float = 1.0  # skala overlay terhadap foto asli
    alt_rois: list = field(default_factory=list)  # lokasi tetesan alternatif [(kotak, rasio skor)]


class _Frame:
    """Transformasi antara citra asli dan citra yang diluruskan (baseline horizontal)."""

    def __init__(self, angle_deg: float, shape):
        self.angle = angle_deg
        self.shape = shape
        self.M = rotation_for(angle_deg, shape)
        self.Mi = cv2.invertAffineTransform(self.M)

    def warp(self, img):
        h, w = self.shape[:2]
        return cv2.warpAffine(img, self.M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def to_rot(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        return pts @ self.M[:, :2].T + self.M[:, 2]

    def to_orig(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        return pts @ self.Mi[:, :2].T + self.Mi[:, 2]


def _roi_to_rot(frame: _Frame, roi):
    x0, y0, x1, y1 = roi
    c = frame.to_rot([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    return (c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max())


MAX_SIDE = 2000  # foto lebih besar diperkecil untuk analisis (sudut tidak bergantung skala)
HYPOTHESES = 4  # jumlah kandidat lokasi tetesan yang diukur ulang pada mode otomatis
DEBUG: dict = {}  # diisi data antara bila tidak kosong (untuk pengembangan, mis. DEBUG["on"] = True)


def analyze(image_bgr: np.ndarray, roi=None, baseline=None, manual_points=None) -> AnalysisResult:
    """Analisis satu foto. Tidak pernah melempar exception: kegagalan dikembalikan sebagai hasil.

    roi           : (x0, y0, x1, y1) area tetesan pada foto asli (opsional)
    baseline      : ((x1, y1), (x2, y2)) dua titik pada permukaan substrat (opsional)
    manual_points : [(x, y), ...] titik di tepi tetesan, klik manual (opsional)
    Semua koordinat input & output (kecuali overlay) dalam piksel foto asli.
    """
    try:
        if image_bgr is None or image_bgr.size == 0:
            return AnalysisResult(False, message="Gambar kosong atau tidak terbaca.")
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        h, w = image_bgr.shape[:2]
        s = min(1.0, MAX_SIDE / max(h, w))
        small = cv2.resize(image_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else image_bgr
        auto = baseline is None and manual_points is None
        n_hyp = HYPOTHESES if auto and roi is None else 1
        # Tahap 1: cari tetesan (beberapa kandidat lokasi) pada foto yang diperkecil.
        coarse = _scaled(_analyze, small, s, roi, baseline, manual_points, orig_shape=image_bgr.shape,
                         n_alt=n_hyp - 1)
        coarse.scale = s
        hints = ([(coarse.roi_suggestion, 1.0)] if coarse.roi_suggestion is not None else []) + coarse.alt_rois
        # Tahap 2: ukur ulang tiap kandidat pada potongan beresolusi penuh (tetesan sering hanya
        # ±100–300 px di foto 4000 px), lalu pilih kandidat yang benar-benar tetesan.
        fines, feats = [], []
        for i, (hint, ratio) in enumerate(hints[:n_hyp]):
            fine = None
            if not (i == 0 and s >= 1.0):  # foto kecil: potongan tidak menambah resolusi
                box = _crop_box(hint, roi, baseline, manual_points, (h, w))
                fine = _analyze_crop(image_bgr, box, hint, roi, baseline, manual_points) if box else None
                if fine is not None:
                    fine.diagnostics["box"] = box
            if i == 0 and coarse.ok and (fine is None or not fine.ok or not _consistent(fine, hint)):
                # potongan tidak menemukan tetesan yang sama (mis. terpecah jadi dua busur): pakai tahap 1
                fine = coarse
                fine.diagnostics["box"] = None
            if fine is None:
                continue
            fines.append(fine)
            feats.append(learned.hyp_features(fine, i, ratio, image_bgr.shape))
        if not fines:
            return coarse
        k = 0
        probs = learned.rank(feats) if len(fines) > 1 else None
        if probs is not None:
            k = int(np.argmax(probs))
        elif not fines[0].ok:  # tanpa model: kandidat pertama yang berhasil
            k = next((i for i, f in enumerate(fines) if f.ok), 0)
        res = fines[k]
        if res.diagnostics.get("box") is not None:
            res.overlay = _paste_overlay(small, res, res.diagnostics["box"], s)
        res.scale = s
        res.diagnostics.update(hyps=feats, chosen=k, rank_probs=None if probs is None else probs.tolist())
        if auto:
            _apply_model(res, feats[k])
        return res
    except Exception as e:  # pragma: no cover - pengaman agar satu foto rusak tidak menghentikan batch
        return AnalysisResult(
            False, message=f"Analisis gagal ({type(e).__name__}). Tandai lokasi tetesan atau gunakan titik tepi.",
            issues=[Issue("error", "analisis", "Analisis gagal",
                          f"Terjadi kesalahan saat memproses foto ini: {e}",
                          "Gunakan mode Lokasi tetesan atau Titik tepi untuk membantu deteksi.")])


def _scaled(fn, img, s: float, roi, baseline, manual_points, offset=(0.0, 0.0), **kw) -> AnalysisResult:
    """Jalankan fn pada citra `img` = foto asli (digeser `offset`, diskalakan `s`); hasil dalam koordinat foto asli."""
    ox, oy = offset

    def fwd(p):
        return ((p[0] - ox) * s, (p[1] - oy) * s)

    def back(p):
        return (p[0] / s + ox, p[1] / s + oy)

    roi_s = None if roi is None else fwd(roi[:2]) + fwd(roi[2:])
    bl_s = None if baseline is None else tuple(fwd(p) for p in baseline)
    mp_s = None if manual_points is None else [fwd(p) for p in manual_points]
    res = fn(img, roi_s, bl_s, mp_s, **kw)
    res.base_width_px /= s
    res.height_px /= s
    res.baseline_pts = None if res.baseline_pts is None else tuple(back(p) for p in res.baseline_pts)
    res.contact_pts = None if res.contact_pts is None else tuple(back(p) for p in res.contact_pts)
    res.roi_suggestion = None if res.roi_suggestion is None else back(res.roi_suggestion[:2]) + back(res.roi_suggestion[2:])
    res.alt_rois = [(back(b[:2]) + back(b[2:]), r) for b, r in res.alt_rois]
    return res


def _consistent(fine: AnalysisResult, hint) -> bool:
    """Hasil potongan mengukur tetesan berukuran sama dengan petunjuk tahap kasar."""
    if fine.roi_suggestion is None:
        return False
    wf = abs(fine.roi_suggestion[2] - fine.roi_suggestion[0])
    wh = abs(hint[2] - hint[0])
    return 0.75 <= wf / max(wh, 1e-6) <= 1.33


def _apply_model(res: AnalysisResult, feat: dict) -> None:
    """Sudut & keyakinan dari model terlatih (acuan ImageJ), bila model tersedia."""
    th = learned.angle(feat)
    if not np.isfinite(th):
        return
    err = learned.expected_error(feat)
    geo = res.theta
    res.diagnostics.update(geom_theta=geo, model_theta=th, model_err=err)
    if res.ok and np.isfinite(geo):
        d = th - geo  # pertahankan selisih kiri–kanan dari geometri
        res.theta_left = res.theta_left + d if np.isfinite(res.theta_left) else th
        res.theta_right = res.theta_right + d if np.isfinite(res.theta_right) else th
    else:
        # kontur tertutup (kamera dari atas): perkiraan dari profil atas tetesan
        if not np.isfinite(feat.get("c_ext_hw", np.nan)) and not np.isfinite(feat.get("c_ext_cir", np.nan)):
            return
        res.theta_left = res.theta_right = th
        res.issues = [i for i in res.issues if i.level != "error"] + [Issue(
            "warning", "foto", "Kamera dari atas: sudut hanya perkiraan",
            "Tetesan terlihat sebagai bentuk tertutup (tapak/pantulan tetesan ikut terlihat). Sudut "
            "diperkirakan dari profil atas tetesan dengan model yang dilatih dari pengukuran ImageJ.",
            "Untuk hasil yang lebih pasti, foto ulang dengan kamera sejajar permukaan pelat.")]
        res.ok = True
        res.message = ""
    res.theta = th
    res.methods.insert(0, MethodResult("Model (ImageJ)", res.theta_left, res.theta_right))
    res.primary_method = "Model (ImageJ)"
    model_conf = int(np.clip(100 - 5 * (err - 2), 0, 100)) if np.isfinite(err) else 100
    res.confidence = min(confidence(res.issues), model_conf)


def _crop_box(hint, roi, baseline, manual_points, shape):
    """Area (x0, y0, x1, y1) foto asli di sekitar tetesan untuk analisis resolusi penuh."""
    pts = []
    for src in (roi and [roi[:2], roi[2:]], hint and [hint[:2], hint[2:]],
                baseline, manual_points):
        if src:
            pts += [tuple(p) for p in src]
    if not pts:
        return None
    pts = np.asarray(pts, float)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    span = max(x1 - x0, 0.03 * max(shape))
    h, w = shape
    box = (int(max(0, x0 - 1.2 * span)), int(max(0, y0 - 0.9 * span)),
           int(min(w, x1 + 1.2 * span)), int(min(h, y1 + 0.6 * span)))
    if box[2] - box[0] < 64 or box[3] - box[1] < 64:
        return None
    return box


def _analyze_crop(image_bgr, box, hint, roi, baseline, manual_points):
    x0, y0, x1, y1 = box
    crop = image_bgr[y0:y1, x0:x1]
    c = min(1.0, MAX_SIDE / max(crop.shape[:2]))
    if c < 1.0:
        crop = cv2.resize(crop, None, fx=c, fy=c, interpolation=cv2.INTER_AREA)
    if roi is None and hint is not None:
        # area tetesan dari tahap kasar sebagai panduan (sedikit diperlebar)
        rx0, ry0, rx1, ry1 = hint
        rx0, rx1 = sorted([rx0, rx1])
        ry0, ry1 = sorted([ry0, ry1])
        pad = 0.15 * (rx1 - rx0)
        roi = (rx0 - pad, ry0 - pad, rx1 + pad, ry1 + 0.5 * pad)
    prefer = None
    if hint is not None:  # ukur ulang tetesan yang sama dengan tahap kasar, bukan objek lain di potongan
        prefer = ((hint[0] - x0) * c, (hint[1] - y0) * c, (hint[2] - x0) * c, (hint[3] - y0) * c)
    res = _scaled(_analyze, crop, c, roi, baseline, manual_points, offset=(x0, y0), orig_shape=image_bgr.shape,
                  n_alt=3 if prefer is not None else 0, prefer=prefer,
                  line_px=_line_px(image_bgr.shape) * c / min(1.0, MAX_SIDE / max(image_bgr.shape[:2])))
    res.alt_rois = []
    return res if res.ok or res.overlay is not None else None


def _paste_overlay(full_small: np.ndarray, fine: AnalysisResult, box, s: float) -> np.ndarray:
    """Overlay seluruh foto (skala s) dengan bagian tetesan diganti overlay resolusi penuh."""
    out = full_small.copy()
    if fine.overlay is None:
        return out
    x0, y0, x1, y1 = [int(round(v * s)) for v in box]
    x1, y1 = min(x1, out.shape[1]), min(y1, out.shape[0])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return out
    out[y0:y1, x0:x1] = cv2.resize(fine.overlay, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    return out


def _det_box(frame: _Frame, line: Line, det: DropDetection, w: int, h: int) -> tuple:
    """Kotak (x0, y0, x1, y1) di citra (belum diluruskan) di sekitar tetesan hasil deteksi."""
    p = det.points
    bx0, bx1, by0 = p[:, 0].min(), p[:, 0].max(), p[:, 1].min()
    by1 = float(line.y_at((bx0 + bx1) / 2))
    pad = 0.3 * max(bx1 - bx0, 10)
    c = frame.to_orig([[bx0 - pad, by0 - pad], [bx1 + pad, by0 - pad], [bx0 - pad, by1], [bx1 + pad, by1]])
    return (max(0.0, c[:, 0].min()), max(0.0, c[:, 1].min()), min(float(w), c[:, 0].max()), min(float(h), c[:, 1].max()))


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _analyze(image_bgr: np.ndarray, roi=None, baseline=None, manual_points=None, orig_shape=None,
             n_alt: int = 0, prefer=None, line_px: float | None = None) -> AnalysisResult:
    if image_bgr is None or image_bgr.size == 0:
        return AnalysisResult(False, message="Gambar kosong atau tidak terbaca.")
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ---------- 1. Baseline ----------
    if baseline is not None:
        (x1, y1), (x2, y2) = baseline
        ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        frame = _Frame(ang, gray.shape)
        p = frame.to_rot([[x1, y1], [x2, y2]])
        m = (p[1, 1] - p[0, 1]) / (p[1, 0] - p[0, 0]) if p[1, 0] != p[0, 0] else 0.0
        candidates = [(frame, Line(float(p[0, 1] - m * p[0, 0]), float(m)), 1, 1.0)]
    else:
        cands = []
        if roi is not None:  # lokasi tetesan dari user: baseline harus berada di dalam area itu
            y0r, y1r = sorted([roi[1], roi[3]])
            cands = find_baseline_candidates(gray, y_range=(y0r, y1r + 0.03 * h))
        if not cands:
            cands = find_baseline_candidates(gray)
        if not cands:
            return AnalysisResult(False, message="Garis permukaan substrat (baseline) tidak ditemukan.")
        candidates = []
        top_score = cands[0].score
        for c in cands[:4]:
            if c.score / top_score < 0.35:  # garis jauh lebih lemah dari kandidat terbaik: hampir tak pernah menang
                continue
            frame = _Frame(c.angle_deg, gray.shape)
            candidates.append((frame, Line(c.y_center, 0.0), c.polarity, c.score / top_score))

    # ---------- 2. Kontur tetesan ----------
    best = None
    found, alt_rois = [], []  # semua hipotesis; lokasi alternatif (kotak di citra ini, rasio skor)
    for frame, line, pol, bl_weight in candidates:
        gray_rot = frame.warp(gray)
        roi_rot = _roi_to_rot(frame, roi) if roi is not None else None
        if baseline is None:  # presisi sub-piksel di resolusi penuh
            line = refine_global(gray_rot, line, pol)
        if manual_points is not None and len(manual_points) >= 3:
            pts = frame.to_rot(manual_points)
            circ = fit_circle(pts)
            if circ is None:  # mis. semua titik segaris
                return AnalysisResult(
                    False, message="Titik tepi tidak membentuk busur. Klik titik di sepanjang tepi melengkung "
                                   "tetesan, dari titik kontak kiri sampai kanan.")
            det = DropDetection(pts, circ, 1.0, None, 0, ["Titik tepi dari input manual."])
            best = (1e9, frame, line, pol, det, gray_rot)
            break
        for det in detect_drops(gray_rot, line, roi_rot, k=1 + n_alt):
            score = (det.score if contact_from_curve(det.curve, line) is not None else -1) * bl_weight
            found.append((score, frame, line, pol, det, gray_rot))
        if baseline is not None:
            break
    if found:
        found.sort(key=lambda t: t[0], reverse=True)
        best = best or found[0]
        if prefer is not None:
            ious = [(_iou(_det_box(f[1], f[2], f[4], w, h), prefer), f[0]) for f in found]
            i_best = max(range(len(found)), key=lambda i: (ious[i][0] > 0.2, ious[i][0] * (ious[i][1] > 0)))
            if ious[i_best][0] > 0.2:
                best = found[i_best]
        for score, frame_a, line_a, _, det_a, _ in found[1:]:
            box = _det_box(frame_a, line_a, det_a, w, h)
            if score > 0 and all(_iou(box, b) < 0.3 for b, _ in alt_rois + [(_det_box(*best[1:3], best[4], w, h), 0)]):
                alt_rois.append((box, score / max(found[0][0], 1e-9)))
        alt_rois = alt_rois[:n_alt]
    if best is None or best[4] is None:
        return AnalysisResult(
            False,
            message="Tetesan tidak terdeteksi. Coba gambar kotak ROI di sekitar tetesan, atur "
                    "baseline manual, atau gunakan mode klik titik tepi.",
            issues=[Issue("error", "analisis", "Tetesan tidak terdeteksi",
                          "Tidak ada kontur berbentuk tetesan yang menyentuh baseline.",
                          "Gunakan ROI atau mode manual. Bila foto sangat gelap/buram, foto ulang "
                          "dengan backlight difus dan fokus pada tetesan.")])
    _, frame, line, pol, det, gray_rot = best

    # ---------- 3. Perhalus baseline lokal (kecuali baseline dari user) ----------
    c0 = contact_from_curve(det.curve, line)
    if baseline is None and c0 is not None and not c0.closed_above:
        xl, xr = sorted([c0.p_left[0], c0.p_right[0]])
        span = max(xr - xl, 20)
        line2 = refine_local(gray_rot, line, xl - 1.5 * span, xr + 1.5 * span, pol,
                             exclude=(xl - 0.1 * span, xr + 0.1 * span))
        if abs(line2.y_at((xl + xr) / 2) - line.y_at((xl + xr) / 2)) < 6:
            if manual_points is None:
                # kurva tetap, profil sub-piksel diambil ulang terhadap baseline yang lebih presisi
                snapped = snap_profile(gray_rot, det.curve, line2)
                if len(snapped) >= 12:
                    det.points = snapped
            line = line2

    pts = det.points
    # ---------- 4. Fitting & sudut ----------
    s_pts = line.height(pts)
    pts_fit = pts[s_pts > 0.5]
    ell = _stable_ellipse(pts_fit, line, w)
    cir = fit_circle(pts_fit)
    c_ell = contact_from_curve(ell, line) if ell is not None else None
    c_cir = contact_from_curve(cir, line) if cir is not None else None
    primary, cres = (ell, c_ell) if c_ell is not None else (cir, c_cir)
    # Cek fisik: tetesan berukuran µL yang pipih karena gravitasi punya sudut tepi sedikit di atas
    # estimasi lingkaran, bukan puluhan derajat. Selisih besar = ujung kontur tidak wajar
    # (umumnya perspektif kamera dari atas) → pakai lingkaran.
    ellipse_rejected = False
    if (c_ell is not None and c_cir is not None and not c_ell.closed_above and not c_cir.closed_above
            and c_ell.theta_mean - c_cir.theta_mean > 25):
        primary, cres, ellipse_rejected = cir, c_cir, True
    if cres is None:
        return AnalysisResult(False, message="Kurva tetesan tidak memotong baseline.")

    methods: list[MethodResult] = []
    closed = cres.closed_above
    if not closed:
        if c_ell is not None and not c_ell.closed_above:
            methods.append(MethodResult("Elips", c_ell.theta_left, c_ell.theta_right))
        if c_cir is not None and not c_cir.closed_above:
            methods.append(MethodResult("Lingkaran", c_cir.theta_left, c_cir.theta_right))
        tl, tr = tangent_poly(pts_fit, line, cres.p_left, cres.p_right)
        methods.append(MethodResult("Tangen lokal", tl, tr))
        sc = spherical_cap_angle(cres.apex_height, cres.base_width)
        methods.append(MethodResult("Tinggi/lebar (θ/2)", sc, sc))

    # Sensitivitas terhadap baseline ±1 px
    sens = np.nan
    if not closed:
        diffs = []
        for dy in (-1.0, 1.0):
            cc = contact_from_curve(primary, line.shifted(dy))
            if cc is not None and not cc.closed_above:
                diffs.append(abs(cc.theta_mean - cres.theta_mean))
        sens = float(max(diffs)) if diffs else np.nan

    # ---------- 5. Pemeriksaan kualitas ----------
    edge_w, contrast = edge_profile_stats(gray_rot, primary, line)
    rms = float(np.sqrt(np.mean(curve_distance(primary, pts_fit) ** 2))) if len(pts_fit) else np.nan
    means = [mm.theta_mean for mm in methods if np.isfinite(mm.theta_mean)]
    spread = float(max(means) - min(means)) if len(means) > 1 else 0.0
    # bounding box tetesan (citra lurus)
    bx0, bx1 = pts[:, 0].min(), pts[:, 0].max()
    by0 = pts[:, 1].min()
    by1 = float(line.y_at((bx0 + bx1) / 2))
    pad = 0.3 * max(bx1 - bx0, 10)
    gx0, gx1 = int(max(0, bx0 - pad)), int(min(w, bx1 + pad))
    gy0, gy1 = int(max(0, by0 - pad)), int(min(h, by1))
    region = gray_rot[gy0:gy1, gx0:gx1]
    glare = float((region >= 250).mean()) if region.size else 0.0
    clutter = 0.0
    if det.edge_map is not None and region.size:
        em = det.edge_map[gy0:gy1, gx0:gx1].copy()
        em_pts = np.column_stack(np.nonzero(em)[::-1]).astype(float) + [gx0, gy0]
        if len(em_pts):
            far = curve_distance(primary, em_pts) > 4
            clutter = float(far.sum()) / max(region.size, 1)
    margin = 3
    touches = bool(bx0 <= margin or bx1 >= w - margin or by0 <= margin)
    if roi is not None:
        rr = _roi_to_rot(frame, roi)
        touches = touches or bx0 <= rr[0] + margin or bx1 >= rr[2] - margin or by0 <= rr[1] + margin

    surface = _surface_visible(gray_rot, line, bx0, bx1) if not closed else 0.0
    vfa = surface if surface > max(30.0, 0.035 * h) else None
    ctx = dict(
        image_shape=orig_shape if orig_shape is not None else image_bgr.shape, base_width=cres.base_width if not closed else bx1 - bx0,
        apex_height=cres.apex_height, theta=np.nan if closed else cres.theta_mean,
        closed_above=closed, view_from_above=vfa, baseline_angle=frame.angle + np.degrees(np.arctan(line.m)),
        asymmetry=abs(cres.theta_left - cres.theta_right) if not closed else np.nan,
        sensitivity=sens, edge_width=edge_w, edge_contrast=contrast, glare_frac=glare,
        touches_border=touches, clutter=clutter, support=det.support, fit_rms=rms,
        method_spread=spread, ellipse_rejected=ellipse_rejected, surface_visible=surface,
        primary_method="Lingkaran" if primary is cir else "Elips",
    )
    issues = run_checks(ctx)
    ctx["cand"] = cand_mod.compute(pts, line, primary, cres, ctx)

    # ---------- 6. Overlay ----------
    overlay = draw_overlay(image_bgr, frame, line, primary, pts, cres, closed, line_px)
    if DEBUG:
        DEBUG.update(gray_rot=gray_rot, line=line, curve=primary, pts=pts, frame=frame, det=det)

    bl_rot = np.array([[0, line.y_at(0)], [w, line.y_at(w)]])
    bl_orig = frame.to_orig(bl_rot)
    res = AnalysisResult(
        ok=not closed,
        theta=np.nan if closed else cres.theta_mean,
        theta_left=np.nan if closed else cres.theta_left,
        theta_right=np.nan if closed else cres.theta_right,
        sensitivity=sens, methods=methods,
        base_width_px=ctx["base_width"], height_px=cres.apex_height,
        issues=issues, confidence=confidence(issues), overlay=overlay,
        baseline_pts=(tuple(bl_orig[0]), tuple(bl_orig[1])),
        contact_pts=None if closed else (tuple(frame.to_orig(cres.p_left)[0]), tuple(frame.to_orig(cres.p_right)[0])),
        roi_suggestion=tuple(np.r_[frame.to_orig([gx0, gy0])[0], frame.to_orig([gx1, gy1])[0]]),
        message="; ".join(det.notes),
        primary_method=ctx["primary_method"],
        diagnostics={k: v for k, v in ctx.items() if k != "image_shape"},
        alt_rois=alt_rois,
    )
    res.diagnostics["det_score"] = float(best[0])
    if closed:
        res.message = "Sudut kontak tidak dapat diukur dari foto ini (lihat peringatan)."
    return res


def _stable_ellipse(pts, line: Line, w: int) -> ParamCurve | None:
    """Elips bersumbu vertikal sebagai acuan stabil; elips bebas (bisa asimetris kiri/kanan)
    hanya dipakai bila hasilnya konsisten dengan acuan."""
    ax = fit_ellipse_axis(pts)
    free = fit_ellipse(pts)
    c_ax = contact_from_curve(ax, line) if ax is not None else None
    if free is None or c_ax is None or c_ax.closed_above:
        return ax if ax is not None else free
    tilt = np.degrees(free.phi) % 90
    if min(tilt, 90 - tilt) > 15 or max(free.a, free.b) > 3 * w:
        return ax
    c_fr = contact_from_curve(free, line)
    if c_fr is None or c_fr.closed_above or abs(c_fr.theta_mean - c_ax.theta_mean) > 8:
        return ax
    return free


def _surface_visible(gray_rot: np.ndarray, line: Line, bx0: float, bx1: float) -> float:
    """Tinggi (px) area tajam/bertekstur di atas baseline, di samping tetesan.

    Pada foto samping yang benar, di belakang tetesan terlihat latar jauh yang buram (halus).
    Bila kamera dari atas, permukaan atas pelat ikut terlihat: tajam & bertekstur karena berada
    pada jarak fokus yang sama dengan tetesan. Diukur dari ketajaman (Laplacian) per pita baris
    dibanding tingkat ketajaman latar.
    """
    h, w = gray_rot.shape
    span = max(bx1 - bx0, 20)
    cols = np.r_[np.arange(int(max(0, bx0 - 2.5 * span)), int(max(0, bx0 - 0.1 * span))),
                 np.arange(int(min(w, bx1 + 0.1 * span)), int(min(w, bx1 + 2.5 * span)))]
    if len(cols) < 20:
        return 0.0
    b = cv2.GaussianBlur(gray_rot, (0, 0), 1.2 * max(1.0, w / 1280))
    lap = np.abs(cv2.Laplacian(b.astype(np.float32), cv2.CV_32F, ksize=3))
    step = max(4, int(round(h / 240)))
    yb = float(line.y_at((bx0 + bx1) / 2))
    vals, ds = [], []
    for d in range(step, int(min(yb - step, h / 4)), step):
        y0, y1 = int(yb - d - step), int(yb - d)
        if y0 < 0:
            break
        vals.append(float(lap[y0:y1, cols].mean()))
        ds.append(d + step)
    if len(vals) < 10:
        return 0.0
    vals = np.asarray(vals)
    far = float(np.percentile(vals[len(vals) // 2:], 30))
    above = vals > 1.6 * far
    # tinggi area tajam dari baseline ke atas; toleransi celah pendek (≤2 pita)
    last, gap = -1, 0
    for i, a in enumerate(above):
        if a:
            last, gap = i, 0
        else:
            gap += 1
            if gap > 2:
                break
    return float(ds[last]) if last >= 0 else 0.0


def _to_int(p):
    return tuple(int(round(v)) for v in p)


def _line_px(shape) -> int:
    """Tebal garis overlay (px) pada overlay seluruh foto (sisi terpanjang ≤ MAX_SIDE)."""
    return max(1, int(round(min(max(shape[:2]), MAX_SIDE) / 700)))


def draw_overlay(img, frame: _Frame, line: Line, curve: ParamCurve, pts, cres: ContactResult, closed: bool,
                 line_px: float | None = None):
    out = img.copy()
    h, w = out.shape[:2]
    th = max(1, int(round(line_px))) if line_px else max(1, int(round(w / 700)))
    # baseline
    bl = frame.to_orig([[0, line.y_at(0)], [w, line.y_at(w)]])
    cv2.line(out, _to_int(bl[0]), _to_int(bl[1]), (40, 190, 255), th, cv2.LINE_AA)
    # titik tepi
    for p in frame.to_orig(pts[:: max(1, len(pts) // 400)]):
        cv2.circle(out, _to_int(p), max(1, th // 2), (235, 235, 235), -1, cv2.LINE_AA)
    # kurva fit (bagian di atas baseline)
    poly = curve.polyline(1500)
    s = line.height(poly)
    seg = frame.to_orig(poly)
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        if s[i] > -1 and s[j] > -1:
            cv2.line(out, _to_int(seg[i]), _to_int(seg[j]), (230, 200, 30), th, cv2.LINE_AA)
    if not closed:
        e = line.direction
        L = max(cres.base_width, cres.apex_height) * 0.6
        for p, ang, sgn in ((cres.p_left, cres.theta_left, 1), (cres.p_right, cres.theta_right, -1)):
            a = np.radians(ang)
            # vektor tangen di frame lurus (y ke bawah)
            v = sgn * e * np.cos(a) + np.array([e[1], -e[0]]) * np.sin(a)
            p0, p1 = frame.to_orig(p)[0], frame.to_orig(p + v * L)[0]
            cv2.line(out, _to_int(p0), _to_int(p1), (60, 60, 230), th, cv2.LINE_AA)
            cv2.circle(out, _to_int(p0), 2 * th, (60, 60, 230), -1, cv2.LINE_AA)
    return out
