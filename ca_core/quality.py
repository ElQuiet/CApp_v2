"""Pemeriksaan kualitas foto & analisis.

Setiap temuan diberi `source`:
- "foto"     → masalah berasal dari cara pengambilan foto (posisi kamera, fokus,
               cahaya, resolusi). Software tidak bisa memperbaikinya; user perlu foto ulang.
- "analisis" → ketidakpastian dari proses deteksi/fitting; periksa overlay atau koreksi manual.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

LEVEL_PENALTY = {"error": 60, "warning": 15, "info": 3}


@dataclass
class Issue:
    level: str  # "error" | "warning" | "info"
    source: str  # "foto" | "analisis"
    title: str
    detail: str
    advice: str = ""


def edge_profile_stats(gray: np.ndarray, curve, line, n: int = 80, half: int = 8):
    """Lebar tepi (blur) dan kontras di sepanjang kontur tetesan."""
    t = np.linspace(0, 2 * np.pi, 4 * n, endpoint=False)
    pts = curve.point(t)
    keep = line.height(pts) > 3
    pts, t = pts[keep], t[keep]
    if len(pts) < 5:
        return np.nan, np.nan
    sel = np.linspace(0, len(pts) - 1, min(n, len(pts))).astype(int)
    pts, t = pts[sel], t[sel]
    d = curve.deriv(t)
    nrm = np.column_stack([d[:, 1], -d[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    offs = np.arange(-half, half + 1, 0.5)
    xs = pts[:, 0:1] + nrm[:, 0:1] * offs
    ys = pts[:, 1:2] + nrm[:, 1:2] * offs
    prof = map_coordinates(gray.astype(np.float32), [ys.ravel(), xs.ravel()], order=1, mode="nearest")
    prof = prof.reshape(xs.shape)
    inner = prof[:, offs < -half / 2].mean(1)
    outer = prof[:, offs > half / 2].mean(1)
    step = np.abs(outer - inner)
    grad = np.abs(np.diff(prof, axis=1)).max(1) / 0.5
    width = step / np.maximum(grad, 1e-6)
    good = step > 5
    if good.sum() < 5:
        return np.nan, float(np.median(step))
    return float(np.median(width[good])), float(np.median(step))


def run_checks(ctx: dict) -> list[Issue]:
    """ctx berisi hasil-hasil antara dari pipeline (lihat analyze.py)."""
    issues: list[Issue] = []
    img_h, img_w = ctx["image_shape"][:2]
    width_px, height_px = ctx["base_width"], ctx["apex_height"]
    theta = ctx["theta"]

    # ---------- Masalah dari pengambilan foto ----------
    if ctx.get("closed_above"):
        issues.append(Issue(
            "error", "foto", "Kamera mengambil dari atas / tetesan tidak di tepi depan pelat",
            "Kontur tetesan terlihat sebagai bentuk tertutup (elips) di atas permukaan, tidak "
            "memotong garis permukaan. Ini terjadi bila lensa melihat dari atas sehingga yang "
            "terlihat adalah 'tapak' tetesan, bukan profil sampingnya. Sudut kontak tidak dapat "
            "diukur dengan benar dari foto ini.",
            "Posisikan lensa sejajar permukaan pelat (kemiringan 0–2°), letakkan tetesan dekat "
            "tepi depan pelat, lalu foto ulang."))
    elif ctx.get("view_from_above"):
        issues.append(Issue(
            "warning", "foto", "Kamera kemungkinan mengambil dari atas",
            f"Permukaan atas pelat yang bertekstur ikut terlihat tajam hingga ±{ctx['view_from_above']:.0f} px "
            "di atas tepi depan. Artinya lensa tidak sejajar permukaan, sehingga yang terlihat bukan "
            "profil samping murni. Perspektif ini membuat sudut kontak terbaca berbeda dari nilai "
            "sebenarnya (umumnya lebih besar), terutama untuk sudut rendah.",
            "Turunkan posisi kamera hingga sejajar permukaan pelat (permukaan atas pelat tidak "
            "terlihat lagi, hanya tepinya)."))

    tilt = abs(ctx.get("baseline_angle", 0.0))
    if tilt > 2.0:
        issues.append(Issue(
            "warning" if tilt > 4 else "info", "foto", f"Foto/substrat miring {tilt:.1f}°",
            "Kemiringan baseline sudah dikoreksi otomatis. Namun bila yang miring adalah "
            "substratnya (bukan kamera), tetesan bisa mengalir dan sudut kiri–kanan berbeda.",
            "Gunakan meja/sample stage yang rata (cek dengan waterpass) dan kamera yang tegak."))

    asym = ctx.get("asymmetry", 0.0)
    if np.isfinite(asym) and asym > 8:
        issues.append(Issue(
            "warning", "foto", f"Sudut kiri–kanan berbeda {asym:.1f}°",
            "Tetesan asimetris. Penyebab umum: substrat miring, kamera tidak sejajar, tetesan "
            "masih bergerak, atau permukaan tidak homogen (pinning/kontaminasi lokal).",
            "Pastikan substrat rata, tunggu tetesan stabil (±5–10 detik), dan foto tegak lurus "
            "dari samping. Bila tetap asimetris, kemungkinan memang sifat permukaannya."))

    if width_px < 150 or height_px < 20:
        issues.append(Issue(
            "warning", "foto", "Tetesan terlalu kecil di dalam foto",
            f"Lebar tetesan hanya ±{width_px:.0f} px dan tinggi ±{height_px:.0f} px. Selisih "
            f"1 piksel saja mengubah sudut sekitar ±{ctx.get('sensitivity', np.nan):.1f}°.",
            "Dekatkan kamera atau gunakan lensa makro/zoom optik sehingga tetesan mengisi "
            "minimal ±30% lebar foto."))
    if max(img_h, img_w) < 1600:
        issues.append(Issue(
            "info", "foto", f"Resolusi foto rendah ({img_w}×{img_h})",
            "Foto kemungkinan sudah dikompresi (mis. dikirim lewat WhatsApp/chat).",
            "Gunakan file asli dari galeri/kamera (resolusi penuh)."))

    blur_w, contrast = ctx.get("edge_width", np.nan), ctx.get("edge_contrast", np.nan)
    if np.isfinite(blur_w) and blur_w > 4.5:
        issues.append(Issue(
            "warning", "foto", "Foto kurang fokus pada tetesan",
            f"Tepi tetesan melebar ±{blur_w:.1f} px (tepi tajam ±1–3 px), sehingga posisi tepi "
            "kurang pasti.",
            "Ketuk layar pada tetesan untuk mengunci fokus (tap-to-focus), gunakan tripod, "
            "dan hindari zoom digital."))
    if np.isfinite(contrast) and contrast < 12:
        issues.append(Issue(
            "warning", "foto", "Kontras tetesan terhadap latar rendah",
            "Tetesan sulit dibedakan dari latar belakang.",
            "Gunakan backlight yang rata dan difus (lampu ditutup kertas kalkir/plastik putih "
            "susu) agar tetesan tampak sebagai siluet gelap yang tegas."))
    if ctx.get("glare_frac", 0) > 0.15:
        issues.append(Issue(
            "info", "foto", "Ada area silau/overexposure di sekitar tetesan",
            f"±{100 * ctx['glare_frac']:.0f}% piksel di sekitar tetesan jenuh (putih penuh).",
            "Gunakan cahaya difus, jangan arahkan lampu titik langsung ke lensa, atau turunkan "
            "exposure kamera."))
    if ctx.get("touches_border"):
        issues.append(Issue(
            "warning", "foto", "Tetesan terpotong tepi foto/ROI",
            "Sebagian kontur berada di luar area gambar.",
            "Posisikan tetesan di tengah frame atau perbesar kotak ROI."))
    if ctx.get("ellipse_rejected"):
        issues.append(Issue(
            "warning", "foto", "Ujung kontur tetesan tidak wajar",
            "Fit elips menghasilkan ujung tetesan yang hampir tegak, tidak sesuai dengan tinggi dan "
            "lebar tetesan. Ini biasanya akibat perspektif kamera dari atas (garis kontak depan ikut "
            "terlihat) atau bayangan di ujung tetesan. Sudut dihitung dengan metode lingkaran.",
            "Foto ulang dengan kamera sejajar permukaan pelat."))
    if ctx.get("clutter", 0) > 0.12:
        issues.append(Issue(
            "info", "foto", "Banyak droplet kecil/kotoran di sekitar tetesan",
            "Permukaan di sekitar tetesan bertekstur atau basah sehingga deteksi tepi lebih sulit.",
            "Keringkan permukaan dan gunakan area pelat yang bersih untuk setiap tetesan."))

    # ---------- Ketidakpastian analisis ----------
    sens = ctx.get("sensitivity", np.nan)
    if np.isfinite(sens) and sens > 3:
        issues.append(Issue(
            "warning", "analisis", f"Hasil sensitif terhadap posisi baseline (±{sens:.1f}°/px)",
            "Menggeser baseline 1 piksel mengubah sudut cukup besar. Ini umum pada sudut rendah "
            "atau tetesan kecil.",
            "Periksa posisi baseline pada overlay; geser manual bila perlu."))
    if ctx.get("support", 1) < 0.5:
        issues.append(Issue(
            "warning", "analisis", "Kontur tetesan hanya terdeteksi sebagian",
            f"Hanya ±{100 * ctx['support']:.0f}% kontur yang didukung tepi yang jelas.",
            "Periksa overlay. Bila tidak pas, gambar kotak ROI di sekitar tetesan atau gunakan "
            "mode klik titik manual."))
    if ctx.get("fit_rms", 0) > 1.5:
        issues.append(Issue(
            "warning", "analisis", "Bentuk kontur tidak teratur",
            f"Rata-rata simpangan titik tepi ke kurva fit {ctx['fit_rms']:.1f} px.",
            "Periksa overlay; mungkin ada tepi lain (kotoran/kilau) yang ikut terdeteksi."))
    spread = ctx.get("method_spread", 0)
    if np.isfinite(spread) and spread > 6:
        issues.append(Issue(
            "info", "analisis", f"Metode fitting berbeda hingga {spread:.1f}°",
            "Elips, lingkaran, dan tangen lokal memberikan hasil cukup berbeda; bentuk tetesan "
            "kemungkinan pipih karena gravitasi atau kontur dekat titik kontak kurang jelas.",
            "Untuk tetesan besar/pipih gunakan hasil elips atau tangen; untuk tetesan kecil "
            "hasil lingkaran sudah memadai."))
    if np.isfinite(theta) and theta < 20:
        issues.append(Issue(
            "info", "analisis", "Sudut sangat rendah (<20°)",
            "Pada sudut rendah tinggi tetesan hanya beberapa piksel, sehingga ketidakpastian "
            "relatif besar untuk metode foto samping.",
            "Gunakan perbesaran lebih tinggi dan pastikan kamera benar-benar sejajar permukaan."))
    if np.isfinite(theta) and theta > 150:
        issues.append(Issue(
            "info", "analisis", "Sudut sangat tinggi (>150°)",
            "Titik kontak pada permukaan superhidrofobik sulit ditentukan secara presisi.",
            "Pastikan baseline tepat dan pencahayaan menampilkan celah di bawah tetesan."))
    return issues


def confidence(issues: list[Issue]) -> int:
    score = 100 - sum(LEVEL_PENALTY.get(i.level, 0) for i in issues)
    return int(np.clip(score, 0, 100))
