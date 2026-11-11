"""Kumpulkan nilai acuan ImageJ dari folder KARAKTERISASI → data/reference_imagej.csv.

Pakai:  .venv\\Scripts\\python scripts\\build_reference.py "C:\\...\\KARAKTERISASI"

Setiap baris CSV = satu foto tetesan (titikN) beserta sudut acuan dari ImageJ:
- plugin "Contact Angle" (tabel Theta C/Theta E): dipakai Theta E (elips), sama seperti metode utama app;
- plugin LBADSA (hanya angka "Contact angle", 3 desimal).
Kolom `ref_ok` = 0 bila keluaran plugin sendiri menunjukkan pengukuran yang tidak masuk akal
(sudut kiri/kanan negatif, baseline miring jauh, kiri–kanan berbeda sangat besar).
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "reference_imagej.csv"

BASE = "01 - Baseline/Contact angle"
PRE = "02 - Pre-Degreasing/Contact Angle"
ECL = "03 - ECL Treatment/05-CONTACT ANGLE"
WB_BASE = f"{BASE}/Contact_Angle_Baseline.xlsx"
WB_PRE = f"{PRE}/Data Contact Angle.xlsx"
WB_ECL = f"{ECL}/Contact angle_ECL Experiment.xlsx"
_B = f"{PRE}/Data sebelum pre-degreasing"
_A = f"{PRE}/Data sesudah pre_degreasing"
_E = ECL

# (workbook, sheet, judul blok atau None = seluruh sheet, folder foto, sampel, tahap)
BLOCKS = [
    (WB_BASE, "Entry 0.3", None, f"{BASE}/0.3_Entry", "Entry 0.3", "baseline"),
    (WB_BASE, "Exit 0.3", None, f"{BASE}/Exit_0.3", "Exit 0.3", "baseline"),
    (WB_PRE, "Sample 1 (5)", None, f"{_B}/Sampel 1 entry 03-20260928T091541Z-1-001/Sampel 1 entry 03", "Entry 03", "before_pre"),
    (WB_PRE, "Sample 2 (10)", None, f"{_B}/Sampel 2 entry 07-20260928T091550Z-1-001/Sampel 2 entry 07", "Entry 07", "before_pre"),
    (WB_PRE, "Sample 3 (15)", None, f"{_B}/Sampel 3 exit 03-20260928T091546Z-1-001/Sampel 3 exit 03", "Exit 03", "before_pre"),
    (WB_PRE, "Sample 4 (20)", None, f"{_B}/Sampel 4 exit 07-20260928T091543Z-1-001/Sampel 4 exit 07", "Exit 07", "before_pre"),
    (WB_PRE, "Sampel 27.A-L-Pre-Cont", None, f"{_B}/Sampel 27.A-L-Pre-Cont-20260928T091602Z-1-001/Sampel 27.A-L-Pre-Cont", "27.A-L", "before_pre"),
    (WB_PRE, "Sampel 37.A-C-Pre-Cont", None, f"{_B}/Sampel 37.A-C-Pre-Cont-20260928T091558Z-1-001/Sampel 37.A-C-Pre-Cont", "37.A-C", "before_pre"),
    (WB_PRE, "Sampel 47.A-R-Pre-Cont", None, f"{_B}/Sampel 47.A-R-Pre-Cont-20260928T091554Z-1-001/Sampel 47.A-R-Pre-Cont", "47.A-R", "before_pre"),
    (WB_PRE, "Pre_5_BL-E-03-CONT", None, f"{_A}/Pre_5_BL-E-03-CONT-20260928T090506Z-1-001/Pre_5_BL-E-03-CONT", "Entry 03", "after_pre"),
    (WB_PRE, "Pre_10_BL-E-07-CONT", None, f"{_A}/Pre_10_BL-E-07-CONT-20260928T090503Z-1-001/Pre_10_BL-E-07-CONT", "Entry 07", "after_pre"),
    (WB_PRE, "Pre_15_EX-03-CONT", None, f"{_A}/Pre_15_EX-03-CONT-20260928T090457Z-1-001/Pre_15_EX-03-CONT", "Exit 03", "after_pre"),
    (WB_PRE, "Pre_20_BL-EX-07-CONT", None, f"{_A}/Pre_20_BL-EX-07-CONT-20260928T084131Z-1-001/Pre_20_BL-EX-07-CONT", "Exit 07", "after_pre"),
    (WB_PRE, "Sampel 27 After Pre", None, f"{_A}/Pre_27_A-L-PRE-CONT-20260928T090440Z-1-001/Pre_27_A-L-PRE-CONT", "27.A-L", "after_pre"),
    (WB_PRE, "Sampel 37 After Pre", None, f"{_A}/Pre_37_A-C-PRE-CONT-20260928T090454Z-1-001/Pre_37_A-C-PRE-CONT", "37.A-C", "after_pre"),
    (WB_PRE, "Sampel 47 After Pre", None, f"{_A}/Pre_47_A-R-PRE-CONT-20260928T090450Z-1-001/Pre_47_A-R-PRE-CONT", "47.A-R", "after_pre"),
    (WB_ECL, "Before pre degreasing and ECL", "32.A-L-Post-Cont", f"{_E}/Before Pre Degreasing and ECL/32.A-L-Post-Cont", "32.A-L", "before_pre"),
    (WB_ECL, "Before pre degreasing and ECL", "42.A-C-Post-Cont", f"{_E}/Before Pre Degreasing and ECL/42.A-C-Post-Cont", "42.A-C", "before_pre"),
    (WB_ECL, "Before pre degreasing and ECL", "52.A-R-Post-Cont", f"{_E}/Before Pre Degreasing and ECL/52.A-R-Post-Cont", "52.A-R", "before_pre"),
    (WB_ECL, "Before pre degreasing and ECL", "49.G1-L-03 Cont_Ang", f"{_E}/Before Pre Degreasing and ECL/49.G1-L-03 Cont_Ang", "49.G1-L", "before_pre"),
    (WB_ECL, "Before pre degreasing and ECL", "58.G1-C-03 Cont_Ang", f"{_E}/Before Pre Degreasing and ECL/58.G1-C-03 Cont_Angle", "58.G1-C", "before_pre"),
    (WB_ECL, "Before pre degreasing and ECL", "67.G1-R-03 Cont_Ang", f"{_E}/Before Pre Degreasing and ECL/67.G1-R-03 Cont_Ang", "67.G1-R", "before_pre"),
    (WB_ECL, "After pre degreasing", "32.A-L-Post-Cont", f"{_E}/After Pre Degreasing/32.A-L-Post-Cont", "32.A-L", "after_pre"),
    (WB_ECL, "After pre degreasing", "42.A-C_Post_Cont", f"{_E}/After Pre Degreasing/42.A-C-Post-Cont", "42.A-C", "after_pre"),
    (WB_ECL, "After pre degreasing", "52.A-C_Post_Cont", f"{_E}/After Pre Degreasing/52.A-R-Post-Cont", "52.A-R", "after_pre"),
    (WB_ECL, "After pre degreasing", "49.G1-L-03 Cont_Ang", f"{_E}/After Pre Degreasing/49.G1-L-03 Cont_Ang ( after pre )", "49.G1-L", "after_pre"),
    (WB_ECL, "After pre degreasing", "58.G1-C-03 Cont_Ang", f"{_E}/After Pre Degreasing/58.G1-C-03-Cont_Ang ( After pre )", "58.G1-C", "after_pre"),
    (WB_ECL, "After pre degreasing", "67.G1-R-03 Cont_Ang", f"{_E}/After Pre Degreasing/67.G1-R-03-Cont_Angle ( after pre )", "67.G1-R", "after_pre"),
    (WB_ECL, "After pre degreasing", "Sampel G2-DS-CO3-CA-TT-GRAV", f"{_E}/After Pre Degreasing/Sampel G2-DS-CO3-CA-TT-GRAV", "G2-DS-C", "after_pre"),
    (WB_ECL, "After pre degreasing", "Sampel G2-DS-LO3-CA-TT-GRAV", f"{_E}/After Pre Degreasing/Sampel G2-DS-LO3-CA-TT-GRAV", "G2-DS-L", "after_pre"),
    (WB_ECL, "After pre degreasing", "Sampel G2-DS-RO3-CA-TT-GRAV", f"{_E}/After Pre Degreasing/Sampel G2-DS-RO3-CA-TT-GRAV", "G2-DS-R", "after_pre"),
    (WB_ECL, "After ECL", "49.G1-L-03 Cont_Ang", f"{_E}/After ECL/49.G1-L-03_Cont_Ang( ECL )", "49.G1-L", "after_ecl"),
    (WB_ECL, "After ECL", "58.G1-C-03 Cont_Ang", f"{_E}/After ECL/58.G1-C-03_Cont ( ECL )", "58.G1-C", "after_ecl"),
    (WB_ECL, "After ECL", "67.G1-R-03 Cont_Ang", f"{_E}/After ECL/67.G1-R-03_Cont ( ECL )", "67.G1-R", "after_ecl"),
    (WB_ECL, "After ECL", "Sampel G2-DS-CO3-CA-TT-GRAV", f"{_E}/After ECL/Sampel G2-DS-CO3-CA-TT-GRAV (ECL)", "G2-DS-C", "after_ecl"),
    (WB_ECL, "After ECL", "Sampel G2-DS-LO3-CA-TT-GRAV", f"{_E}/After ECL/Sampel G2-DS-LO3-CA-TT-GRAV (ECL)", "G2-DS-L", "after_ecl"),
    (WB_ECL, "After ECL", "Sampel G2-DS-RO3-CA-TT-GRAV", f"{_E}/After ECL/Sampel G2-DS-RO3-CA-TT-GRAV (ECL)", "G2-DS-R", "after_ecl"),
]

TITIK = re.compile(r"titik\s*(\d+)", re.I)


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _block_rows(rows: list[tuple], title: str | None):
    """Baris milik satu blok: dari judul sampai baris rata-rata/judul berikutnya."""
    if title is None:
        return rows
    start = next(i for i, r in enumerate(rows) if any(isinstance(v, str) and v.strip() == title for v in r))
    out = []
    for r in rows[start + 1:start + 12]:
        texts = [v for v in r if isinstance(v, str)]
        if any(t.strip() != title and re.match(r"^(\d\d\.|Sampel|14 Okt)", t.strip()) for t in texts):
            break
        out.append(r)
    return out


def parse_block(rows, title):
    rows = _block_rows(rows, title)
    hdr = {}
    items = {}
    for r in rows:
        cells = [str(v).strip().lower() if isinstance(v, str) else v for v in r]
        if "theta e" in cells or "contact angle" in cells or "sudut kontak" in cells:
            hdr = {c: j for j, c in enumerate(cells) if isinstance(c, str)}
            continue
        m = next((TITIK.search(v) for v in r if isinstance(v, str) and TITIK.search(v)), None)
        if m:
            n = int(m.group(1))
        else:  # nomor urut di kolom indeks (mis. "1.0")
            idx = [v for v in r[:3] if isinstance(v, (int, float))]
            if not idx or not float(idx[0]).is_integer() or not 1 <= idx[0] <= 6:
                continue
            n = int(idx[0])
        get = (lambda k: _num(r[hdr[k]]) if k in hdr and hdr[k] < len(r) else None)
        if "theta e" in hdr:
            theta, src = get("theta e"), "imagej_contact_angle"
            left, right, ang = get("theta left"), get("theta right"), get("angle")
        else:
            nums = [_num(v) for v in r if _num(v) is not None and not (isinstance(v, (int, float)) and v == n)]
            theta, src = (nums[-1] if nums else None), "imagej_lbadsa"
            left = right = ang = None
        if theta is None or n in items:
            continue
        ok = 0 < theta < 180
        if left is not None and right is not None:
            ok &= 0 < left < 180 and 0 < right < 180 and abs(left - right) <= 20
        if ang is not None:
            ok &= abs(abs(ang) - 180) <= 8
        items[n] = dict(theta_ref=theta, theta_left=left, theta_right=right, ref_source=src, ref_ok=int(ok))
    return items


def find_photo(folder: Path, n: int):
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and re.fullmatch(rf"titik\s*{n}", p.stem, re.I):
            return p
    return None


def main(dataset: Path):
    books, out = {}, []
    for wb, sheet, title, folder, sample, stage in BLOCKS:
        if wb not in books:
            books[wb] = openpyxl.load_workbook(dataset / wb, data_only=True, read_only=True)
        rows = [tuple(r) for r in books[wb][sheet].iter_rows(values_only=True)]
        items = parse_block(rows, title)
        if not items:
            print(f"!! tidak ada nilai: {sheet} / {title}")
        for n, it in sorted(items.items()):
            photo = find_photo(dataset / folder, n)
            if photo is None:
                print(f"!! foto tidak ada: {folder} titik {n}")
                continue
            out.append(dict(image=photo.relative_to(dataset).as_posix(), sample=sample, stage=stage, titik=n,
                            **{k: ("" if v is None else round(v, 3) if isinstance(v, float) else v)
                               for k, v in it.items()}))
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    n_ok = sum(r["ref_ok"] for r in out)
    print(f"{len(out)} foto berlabel ({n_ok} label andal) → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
