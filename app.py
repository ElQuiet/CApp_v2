"""Contact Angle Analyzer — antarmuka web (Streamlit).

Alur: Upload → Atur → Hasil.
Jalankan:  .venv\\Scripts\\streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
import io
import math
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ca_core import analyze  # noqa: E402
from ca_core.worker import analyze_bytes  # noqa: E402
from ca_ui import LRUCache, clicker, jpeg_data_url, strip  # noqa: E402
from ca_ui.report import (annotated_zip, compare_groups, dataset_zip, describe_comparison,  # noqa: E402
                          html_report, text_summary)

PAL = {
    "light": dict(ink="#1C2522", muted="#66726D", faint="#A9B4AF", rule="#DCE2DF", accent="#0F6E7C",
                  surface="#FFFFFF", bg="#F5F7F6", track="#E7ECEA", soft="rgba(15,110,124,.12)",
                  lift="0 6px 18px rgba(15,110,124,.28)", warn="#C08A2E", err="#A33A2C", good="#3F8F5A"),
    "dark": dict(ink="#E3E8E5", muted="#9AA6A0", faint="#5F6B66", rule="#2E3935", accent="#3FA7B5",
                 surface="#1F2825", bg="#161D1B", track="#1F2825", soft="rgba(63,167,181,.18)",
                 lift="0 6px 20px rgba(0,0,0,.55)", warn="#D9A246", err="#D9695A", good="#5DB37A"),
}
STEPS = {1: "Upload", 2: "Atur", 3: "Hasil"}
MODES = {"auto": "Otomatis", "locate": "Lokasi tetesan", "baseline": "Baseline", "points": "Titik tepi"}
MODE_HELP = {
    "auto": "Scroll untuk zoom dan seret untuk menggeser. Bila tetesan yang terdeteksi salah, pilih Lokasi tetesan.",
    "locate": "Klik tetesan yang benar atau seret kotak di sekelilingnya; deteksi diulang di area itu.",
    "baseline": "Klik titik kontak kiri lalu kanan pada permukaan substrat.",
    "points": "Klik di tepi tetesan, lalu tekan Hitung. Kaca pembesar membantu menempatkan titik.",
}
WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))


def css(p: dict, step: int) -> str:
    return f"""
<style>
:root {{ --ca-ink:{p['ink']}; --ca-muted:{p['muted']}; --ca-faint:{p['faint']}; --ca-rule:{p['rule']};
        --ca-accent:{p['accent']}; --ca-surface:{p['surface']}; --ca-track:{p['track']}; --ca-soft:{p['soft']};
        --ca-lift:{p['lift']}; --ca-warn:{p['warn']}; --ca-err:{p['err']}; }}
.block-container {{ max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebarCollapsedControl"], section[data-testid="stSidebar"], [data-testid="stMainMenu"] {{ display: none; }}
[data-testid="stHeaderActionElements"] {{ display: none; }}
[data-stale="true"] {{ opacity: 1 !important; transition: none !important; }}
.st-key-theme_js {{ display: none; }}
h1, h2, h3 {{ letter-spacing: -0.01em; font-weight: 600; color: var(--ca-ink); text-wrap: balance; }}
.ca-brand {{ font-size: 1.05rem; font-weight: 600; color: var(--ca-ink); margin: 0; }}
.ca-brand span {{ color: var(--ca-muted); font-weight: 400; }}
.ca-lead {{ color: var(--ca-muted); max-width: 62ch; margin: .25rem 0 1.4rem; line-height: 1.5; }}

/* langkah: pil dalam satu lintasan, terangkat saat disorot */
.st-key-stepper {{ background: var(--ca-track); border-radius: 999px; padding: 4px; }}
.st-key-stepper [data-testid="stHorizontalBlock"] {{ gap: 4px; }}
div[class*="st-key-step_"] button {{ border: 0 !important; border-radius: 999px; padding: .45rem 1rem;
    background: transparent; color: var(--ca-muted); font-weight: 500;
    transition: background-color .16s ease, color .16s ease, transform .16s ease, box-shadow .16s ease; }}
div[class*="st-key-step_"] button:hover:not(:disabled) {{ background: var(--ca-surface); color: var(--ca-ink);
    transform: translateY(-2px) scale(1.03); box-shadow: var(--ca-lift); }}
div[class*="st-key-step_"] button:active:not(:disabled) {{ transform: translateY(0) scale(.99); }}
div[class*="st-key-step_"] button:focus-visible {{ outline: 2px solid var(--ca-accent); outline-offset: 2px; }}
div[class*="st-key-step_"] button:disabled {{ color: var(--ca-faint); background: transparent; }}
.st-key-step_{step} button, .st-key-step_{step} button:hover:not(:disabled) {{ background: var(--ca-accent) !important;
    color: #fff !important; box-shadow: var(--ca-lift); }}
.st-key-step_{step} button p {{ color: #fff !important; }}

/* halaman: hanya langkah aktif yang tampil, dengan satu transisi masuk */
div[class*="st-key-page_"]:not(.st-key-page_{step}) {{ display: none; }}
.st-key-page_{step} {{ animation: ca-in .24s ease-out both; }}
@keyframes ca-in {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: none; }} }}

.ca-readout {{ font-size: 4.2rem; line-height: 1; font-weight: 500; letter-spacing: -0.03em;
              font-variant-numeric: tabular-nums; color: var(--ca-ink); margin: .2rem 0 .4rem; }}
.ca-readout small {{ font-size: 1.1rem; color: var(--ca-muted); font-weight: 400; letter-spacing: 0; margin-left: .5rem; }}
.ca-pair {{ display: flex; gap: 2rem; font-variant-numeric: tabular-nums; margin-bottom: .9rem; flex-wrap: wrap; }}
.ca-pair div {{ color: var(--ca-muted); font-size: .85rem; }}
.ca-pair b {{ display: block; color: var(--ca-ink); font-size: 1.25rem; font-weight: 500; }}
.ca-conf {{ display: inline-flex; align-items: center; gap: .5rem; font-size: .9rem; color: var(--ca-ink); }}
.ca-dot {{ width: .6rem; height: .6rem; border-radius: 50%; display: inline-block; }}
.ca-meta {{ color: var(--ca-muted); font-size: .85rem; margin: .35rem 0 1rem; }}
.ca-issues {{ margin: .4rem 0 1.2rem; }}
.ca-group {{ font-weight: 600; font-size: .95rem; margin: 1.1rem 0 .1rem; color: var(--ca-ink); }}
.ca-group-sub {{ color: var(--ca-muted); font-size: .82rem; margin-bottom: .5rem; }}
.ca-issue {{ border-left: 3px solid var(--ca-rule); padding: .35rem 0 .45rem .8rem; margin: .45rem 0; }}
.ca-issue.error {{ border-color: var(--ca-err); }}
.ca-issue.warning {{ border-color: var(--ca-warn); }}
.ca-issue.info {{ border-color: var(--ca-faint); }}
.ca-issue p {{ margin: 0; font-size: .88rem; line-height: 1.45; color: var(--ca-muted); max-width: 70ch; }}
.ca-issue p.t {{ color: var(--ca-ink); font-weight: 500; font-size: .92rem; }}
.ca-issue p.a {{ color: var(--ca-ink); margin-top: .2rem; }}
.ca-table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: .92rem; }}
.ca-table th {{ text-align: left; font-weight: 500; color: var(--ca-muted); padding: .45rem .6rem .45rem 0;
               border-bottom: 1px solid var(--ca-rule); }}
.ca-table td {{ padding: .6rem .6rem .6rem 0; border-bottom: 1px solid var(--ca-rule); color: var(--ca-ink);
               overflow-wrap: anywhere; }}
.ca-table td b {{ font-weight: 600; }}
.ca-empty {{ color: var(--ca-muted); padding: 2.2rem 0; }}
.ca-legend {{ color: var(--ca-muted); font-size: .8rem; margin-top: .2rem; }}
.ca-legend i {{ display: inline-block; width: .9rem; height: 2px; vertical-align: middle; margin: 0 .3rem 0 .9rem; }}
.ca-legend i:first-child {{ margin-left: 0; }}
[class*="st-key-sample_btn"] button p {{ color: var(--ca-accent); text-decoration: underline; text-underline-offset: 3px; }}
.ca-rule {{ border: 0; border-top: 1px solid var(--ca-rule); margin: .9rem 0 1.6rem; }}
@media (max-width: 640px) {{ .ca-readout {{ font-size: 3rem; }} }}
@media (prefers-reduced-motion: reduce) {{
  .st-key-page_{step} {{ animation: none; }}
  div[class*="st-key-step_"] button, div[class*="st-key-step_"] button:hover {{ transition: none; transform: none !important; }}
}}
</style>
"""


THEME_JS = """
<script>
// Ganti tema bawaan Streamlit tanpa reload: Streamlit mengikuti "prefers-color-scheme" untuk tema
// Custom Theme Auto dan memeriksanya ulang pada event 'afterprint'.
(function () {
  const w = window.parent, want = "%s";
  try { w.localStorage.removeItem("stActiveTheme-" + w.location.pathname + "-v2"); } catch (e) {}
  if (!w.__caMatchMedia) w.__caMatchMedia = w.matchMedia.bind(w);
  w.__caScheme = want;
  w.matchMedia = function (q) {
    if (typeof q === "string" && q.indexOf("prefers-color-scheme") >= 0) {
      const matches = (q.indexOf("dark") >= 0) === (w.__caScheme === "dark");
      return { matches, media: q, onchange: null, addListener() {}, removeListener() {},
               addEventListener() {}, removeEventListener() {}, dispatchEvent() { return false; } };
    }
    return w.__caMatchMedia(q);
  };
  w.document.documentElement.style.colorScheme = want;
  w.dispatchEvent(new Event("afterprint"));
})();
</script>
"""


# ---------------------------------------------------------------- state, cache & analysis
def _state():
    ss = st.session_state
    ss.setdefault("images", {})  # id -> dict
    ss.setdefault("order", [])
    ss.setdefault("selected", None)
    ss.setdefault("last_click", {})
    ss.setdefault("step", 1)
    ss.setdefault("uploader_key", 0)
    ss.setdefault("sel", set())
    ss.setdefault("sel_ver", 0)
    if "theme" not in ss:  # pilihan tersimpan di URL (?tema=) agar bertahan saat halaman dimuat ulang
        q = st.query_params.get("tema")
        ss.theme = q if q in PAL else ("dark" if getattr(st.context.theme, "type", None) == "dark" else "light")
    return ss


@st.cache_resource
def _store() -> dict:
    """Cache bersama semua sesi: gambar terdekode, hasil analisis (kunci = hash file + koreksi), thumbnail.

    Ukuran dibatasi (LRU) agar memori server tidak terus bertambah bila app berjalan berhari-hari.
    Foto terdekode ±30 MB, hasil (dengan overlay) ±10 MB per entri.
    """
    return {"img": LRUCache(24), "res": LRUCache(200), "thumb": LRUCache(1000)}


def image_of(item):
    store = _store()["img"]
    img = store.get(item["hash"])
    if img is None:
        img = store[item["hash"]] = cv2.imdecode(np.frombuffer(item["bytes"], np.uint8), cv2.IMREAD_COLOR)
    return img


def _params(item):
    s = item["settings"]
    roi = tuple(round(v, 2) for v in s["roi"]) if s["roi"] else None
    bl = tuple(tuple(round(v, 2) for v in p) for p in s["baseline"]) if len(s["baseline"]) == 2 else None
    pts = tuple(tuple(round(v, 2) for v in p) for p in s["points"]) if len(s["points"]) >= 3 else None
    return roi, bl, pts


def _rkey(item):
    return (item["hash"],) + _params(item)


def _compute(item):
    roi, bl, pts = _params(item)
    return analyze(image_of(item), roi=roi, baseline=bl, manual_points=pts)


def result_of(item):
    res = _store()["res"]
    k = _rkey(item)
    r = res.get(k)
    if r is None:
        r = res[k] = _compute(item)
    return r


@st.cache_resource
def _pool():
    """Proses pekerja persisten: analisis CPU-bound berjalan benar-benar paralel."""
    try:
        return ProcessPoolExecutor(WORKERS)
    except Exception:  # lingkungan tanpa multiprocessing
        return ThreadPoolExecutor(WORKERS)


def ensure_results(keys, label="Menganalisis"):
    """Analisis paralel untuk semua foto yang belum ada di cache, dengan satu progress bar."""
    res = _store()["res"]
    todo = [ss.images[k] for k in keys if _rkey(ss.images[k]) not in res]
    if not todo:
        return
    bar = st.progress(0.0, text=f"{label} {len(todo)} foto…")
    ex = _pool()
    futs = {ex.submit(analyze_bytes, it["bytes"], *_params(it)): it for it in todo}
    for n, f in enumerate(as_completed(futs), 1):
        it = futs[f]
        try:
            res[_rkey(it)] = f.result()
        except Exception:  # pekerja gagal (mis. proses mati): hitung di proses utama
            res[_rkey(it)] = _compute(it)
        bar.progress(n / len(todo), text=f"{label} foto {n} dari {len(todo)}…")
    bar.empty()


def fmt(v, nd=1):
    return "–" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def included(item, res) -> bool:
    if item["include"] is not None:
        return bool(item["include"] and res.ok)
    return res.ok and not any(i.level == "error" for i in res.issues)


def add_files(files, label):
    added = []
    for f in files:
        data = f.getvalue()
        key = hashlib.md5(data).hexdigest()[:12]
        if key in ss.images:
            continue
        n_same = sum(1 for i in ss.images.values() if i["label"] == label)
        ss.images[key] = dict(
            name=f.name, bytes=data, hash=key, label=label, replicate=n_same + 1, include=None,
            settings=dict(mode="auto", roi=None, baseline=[], points=[]),
        )
        ss.order.append(key)
        added.append(key)
        if ss.selected is None:
            ss.selected = key
    return added


def remove_files(keys):
    for k in keys:
        ss.images.pop(k, None)
        if k in ss.order:
            ss.order.remove(k)
    if ss.selected not in ss.order:
        ss.selected = ss.order[0] if ss.order else None


def issue_html(issues, title, sub):
    if not issues:
        return ""
    rows = "".join(
        f'<div class="ca-issue {i.level}"><p class="t">{html.escape(i.title)}</p>'
        f'<p>{html.escape(i.detail)}</p><p class="a">{html.escape(i.advice)}</p></div>'
        for i in issues)
    return (f'<div class="ca-issues"><div class="ca-group">{title}</div>'
            f'<div class="ca-group-sub">{sub}</div>{rows}</div>')


def go(step):
    st.session_state.step = step


def _set_theme():
    ss_ = st.session_state
    if ss_.theme_pick:
        ss_.theme = ss_.theme_pick
        st.query_params["tema"] = ss_.theme
    else:  # klik ulang pada pilihan aktif: tetap pada tema sekarang
        ss_.theme_pick = ss_.theme


# ---------------------------------------------------------------- helpers untuk banyak foto
def tone(res) -> str:
    if not res.ok or any(i.level == "error" for i in res.issues):
        return "err"
    return "good" if res.confidence >= 75 else "warn" if res.confidence >= 50 else "err"


def thumb_url(item, res) -> str:
    """Thumbnail kecil (crop di sekitar tetesan) dari overlay, disimpan di cache."""
    store = _store()["thumb"]
    k = _rkey(item)
    if store.get(k) is None:
        ov = res.overlay if res.overlay is not None else image_of(item)
        t = ov
        if res.roi_suggestion is not None:
            s = res.scale
            x0, y0, x1, y1 = [int(v * s) for v in res.roi_suggestion]
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])
            pad = int(0.5 * max(x1 - x0, 30))
            h, w = ov.shape[:2]
            t = ov[max(0, y0 - pad):min(h, y1 + pad // 2), max(0, x0 - pad):min(w, x1 + pad)]
            if t.size == 0:
                t = ov
        t = cv2.resize(t, (224, 128), interpolation=cv2.INTER_AREA)
        store[k] = jpeg_data_url(t, 72)
    return store[k]


def renumber():
    """Nomor ulangan berurutan per sampel, mengikuti urutan foto."""
    seen = {}
    for k in ss.order:
        it = ss.images[k]
        seen[it["label"]] = seen.get(it["label"], 0) + 1
        it["replicate"] = seen[it["label"]]


def photo_name(k):
    it = ss.images[k]
    return f"{it['label']}, ulangan {it['replicate']} ({it['name']})"


def clear_all():
    for k in ("images", "order", "selected", "last_click", "sel", "groups", "pick"):
        ss.pop(k, None)
    ss.step = 1
    ss.uploader_key += 1
    _state()


# ---------------------------------------------------------------- Upload
def _set_sel(keys):
    ss.sel = set(keys)
    ss.sel_ver = ss.get("sel_ver", 0) + 1


def page_upload():
    st.markdown("## Upload foto")
    st.markdown('<p class="ca-lead">Foto dengan nama sampel yang sama dihitung sebagai ulangan. '
                "Nama dan nomor ulangan tetap bisa diubah di daftar di bawah.</p>", unsafe_allow_html=True)
    label = st.text_input("Nama sampel", value=ss.get("last_label", "Sampel 1"),
                          placeholder="Mis. Entri 0.3…", max_chars=60)
    ss.last_label = label
    files = st.file_uploader("Foto tetesan", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                             accept_multiple_files=True, key=f"uploader_{ss.uploader_key}",
                             help="JPG, PNG, BMP, atau TIFF. Gunakan file asli dari kamera, bukan hasil kiriman chat.")
    if files:
        added = add_files(files, label.strip() or "Sampel")
        ensure_results(added)
        ss.uploader_key += 1  # kosongkan uploader; daftar di bawah menjadi sumber data
        st.rerun()

    sample_files = sorted((Path(__file__).parent / "samples").glob("drop_*.*"))
    if sample_files and not ss.order:
        if st.button("Pakai foto contoh", type="tertiary", key="sample_btn"):
            class _F:
                def __init__(self, p):
                    self.name, self._b = p.name, p.read_bytes()

                def getvalue(self):
                    return self._b
            ensure_results(add_files([_F(p) for p in sample_files], "Contoh"))
            st.rerun()
    if not ss.order:
        return

    ensure_results(ss.order)
    sel = ss.sel & set(ss.order)
    st.markdown(f"### {len(ss.order)} foto")
    labels = list(dict.fromkeys(ss.images[k]["label"] for k in ss.order))
    t1, t2, t3, t4 = st.columns([1, 1.1, 2.2, 1.1], vertical_alignment="bottom")
    t1.button("Pilih semua", on_click=_set_sel, args=(ss.order,), width="stretch")
    t2.button("Kosongkan pilihan", on_click=_set_sel, args=([],), width="stretch", disabled=not sel)
    by = t3.multiselect("Pilih menurut sampel", labels, placeholder="Pilih sampel…", key=f"by_{ss.sel_ver}")
    if by:
        _set_sel([k for k in ss.order if ss.images[k]["label"] in by] + list(sel))
        st.rerun()
    with t4.popover("Hapus semua", width="stretch"):
        st.markdown("Hapus semua foto dan koreksi, lalu mulai dari awal? Tindakan ini tidak bisa dibatalkan.")
        if st.button("Ya, mulai baru", type="primary", key="confirm_clear"):
            clear_all()
            st.rerun()

    rows = []
    for k in ss.order:
        it, r = ss.images[k], result_of(ss.images[k])
        rows.append({"Pilih": k in sel, "File": it["name"], "Sampel": it["label"], "Ulangan": int(it["replicate"]),
                     "θ (°)": r.theta if r.ok else None, "Keyakinan": r.confidence})
    edited = st.data_editor(
        pd.DataFrame(rows, index=ss.order), hide_index=True, width="stretch", height=min(38 + 35 * len(rows), 460),
        key=f"editor_{len(ss.order)}_{ss.uploader_key}_{ss.sel_ver}",
        column_config={
            "Pilih": st.column_config.CheckboxColumn(width="small"),
            "File": st.column_config.TextColumn(disabled=True),
            "Sampel": st.column_config.TextColumn(required=True, max_chars=60),
            "Ulangan": st.column_config.NumberColumn(min_value=1, max_value=999, step=1, width="small"),
            "θ (°)": st.column_config.NumberColumn(disabled=True, format="%.1f", width="small"),
            "Keyakinan": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%", width="small"),
        })
    for k, row in edited.iterrows():
        ss.images[k]["label"] = str(row["Sampel"]).strip() or ss.images[k]["label"]
        if pd.notna(row["Ulangan"]):
            ss.images[k]["replicate"] = int(row["Ulangan"])
    sel = {k for k, row in edited.iterrows() if row["Pilih"]}
    ss.sel = sel

    c1, c2, c3, _ = st.columns([1.2, 1.4, 1.1, 1.5])
    c1.button("Lanjut ke Atur", type="primary", on_click=go, args=(2,), width="stretch")
    with c2.popover(f"Ganti nama sampel ({len(sel)})", width="stretch", disabled=not sel):
        new = st.text_input("Nama sampel baru", key="rename_to", placeholder="Mis. After ECL…", max_chars=60)
        if st.button("Terapkan", type="primary", key="rename_go", disabled=not new.strip()):
            for k in sel:
                ss.images[k]["label"] = new.strip()
            renumber()
            _set_sel([])
            st.rerun()
    with c3.popover(f"Hapus ({len(sel)})", width="stretch", disabled=not sel):
        st.markdown(f"Hapus {len(sel)} foto terpilih beserta koreksinya? Tindakan ini tidak bisa dibatalkan.")
        if st.button("Ya, hapus", type="primary", key="confirm_del"):
            remove_files(sel)
            _set_sel([])
            st.rerun()


# ---------------------------------------------------------------- Atur
def _focus_box(res, W, H):
    if res.roi_suggestion is None:
        return None
    x0, y0, x1, y1 = res.roi_suggestion
    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])
    pad = 0.35 * max(x1 - x0, 0.04 * W)
    return [max(0, (x0 - pad) / W), max(0, (y0 - pad) / H), min(1, (x1 + pad) / W), min(1, (y1 + pad * 0.6) / H)]


FILTERS = {"all": "Semua", "check": "Perlu dicek", "fail": "Gagal"}


@st.fragment
def panel_atur():
    results = {k: result_of(ss.images[k]) for k in ss.order}
    tones = {k: tone(results[k]) for k in ss.order}
    counts = {"all": len(ss.order), "check": sum(t == "warn" for t in tones.values()),
              "fail": sum(t == "err" for t in tones.values())}
    flt = st.segmented_control("Tampilkan", list(FILTERS), default="all", key="flt",
                               format_func=lambda f: f"{FILTERS[f]} ({counts[f]})",
                               label_visibility="collapsed") or "all"
    keys = [k for k in ss.order if flt == "all" or (flt == "check" and tones[k] == "warn")
            or (flt == "fail" and tones[k] == "err")]
    if not keys:
        st.markdown('<p class="ca-empty">Tidak ada foto pada filter ini.</p>', unsafe_allow_html=True)
        return

    # pilihan dari thumbnail (nilai komponen sudah ada sebelum digambar)
    sv = ss.get("strip")
    if sv and sv.get("nonce") != ss.get("strip_nonce"):
        ss.strip_nonce = sv["nonce"]
        if sv["id"] in keys:
            ss["pick"] = sv["id"]
    if ss.get("pick") not in keys:
        ss["pick"] = ss.selected if ss.selected in keys else keys[0]

    def _move(d):
        ss["pick"] = keys[(keys.index(ss["pick"]) + d) % len(keys)]

    idx = keys.index(ss["pick"])
    n1, n2, n3 = st.columns([0.7, 4, 0.7], vertical_alignment="bottom")
    n1.button("Sebelumnya", on_click=_move, args=(-1,), width="stretch", disabled=len(keys) < 2)
    n2.selectbox(f"Foto {idx + 1} dari {len(keys)}", keys, format_func=photo_name, key="pick")
    n3.button("Berikutnya", on_click=_move, args=(1,), width="stretch", disabled=len(keys) < 2)
    strip([{"id": k, "src": thumb_url(ss.images[k], results[k]), "tone": tones[k],
            "label": (f"{results[k].theta:.1f}°" if results[k].ok else "gagal"),
            "sub": f"{ss.images[k]['label']} #{ss.images[k]['replicate']}", "title": photo_name(k)} for k in keys],
          ss["pick"], theme={k: P[k] for k in ("ink", "muted", "rule", "accent", "surface", "good", "warn", "err")},
          key="strip")
    ss.selected = key = ss["pick"]

    item = ss.images[key]
    sset = item["settings"]
    img = image_of(item)
    if img is None:
        st.error("File ini tidak bisa dibaca sebagai gambar. Hapus di langkah Upload lalu upload ulang.")
        return
    res = results[key]
    H, W = img.shape[:2]

    # nilai klik terbaru diterapkan sebelum menggambar, sehingga cukup satu kali render
    val = ss.get(f"clk_{key}")
    if val and val.get("nonce") != ss.last_click.get(key):
        ss.last_click[key] = val["nonce"]
        kind = val["kind"]
        if kind == "baseline":
            sset["baseline"] = [[p[0] * W, p[1] * H] for p in val["pts"]]
        elif kind == "roi":
            r = val["roi"]
            sset.update(roi=[r[0] * W, r[1] * H, r[2] * W, r[3] * H], points=[])
        elif kind == "locate":
            x, y = val["pt"][0] * W, val["pt"][1] * H
            half = float(np.clip(0.3 * val.get("view", 0.3) * W, 0.04 * max(W, H), 0.2 * max(W, H)))
            sset.update(roi=[x - half, y - half, x + half, y + 0.8 * half], baseline=[], points=[])
        elif kind == "points":
            sset["points"] = [[p[0] * W, p[1] * H] for p in val["pts"]]
        res = result_of(item)

    st.write("")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        mode = st.segmented_control("Koreksi", list(MODES), format_func=lambda m: MODES[m],
                                    default=sset["mode"] if sset["mode"] in MODES else "auto",
                                    key=f"mode_{key}") or "auto"
        sset["mode"] = mode
        st.caption(MODE_HELP[mode])
        ov = res.overlay if res.overlay is not None else img
        if max(ov.shape[:2]) > 2000:
            f = 2000 / max(ov.shape[:2])
            ov = cv2.resize(ov, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        roi_n = [sset["roi"][0] / W, sset["roi"][1] / H, sset["roi"][2] / W, sset["roi"][3] / H] if sset["roi"] else None
        clicker(ov, mode, vid=item["hash"], focus=_focus_box(res, W, H),
                pts=[[p[0] / W, p[1] / H] for p in sset["points"]],
                base=[[p[0] / W, p[1] / H] for p in sset["baseline"]], roi=roi_n,
                theme={k: P[k] for k in ("ink", "muted", "rule", "accent", "surface")}, key=f"clk_{key}")
        st.markdown(
            '<div class="ca-legend"><i style="background:#FFBE28"></i>baseline'
            '<i style="background:#1EC8E6"></i>kontur<i style="background:#E63C3C"></i>garis tangen</div>',
            unsafe_allow_html=True)
        notes = []
        if sset["roi"]:
            notes.append("lokasi tetesan ditentukan")
        if len(sset["baseline"]) == 2:
            notes.append("baseline manual")
        if len(sset["points"]) >= 3:
            notes.append(f"{len(sset['points'])} titik tepi")
        c1, c2 = st.columns([1, 2], vertical_alignment="center")
        if c1.button("Kembali ke otomatis", width="stretch", disabled=not (notes or mode != "auto")):
            item["settings"] = dict(mode="auto", roi=None, baseline=[], points=[])
            ss.pop(f"mode_{key}", None)
            st.rerun(scope="fragment")
        if notes:
            c2.caption("Koreksi aktif: " + ", ".join(notes) + ".")

    with right:
        if not res.ok:
            st.markdown(f'<p class="ca-group">Sudut kontak tidak dapat dihitung</p>'
                        f'<p class="ca-meta">{html.escape(res.message or "")}</p>', unsafe_allow_html=True)
        else:
            c = res.confidence
            dot, word = ((P["good"], "andal") if c >= 75 else (P["warn"], "perlu dicek") if c >= 50
                         else (P["err"], "tidak andal"))
            wet = ("Spreading kuat" if res.theta < 10 else "Hidrofilik" if res.theta < 90
                   else "Superhidrofobik" if res.theta >= 150 else "Hidrofobik")
            st.markdown(
                f'<div class="ca-readout">{fmt(res.theta)}°<small>±{fmt(res.sensitivity)}° per px</small></div>'
                f'<div class="ca-pair"><div>Kiri<b>{fmt(res.theta_left)}°</b></div>'
                f'<div>Kanan<b>{fmt(res.theta_right)}°</b></div><div>Sifat<b>{wet}</b></div></div>'
                f'<span class="ca-conf"><span class="ca-dot" style="background:{dot}"></span>'
                f'Keyakinan {c}%, {word}</span>'
                f'<p class="ca-meta">Metode {res.primary_method.lower()}. Lebar dasar {fmt(res.base_width_px, 0)} px, '
                f'tinggi {fmt(res.height_px, 0)} px.</p>', unsafe_allow_html=True)
            with st.expander("Perbandingan metode"):
                st.dataframe(pd.DataFrame([{"Metode": m.name, "Kiri": m.theta_left, "Kanan": m.theta_right,
                                            "Rata-rata": m.theta_mean} for m in res.methods]),
                             hide_index=True, width="stretch",
                             column_config={c_: st.column_config.NumberColumn(format="%.1f°")
                                            for c_ in ("Kiri", "Kanan", "Rata-rata")})
                st.caption("Elips sama dengan plugin ImageJ. Lingkaran dan tinggi/lebar menganggap tetesan "
                           "berbentuk tutup bola. Tangen lokal hanya memakai kontur di dekat titik kontak.")
        foto = [i for i in res.issues if i.source == "foto"]
        anal = [i for i in res.issues if i.source == "analisis"]
        st.markdown(
            issue_html(foto, "Dari pengambilan foto", "Tidak bisa diperbaiki software. Perbaiki saat memotret ulang.")
            + issue_html(anal, "Dari analisis", "Periksa overlay, lalu gunakan koreksi bila kontur atau baseline meleset."),
            unsafe_allow_html=True)
        if not foto and not anal and res.ok:
            st.markdown('<p class="ca-meta">Tidak ada masalah terdeteksi.</p>', unsafe_allow_html=True)
        item["include"] = st.checkbox("Sertakan di hasil", value=included(item, res), key=f"inc_{key}",
                                      disabled=not res.ok)


def page_atur():
    if not ss.order:
        st.markdown('<p class="ca-empty">Belum ada foto. Upload foto terlebih dahulu.</p>', unsafe_allow_html=True)
        st.button("Ke Upload", on_click=go, args=(1,))
        return
    ensure_results(ss.order)
    panel_atur()
    st.write("")
    f1, f2, _ = st.columns([1, 1.2, 3])
    f1.button("Kembali", on_click=go, args=(1,), width="stretch")
    f2.button("Lanjut ke Hasil", type="primary", on_click=go, args=(3,), width="stretch")


# ---------------------------------------------------------------- Hasil
def build_tables():
    rows = []
    for k in ss.order:
        it = ss.images[k]
        r = result_of(it)
        rows.append({
            "File": it["name"], "Sampel": it["label"], "Ulangan": it["replicate"],
            "θ kiri (°)": r.theta_left, "θ kanan (°)": r.theta_right, "θ rata-rata (°)": r.theta,
            "±° per px": r.sensitivity, "Metode": r.primary_method if r.ok else "",
            "Keyakinan (%)": r.confidence, "Disertakan": included(it, r),
            "Koreksi": MODES.get(it["settings"]["mode"], ""),
            "Masalah foto": "; ".join(i.title for i in r.issues if i.source == "foto"),
            "Ketidakpastian analisis": "; ".join(i.title for i in r.issues if i.source == "analisis"),
        })
    detail = pd.DataFrame(rows, index=ss.order)
    used = detail[detail["Disertakan"]]
    if len(used):
        summ = (used.groupby("Sampel", sort=False)["θ rata-rata (°)"]
                .agg(**{"n": "count", "θ rata-rata (°)": "mean", "SD (°)": "std", "Min (°)": "min",
                        "Maks (°)": "max"})
                .reset_index())
        summ["SD (°)"] = summ["SD (°)"].fillna(0.0)
    else:
        summ = pd.DataFrame(columns=["Sampel", "n", "θ rata-rata (°)", "SD (°)", "Min (°)", "Maks (°)"])
    return detail, summ


def _axes_style(q, bg):
    return {"font.size": 9, "axes.edgecolor": q["muted"], "axes.labelcolor": q["ink"], "xtick.color": q["ink"],
            "ytick.color": q["muted"], "text.color": q["ink"], "axes.facecolor": bg, "figure.facecolor": bg}


def _ylim(ax, tops, q):
    top = float(np.nanmax(tops)) if len(tops) else 90
    ax.set_ylim(0, max(100, math.ceil((top + 15) / 10) * 10))
    ax.axhline(90, color=q["muted"], ls=(0, (3, 3)), lw=0.8)
    ax.annotate("90°", (1, 90), xycoords=("axes fraction", "data"), xytext=(4, 0),
                textcoords="offset points", va="center", color=q["muted"])


def _png(fig, bg):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner=False, max_entries=32)
def make_chart(summ: pd.DataFrame, p: dict, for_export: bool = False):
    # grafik untuk file selalu berlatar putih; di layar mengikuti tema
    q = PAL["light"] if for_export else p
    bg = "white" if for_export else q["bg"]
    with plt.rc_context(_axes_style(q, bg)):
        fig, ax = plt.subplots(figsize=(max(4.8, 1.25 * len(summ) + 2.2), 3.8), dpi=150)
        x = np.arange(len(summ))
        ax.bar(x, summ["θ rata-rata (°)"], width=0.56, yerr=summ["SD (°)"], capsize=4, color=q["accent"],
               error_kw=dict(ecolor=q["ink"], lw=0.9))
        for xi, (m, sd) in enumerate(zip(summ["θ rata-rata (°)"], summ["SD (°)"])):
            ax.text(xi, m + sd + 2, f"{m:.1f}", ha="center", va="bottom")
        ax.set_xticks(x, summ["Sampel"])
        ax.set_ylabel("Sudut kontak rata-rata (°)")
        _ylim(ax, summ["θ rata-rata (°)"] + summ["SD (°)"], q)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0, axis="x")
        fig.tight_layout()
        return _png(fig, bg)


GROUP_COLORS = ["#0F6E7C", "#C08A2E", "#6B5CA5", "#3F8F5A", "#A33A2C", "#4A6FA5"]


@st.cache_data(show_spinner=False, max_entries=32)
def comparison_chart(names: tuple, values: tuple, p: dict, for_export: bool = False):
    q = PAL["light"] if for_export else p
    bg = "white" if for_export else q["bg"]
    with plt.rc_context(_axes_style(q, bg)):
        fig, ax = plt.subplots(figsize=(max(4.8, 1.4 * len(names) + 2.2), 3.8), dpi=150)
        tops = []
        rng = np.random.default_rng(0)
        for i, v in enumerate(values):
            v = np.asarray([x for x in v if np.isfinite(x)], float)
            if not len(v):
                continue
            col = GROUP_COLORS[i % len(GROUP_COLORS)]
            m, sd = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
            ax.bar(i, m, width=0.56, color=col, alpha=.28, edgecolor=col, lw=1.2)
            ax.errorbar(i, m, yerr=sd, color=q["ink"], capsize=4, lw=0.9)
            ax.scatter(i + rng.uniform(-0.14, 0.14, len(v)), v, s=16, color=col, zorder=3, edgecolor="white", lw=.5)
            ax.text(i, max(m + sd, v.max()) + 2, f"{m:.1f}", ha="center", va="bottom")
            tops.append(max(m + sd, v.max()))
        ax.set_xticks(range(len(names)), names)
        ax.set_ylabel("Sudut kontak (°)")
        _ylim(ax, tops, q)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0, axis="x")
        fig.tight_layout()
        return _png(fig, bg)


@st.cache_data(show_spinner=False, max_entries=16)
def build_excel(summ: pd.DataFrame, detail: pd.DataFrame, comp: pd.DataFrame | None, png: bytes | None) -> bytes:
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="openpyxl") as xw:
        summ.round(3).to_excel(xw, sheet_name="Ringkasan", index=False)
        detail.round(3).to_excel(xw, sheet_name="Detail", index=False)
        if comp is not None and len(comp):
            comp.round(4).to_excel(xw, sheet_name="Perbandingan", index=False)
        if png:
            from openpyxl.drawing.image import Image as XLImage

            xi = XLImage(io.BytesIO(png))
            xi.width, xi.height = xi.width * 0.45, xi.height * 0.45
            xw.sheets["Ringkasan"].add_image(xi, f"A{len(summ) + 4}")
    return xbuf.getvalue()


def _init_groups(detail):
    if ss.get("groups"):
        return
    labels = list(dict.fromkeys(detail.loc[detail["Disertakan"], "Sampel"]))
    ss.groups = [{"id": i, "name": lab,
                  "keys": [k for k in detail.index if detail.at[k, "Disertakan"] and detail.at[k, "Sampel"] == lab]}
                 for i, lab in enumerate(labels[:2])]
    while len(ss.groups) < 2:
        ss.groups.append({"id": len(ss.groups), "name": f"Kelompok {chr(65 + len(ss.groups))}", "keys": []})
    ss.group_next = len(ss.groups)


def _fill_group(gid):
    lab = ss.get(f"gfill_{gid}")
    g = next(g for g in ss.groups if g["id"] == gid)
    if lab:
        g["keys"] = [k for k in ss.order
                     if ss.images[k]["label"] == lab and included(ss.images[k], result_of(ss.images[k]))]
        g["name"] = lab
        ss[f"gname_{gid}"] = lab
        ss[f"gkeys_{gid}"] = g["keys"]
        ss[f"gfill_{gid}"] = None


def _add_group():
    gid = ss.group_next
    ss.group_next += 1
    ss.groups.append({"id": gid, "name": f"Kelompok {chr(65 + len(ss.groups))}", "keys": []})


def _del_group(gid):
    ss.groups = [x for x in ss.groups if x["id"] != gid]


def section_compare(detail):
    _init_groups(detail)
    st.markdown("### Perbandingan")
    st.markdown('<p class="ca-lead">Pilih foto untuk tiap kelompok, mis. 6 foto sebelum perlakuan dibandingkan '
                "dengan 6 foto sesudahnya. Kelompok pertama menjadi acuan.</p>", unsafe_allow_html=True)
    usable = [k for k in detail.index if detail.at[k, "Disertakan"]]
    labels = list(dict.fromkeys(detail.loc[usable, "Sampel"]))
    cols = st.columns(min(3, len(ss.groups)), gap="medium")
    for i, g in enumerate(list(ss.groups)):
        with cols[i % len(cols)]:
            gid = g["id"]
            g["name"] = st.text_input("Nama kelompok", g["name"], key=f"gname_{gid}", max_chars=40) or g["name"]
            st.selectbox("Isi dari sampel", labels, index=None, placeholder="Pilih sampel…", key=f"gfill_{gid}",
                         on_change=_fill_group, args=(gid,))
            g["keys"] = st.multiselect("Foto", usable, default=[k for k in g["keys"] if k in usable],
                                       format_func=photo_name, key=f"gkeys_{gid}", placeholder="Pilih foto…")
            if len(ss.groups) > 2:
                st.button("Hapus kelompok", key=f"gdel_{gid}", type="tertiary", on_click=_del_group, args=(gid,))
    if len(ss.groups) < 6:
        st.button("Tambah kelompok", type="tertiary", on_click=_add_group)

    groups = [{"name": g["name"], "values": [float(detail.at[k, "θ rata-rata (°)"]) for k in g["keys"]]}
              for g in ss.groups if g["keys"]]
    if len(groups) < 2:
        st.markdown('<p class="ca-meta">Isi minimal dua kelompok untuk melihat perbandingan.</p>',
                    unsafe_allow_html=True)
        return None, None, []
    comp = compare_groups(groups)
    texts = describe_comparison(comp)
    names, vals = tuple(g["name"] for g in groups), tuple(tuple(g["values"]) for g in groups)

    def cell(v, f):
        return "–" if not np.isfinite(v) else format(v, f)

    c1, c2 = st.columns([1, 1.15], gap="large")
    with c1:
        body = "".join(
            f"<tr><td>{html.escape(str(r['Kelompok']))}</td><td>{int(r['n'])}</td>"
            f"<td><b>{cell(r['Rata-rata (°)'], '.2f')}</b></td><td>{cell(r['SD (°)'], '.2f')}</td>"
            f"<td>{cell(r['Δ (°)'], '+.2f')}</td><td>{cell(r['p'], '.3g')}</td></tr>" for _, r in comp.iterrows())
        st.markdown('<table class="ca-table"><thead><tr><th>Kelompok</th><th>n</th><th>Rata-rata</th><th>SD</th>'
                    f'<th>Δ</th><th>p</th></tr></thead><tbody>{body}</tbody></table>'
                    + "".join(f'<p class="ca-meta">{html.escape(t)}</p>' for t in texts)
                    + '<p class="ca-meta">p dari uji t Welch (tidak mengasumsikan varians sama).</p>',
                    unsafe_allow_html=True)
    with c2:
        st.image(comparison_chart(names, vals, P), width="stretch")
    return comp, comparison_chart(names, vals, P, for_export=True), texts


def page_hasil():
    if not ss.order:
        st.markdown('<p class="ca-empty">Belum ada foto. Upload foto terlebih dahulu.</p>', unsafe_allow_html=True)
        st.button("Ke Upload", on_click=go, args=(1,))
        return
    ensure_results(ss.order)
    detail, summ = build_tables()
    st.markdown("## Hasil")
    n_used = int(detail["Disertakan"].sum())
    lead = f"{n_used} dari {len(detail)} foto dipakai."
    if n_used < len(detail):
        lead += " Foto lain gagal dianalisis, punya masalah berat, atau dikecualikan di langkah Atur."
    st.markdown(f'<p class="ca-lead">{lead}</p>', unsafe_allow_html=True)

    png = None
    if len(summ):
        g1, g2 = st.columns([1, 1.15], gap="large")
        with g1:
            body = "".join(
                f"<tr><td>{html.escape(str(r['Sampel']))}</td><td>{int(r['n'])}</td>"
                f"<td><b>{r['θ rata-rata (°)']:.2f}</b></td><td>{r['SD (°)']:.2f}</td>"
                f"<td>{r['Min (°)']:.1f}–{r['Maks (°)']:.1f}</td></tr>" for _, r in summ.iterrows())
            st.markdown('<table class="ca-table"><thead><tr><th>Sampel</th><th>n</th><th>θ rata-rata</th>'
                        f'<th>SD</th><th>Rentang</th></tr></thead><tbody>{body}</tbody></table>'
                        '<p class="ca-meta">Semua nilai dalam derajat.</p>', unsafe_allow_html=True)
        with g2:
            st.image(make_chart(summ, P), width="stretch")
        png = make_chart(summ, P, for_export=True)
    else:
        st.markdown('<p class="ca-empty">Belum ada foto yang disertakan. Centang “Sertakan di hasil” di langkah '
                    "Atur.</p>", unsafe_allow_html=True)

    with st.expander(f"Detail per foto ({len(detail)})"):
        st.dataframe(detail, hide_index=True, width="stretch",
                     column_config={c_: st.column_config.NumberColumn(format="%.1f")
                                    for c_ in ("θ kiri (°)", "θ kanan (°)", "θ rata-rata (°)", "±° per px")})

    st.markdown('<hr class="ca-rule">', unsafe_allow_html=True)
    comp, comp_png, comp_text = section_compare(detail)

    # ---- bagikan & unduh
    st.markdown('<hr class="ca-rule">', unsafe_allow_html=True)
    st.markdown("### Bagikan dan unduh")
    title = st.text_input("Judul laporan", value="Hasil contact angle", key="report_title", max_chars=80)
    rows = [{"label": ss.images[k]["label"], "replicate": ss.images[k]["replicate"], "name": ss.images[k]["name"],
             "res": result_of(ss.images[k]), "settings": ss.images[k]["settings"], "hash": ss.images[k]["hash"],
             "bytes": ss.images[k]["bytes"]} for k in ss.order]
    used_rows = [r for r, k in zip(rows, ss.order) if detail.at[k, "Disertakan"]]
    stamp = pd.Timestamp.now().strftime("%Y%m%d")
    summ_r = summ.round(3)
    d1, d2, d3 = st.columns(3)
    d1.download_button("Laporan (HTML)",
                       lambda: html_report(title, summ_r, png, comp, comp_png, comp_text, used_rows),
                       f"laporan_contact_angle_{stamp}.html", "text/html", type="primary", width="stretch",
                       on_click="ignore", help="Satu file berisi tabel, grafik, dan foto beranotasi. "
                                               "Bisa dikirim lewat email atau chat dan dibuka di browser.")
    d2.download_button("Excel", build_excel(summ, detail, comp, png), f"contact_angle_{stamp}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch",
                       on_click="ignore")
    d3.download_button(f"Foto beranotasi ({len(rows)}, ZIP)", lambda: annotated_zip(rows),
                       f"foto_beranotasi_{stamp}.zip", "application/zip", width="stretch", on_click="ignore")
    e1, e2, e3 = st.columns(3)
    e1.download_button("CSV detail", detail.to_csv(index=False).encode("utf-8-sig"), f"contact_angle_{stamp}.csv",
                       "text/csv", width="stretch", on_click="ignore")
    if png:
        e2.download_button("Grafik (PNG)", png, f"grafik_contact_angle_{stamp}.png", "image/png", width="stretch",
                           on_click="ignore")
    e3.download_button("Data latih (ZIP)", lambda: dataset_zip(rows), f"data_latih_{stamp}.zip", "application/zip",
                       width="stretch", on_click="ignore",
                       help="Foto asli, koreksi Anda, dan hasil analisis, untuk evaluasi dan penyempurnaan algoritma.")
    if len(summ):
        with st.expander("Ringkasan teks untuk chat"):
            st.code(text_summary(summ, comp_text), language=None, wrap_lines=True)
    st.write("")
    st.button("Kembali ke Atur", on_click=go, args=(2,))


# ---------------------------------------------------------------- app
def main():
    global ss, P
    st.set_page_config(page_title="Contact Angle", layout="wide", initial_sidebar_state="collapsed")
    ss = _state()
    P = PAL[ss.theme]
    st.markdown(css(P, ss.step), unsafe_allow_html=True)
    with st.container(key="theme_js"):
        st.iframe(THEME_JS % ss.theme, height=1)

    h1_, h2_, h3_ = st.columns([1.1, 1.5, 0.45], vertical_alignment="center")
    h1_.markdown('<p class="ca-brand">Contact Angle <span>sessile drop</span></p>', unsafe_allow_html=True)
    with h2_:
        with st.container(key="stepper"):
            cols = st.columns(3)
            for n, name in STEPS.items():
                cols[n - 1].button(name, key=f"step_{n}", on_click=go, args=(n,),
                                   disabled=(n > 1 and not ss.order), width="stretch")
    with h3_:
        st.segmented_control("Tema", ["light", "dark"], default=ss.theme, key="theme_pick", on_change=_set_theme,
                             format_func=lambda t: ":material/light_mode:" if t == "light" else ":material/dark_mode:",
                             label_visibility="collapsed", width="stretch",
                             help="Mode terang atau gelap")
    st.markdown('<hr class="ca-rule">', unsafe_allow_html=True)


    with st.container(key=f"page_{ss.step}"):
        {1: page_upload, 2: page_atur, 3: page_hasil}[ss.step]()


ss = None
P = PAL["light"]
if __name__ != "__mp_main__":  # proses pekerja multiprocessing mengimpor file ini; jangan jalankan UI di sana
    main()
