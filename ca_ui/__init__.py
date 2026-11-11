"""Komponen antarmuka khusus untuk app Streamlit."""
from __future__ import annotations

import base64
import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import streamlit.components.v1 as components

_clicker = components.declare_component("ca_clicker", path=str(Path(__file__).parent / "clicker"))
_strip = components.declare_component("ca_strip", path=str(Path(__file__).parent / "strip"))


class LRUCache:
    """Dict sederhana dengan batas jumlah entri; entri yang paling lama tidak dipakai dibuang.

    Aman dipakai bersama oleh beberapa sesi Streamlit (thread berbeda).
    """

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            if key not in self._d:
                return default
            self._d.move_to_end(key)
            return self._d[key]

    def __getitem__(self, key):
        with self._lock:
            self._d.move_to_end(key)
            return self._d[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def __contains__(self, key):
        with self._lock:
            return key in self._d

    def __len__(self):
        return len(self._d)


def jpeg_data_url(image_bgr: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def clicker(image_bgr: np.ndarray, mode: str, vid: str, focus=None, pts=None, base=None, roi=None,
            theme=None, key=None):
    """Foto interaktif dengan zoom (scroll/tombol), geser (seret), dan penanda koreksi.

    Titik digambar langsung di browser; nilai dikirim ke Python hanya saat input lengkap.
    Semua koordinat ternormalisasi 0..1 terhadap seluruh foto.
    """
    state = {"pts": pts or [], "base": base or [], "roi": roi}
    sid = hashlib.md5(json.dumps([mode, state], sort_keys=True).encode()).hexdigest()[:10]
    return _clicker(src=jpeg_data_url(image_bgr), mode=mode, sid=sid, vid=vid, focus=focus,
                    theme=theme or {}, key=key, default=None, **state)


def strip(items: list[dict], selected: str, theme=None, key=None):
    """Deretan thumbnail untuk berpindah foto dengan cepat. items: {id, src, label, sub, tone}."""
    return _strip(items=items, selected=selected, theme=theme or {}, key=key, default=None)
