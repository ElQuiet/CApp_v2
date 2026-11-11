"""Fungsi tingkat-modul untuk analisis di proses terpisah (ProcessPoolExecutor)."""
from __future__ import annotations

import cv2
import numpy as np

from .analyze import AnalysisResult, analyze


def analyze_bytes(data: bytes, roi=None, baseline=None, manual_points=None) -> AnalysisResult:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return analyze(img, roi=roi, baseline=baseline, manual_points=manual_points)
