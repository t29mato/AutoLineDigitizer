# -*- coding: utf-8 -*-
"""
chart_detector.py — Chart-region detection using MinerU's PP-DocLayoutV2.

Vendored from MinerU (https://github.com/opendatalab/MinerU, MinerU Open Source
License, Apache-2.0 based — see MINERU_LICENSE.md). PP-DocLayoutV2 is an
RT-DETR layout detector with a dedicated `chart` class, distinct from `image`
(photos / SEM / schematics). This is the key advantage over DocLayout-YOLO,
whose single `figure` class returns one box around an entire composite figure.

Drop-in usage inside AutoLineDigitizer:

    from mineru_layout import ChartDetector

    det = ChartDetector(weights_dir, device="mps")   # "cpu" / "cuda" / "mps"
    charts = det.detect_charts(page_bgr)             # page rendered ~200 dpi
    # -> [{"bbox": [x0,y0,x1,y1], "bbox_norm": [...], "score": float,
    #      "label": "chart"}, ...]   (pixel coords on the input image)

Weights (one-time download, ~few hundred MB):

    pip install huggingface_hub
    python download_weights.py ./pp_doclayoutv2_weights

Composite figures: PP-DocLayoutV2 is instance-based, so well-separated panels
usually come back as separate `chart` boxes. For tightly packed composites it
may still return one merged box — `detect_charts(..., gutter_split=True)`
applies a whitespace projection-profile split as a deterministic post-step.
"""

from typing import List, Dict, Optional

import numpy as np

try:
    import cv2
except ImportError:  # cv2 only needed for gutter_split / BGR input
    cv2 = None

from .pp_doclayoutv2 import PPDocLayoutV2LayoutModel

CHART_LABEL = "chart"
IMAGE_LABEL = "image"


class ChartDetector:
    """Thin wrapper around PP-DocLayoutV2 that returns chart regions only."""

    def __init__(self, weights_dir: str, device: str = "cpu", conf: float = 0.45):
        self.model = PPDocLayoutV2LayoutModel(
            weight=weights_dir, device=device, conf=conf
        )

    def detect_layout(self, img) -> List[Dict]:
        """Full layout prediction (all classes). img: np.ndarray (RGB or BGR-
        converted beforehand) or PIL.Image."""
        if isinstance(img, np.ndarray) and img.ndim == 3 and cv2 is not None:
            # assume BGR if it came from cv2; PP-DocLayoutV2 wants RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.model.predict(img)

    def detect_charts(self, img, include_images: bool = False,
                      gutter_split: bool = False,
                      margin_frac: float = 0.01) -> List[Dict]:
        """
        Return chart bounding boxes on the given page/figure image.

        include_images: also return `image`-class regions (photos/schematics) —
            useful if you want to route them elsewhere or debug.
        gutter_split:   post-split each chart box along whitespace gutters
            (handles tightly packed composites the detector merged).
        margin_frac:    small margin added around each box so axis labels are
            not clipped (fraction of image min-dimension).
        """
        if isinstance(img, np.ndarray):
            H, W = img.shape[:2]
        else:
            W, H = img.size

        results = self.detect_layout(img)
        wanted = {CHART_LABEL} | ({IMAGE_LABEL} if include_images else set())

        out = []
        for r in results:
            if r.get("label") not in wanted:
                continue
            x0, y0, x1, y1 = [float(v) for v in r["bbox"]]
            m = margin_frac * min(H, W)
            x0, y0 = max(0.0, x0 - m), max(0.0, y0 - m)
            x1, y1 = min(float(W), x1 + m), min(float(H), y1 + m)
            out.append({
                "bbox": [x0, y0, x1, y1],
                "bbox_norm": [x0 / W, y0 / H, x1 / W, y1 / H],
                "score": float(r.get("score", 0.0)),
                "label": r["label"],
            })

        if gutter_split and cv2 is not None and isinstance(img, np.ndarray):
            out = self._split_composites(img, out)
        return out

    # ------------------------------------------------------------------
    # Whitespace gutter splitting for merged composite boxes
    # ------------------------------------------------------------------

    def _split_composites(self, img_arr: np.ndarray, boxes: List[Dict]) -> List[Dict]:
        H, W = img_arr.shape[:2]
        gray = (cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
                if img_arr.ndim == 3 else img_arr)
        split_out = []
        for b in boxes:
            if b["label"] != CHART_LABEL:
                split_out.append(b)
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in b["bbox"]]
            crop = gray[y0:y1, x0:x1]
            sub_boxes = _gutter_split(crop)
            if len(sub_boxes) <= 1:
                split_out.append(b)
                continue
            for (sx0, sy0, sx1, sy1) in sub_boxes:
                gx0, gy0 = x0 + sx0, y0 + sy0
                gx1, gy1 = x0 + sx1, y0 + sy1
                split_out.append({
                    "bbox": [float(gx0), float(gy0), float(gx1), float(gy1)],
                    "bbox_norm": [gx0 / W, gy0 / H, gx1 / W, gy1 / H],
                    "score": b["score"],
                    "label": CHART_LABEL,
                    "split_from": b["bbox"],
                })
        return split_out


def _gutter_split(gray_crop: np.ndarray,
                  min_gap_frac: float = 0.03,
                  min_panel_frac: float = 0.22,
                  ink_thresh: int = 210) -> List[tuple]:
    """Split a grayscale crop into sub-panel boxes along full-blank gutters.

    Returns a list of (x0, y0, x1, y1) in crop-local coordinates. A single
    element means no split was found. Recurses one level to handle 2x2 grids.
    """
    H, W = gray_crop.shape[:2]
    if H < 40 or W < 40:
        return [(0, 0, W, H)]
    ink = (gray_crop < ink_thresh).astype(np.uint8)

    def gutters(profile, length, min_gap):
        blank = profile == 0
        cuts, start = [], None
        for i, is_blank in enumerate(blank):
            if is_blank and start is None:
                start = i
            elif not is_blank and start is not None:
                if i - start >= min_gap and start > 0:
                    cuts.append((start + i) // 2)
                start = None
        return cuts

    h_cuts = gutters(ink.sum(axis=1), H, max(4, int(H * min_gap_frac)))
    v_cuts = gutters(ink.sum(axis=0), W, max(4, int(W * min_gap_frac)))
    cuts, axis = (h_cuts, 0) if len(h_cuts) >= len(v_cuts) else (v_cuts, 1)
    if not cuts:
        return [(0, 0, W, H)]

    panels, prev = [], 0
    limit = H if axis == 0 else W
    for c in cuts + [limit]:
        if axis == 0:
            sub = (0, prev, W, c)
            size_ok = (c - prev) > min_panel_frac * H
        else:
            sub = (prev, 0, c, H)
            size_ok = (c - prev) > min_panel_frac * W
        if size_ok:
            # one level of recursion for grid layouts
            sx0, sy0, sx1, sy1 = sub
            inner = _gutter_split(gray_crop[sy0:sy1, sx0:sx1])
            for (ix0, iy0, ix1, iy1) in inner:
                panels.append((sx0 + ix0, sy0 + iy0, sx0 + ix1, sy0 + iy1))
        prev = c
    return panels if panels else [(0, 0, W, H)]
