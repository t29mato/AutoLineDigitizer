# -*- coding: utf-8 -*-
"""
LineFormer Desktop App
Chart line data extraction using LineFormer with Flet UI.

Features:
- LineFormer inference + color-aware refinement
- ChartDete axis detection + OCR
- VLM verification (Claude)
- Manual editing: click-to-select line, Erase, Add (spline), Delete Line
- Live data table with axis-calibrated values + CSV export
"""

import sys
import os

APP_VERSION = "dev"

if getattr(sys, 'frozen', False):
    SCRIPT_DIR = sys._MEIPASS
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CHARTDETE_DIR = os.path.join(SCRIPT_DIR, "submodules", "chartdete")
LINEFORMER_DIR = os.path.join(SCRIPT_DIR, "submodules", "lineformer")
MMDET_DIR = os.path.join(LINEFORMER_DIR, "mmdetection")
SRC_DIR = os.path.join(SCRIPT_DIR, "src")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, MMDET_DIR)
sys.path.insert(0, LINEFORMER_DIR)
sys.path.insert(0, CHARTDETE_DIR)

CHARTDETE_AVAILABLE = False
try:
    import mmdet  # noqa: F401
    from mmdet.models.roi_heads.cascade_roi_head_LGF import CascadeRoIHead_LGF  # noqa: F401
    CHARTDETE_AVAILABLE = True
except Exception as e:
    print(f"ChartDete custom models not available: {e}")

import flet as ft
import cv2
import numpy as np
import json
import io
import csv
import math
import zipfile
import tarfile
import base64
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from color_line_extractor import ColorLineExtractor
from smart_axis_extractor import SmartAxisExtractor

try:
    from vlm_verifier import VLMVerifier, ANTHROPIC_AVAILABLE
    VLM_VERIFIER_AVAILABLE = True
except Exception as _vlm_err:
    VLM_VERIFIER_AVAILABLE = False
    ANTHROPIC_AVAILABLE = False
    print(f"VLM verifier not available: {_vlm_err}")

try:
    from line_marker_detector import find_markers_on_traced_line
    MARKER_DETECTOR_AVAILABLE = True
except Exception as _md_err:
    MARKER_DETECTOR_AVAILABLE = False
    print(f"line_marker_detector not available: {_md_err}")

# PDF figure extraction (shared with batch_extract via pdf_figures.py).
try:
    from pdf_figures import (
        extract_figures,
        extract_figures_via_page_render,
        extract_figures_via_doclayout,
        extract_figures_via_mineru,
        doclayout_available,
        mineru_available,
        render_pdf_page,
    )
    PDF_SUPPORT = True
    DOCLAYOUT_OK = doclayout_available()
    MINERU_OK = mineru_available()
except Exception as _pdf_err:
    PDF_SUPPORT = False
    DOCLAYOUT_OK = False
    MINERU_OK = False
    print(f"PDF support not available: {_pdf_err}")

# VLM screener for the page-render fallback (vector-only figures).
try:
    from vlm_screener import VLMScreener, ANTHROPIC_AVAILABLE as VLM_SCREENER_SDK_AVAILABLE
    VLM_SCREENER_AVAILABLE = True
except Exception as _scr_err:
    VLMScreener = None
    VLM_SCREENER_SDK_AVAILABLE = False
    VLM_SCREENER_AVAILABLE = False
    print(f"VLM screener not available: {_scr_err}")

# KMDS structured-metadata extraction (parallel Claude calls on the whole PDF).
try:
    import kmds_parallel
    KMDS_AVAILABLE = kmds_parallel.ANTHROPIC_AVAILABLE
except Exception as _kmds_err:
    kmds_parallel = None
    KMDS_AVAILABLE = False
    print(f"KMDS extraction not available: {_kmds_err}")

GITHUB_REPO = "t29mato/AutoLineDigitizer"
GITHUB_RELEASE_TAG = "models"

MODEL_FILES = {
    "iter_3000.pth": "LineFormer",
    "checkpoint.pth": "ChartDete",
}

MODEL_HASHES = {
    "iter_3000.pth": "ac03d7d52a11ce25",
    "checkpoint.pth": "aef812b0e37faf7c",
    "lineformer_battery_iter_5000.pth": "587f492d381674bd",
    "lineformer_battery_best_iter_1300.pth": "e3189abf87a7bff7",
}

HUGGINGFACE_REPO = "t29mato/lineformer-battery-finetuned"

LINEFORMER_MODELS = {
    "general": {"name": "General", "checkpoint": "iter_3000.pth"},
    "battery_finetuned": {"name": "Battery (fine-tuned)", "checkpoint": "lineformer_battery_finetuned.pth"},
    "general_v2": {"name": "General (multi-category)", "checkpoint": "lineformer_general.pth"},
    "battery_realistic": {"name": "Battery (realistic)", "checkpoint": "lineformer_battery_realistic.pth"},
    "battery_iter5000": {
        "name": "Battery (iter_5000)",
        "checkpoint": "lineformer_battery_iter_5000.pth",
        "huggingface_filename": "lineformer_battery_iter_5000.pth",
    },
    "battery_best": {
        "name": "Battery (best)",
        "checkpoint": "lineformer_battery_best_iter_1300.pth",
        "huggingface_filename": "lineformer_battery_best_iter_1300.pth",
    },
}


def get_models_dir():
    if getattr(sys, 'frozen', False):
        if sys.platform == 'darwin':
            data_dir = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'AutoLineDigitizer')
        elif sys.platform == 'win32':
            data_dir = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), 'AutoLineDigitizer')
        else:
            data_dir = os.path.join(os.path.expanduser('~'), '.autolinedigitizer')
        models_dir = os.path.join(data_dir, 'models')
    else:
        models_dir = os.path.join(SCRIPT_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def check_models_exist():
    models_dir = get_models_dir()
    missing = []
    for filename in MODEL_FILES:
        if not os.path.exists(os.path.join(models_dir, filename)):
            missing.append(filename)
    return missing


def verify_model_hash(file_path, expected_filename):
    import hashlib
    expected = MODEL_HASHES.get(expected_filename)
    if not expected:
        return True, ""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    actual = h.hexdigest()[:16]
    return actual == expected, actual


def download_file(url, dest_path, progress_callback=None):
    import ssl
    tmp_path = dest_path + '.tmp'
    req = urllib.request.Request(url)
    try:
        response_ctx = urllib.request.urlopen(req)
    except Exception:
        ctx = ssl._create_unverified_context()
        response_ctx = urllib.request.urlopen(req, context=ctx)
    with response_ctx as response:
        total_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        block_size = 1024 * 1024
        with open(tmp_path, 'wb') as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total_size > 0:
                    progress_callback(downloaded, total_size)
    os.replace(tmp_path, dest_path)


GITHUB_MODELS_URL = f"https://github.com/{GITHUB_REPO}/releases/tag/{GITHUB_RELEASE_TAG}"
HUGGINGFACE_MODELS_URL = f"https://huggingface.co/{HUGGINGFACE_REPO}/tree/main"


def download_model(filename, models_dir, progress_callback=None):
    url = f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}/{filename}"
    download_file(url, os.path.join(models_dir, filename), progress_callback)


def download_finetuned_model(model_info, dest_path, progress_callback=None):
    hf_filename = model_info.get("huggingface_filename")
    if hf_filename:
        url = f"https://huggingface.co/{HUGGINGFACE_REPO}/resolve/main/{hf_filename}"
        download_file(url, dest_path, progress_callback)


class LineFormerApp:
    def __init__(self):
        self.infer_module = None
        self.chartdete_module = None
        self.current_image = None
        self.current_image_path = None
        self.data_series = None
        self.result_image = None
        self.axis_config = None
        self.ocr_results = None
        self.raw_lines = None
        self.current_model_key = "general"
        self.vlm = None
        self.cached_plot_area = None

        self.sort_mode = "mean_y_desc"
        self.downsample_mode = "max_points"
        self.fixed_step = 10
        self.max_points = 20
        self.auto_axis = True
        self.use_color_refinement = True
        self.smart_axis = SmartAxisExtractor()

        # Manual editing state
        self.selected_line_idx = None
        self.edit_mode = None
        self.add_anchors = []
        self.eraser_radius = 18

        # Axis label names (auto from OCR, user-editable)
        self.x_axis_name = ""
        self.y_axis_name = ""

        # PDF batch state: figures pulled from a loaded PDF for the gallery.
        self.pdf_path = None
        self.pdf_figures = []        # list of (img_bgr, meta) for the gallery
        self.current_figure_idx = None

        # Paper-record handoff: parsed KMDS record + per-figure digitizations
        # staged for "Open Paper Record" (the KMDS viewer).
        self.kmds_record = None
        self.fig_digitizations = {}

    def _vlm_screener_available(self):
        return VLM_SCREENER_AVAILABLE and VLM_SCREENER_SDK_AVAILABLE

    def extract_pdf_figures(self, pdf_path, prefer_vlm=True,
                            vlm_model="claude-haiku-4-5-20251001",
                            page_dpi=200, progress=None,
                            refine=True, refine_model="claude-sonnet-4-6",
                            detector="mineru"):
        """
        Pull figures out of a PDF for the thumbnail gallery.

        Figure-location strategies (the `detector` argument):
          - "mineru": MinerU PP-DocLayoutV2 (RT-DETR). Dedicated `chart` class
            (separates charts from photos) and instance-based, so composite
            panels usually come back separate. Offline, no API. Best; default.
          - "doclayout": DocLayout-YOLO locates whole figures (offline, no API).
          - "vlm": render every page and ask Claude to locate chart bounding
            boxes (+ refine). Costs API.
          - "raster": pull embedded bitmaps via PyMuPDF (offline, no models).

        Falls back to raster if the chosen detector finds nothing. Returns a list
        of (img_bgr, meta) and records it on self.pdf_figures.
        """
        if not PDF_SUPPORT:
            raise RuntimeError("PDF support unavailable (PyMuPDF not installed).")

        figs = []
        use_mineru = (detector == "mineru") and MINERU_OK
        use_vlm = (detector == "vlm") and prefer_vlm and self._vlm_screener_available()
        use_doclayout = (detector == "doclayout") and DOCLAYOUT_OK

        # 0a) MinerU PP-DocLayoutV2 chart location (offline, no API).
        if use_mineru:
            try:
                if progress:
                    progress(0, "mineru-start")
                for img_bgr, meta in extract_figures_via_mineru(
                    pdf_path, dpi=page_dpi, min_size=200
                ):
                    meta = dict(meta)
                    meta["source"] = "mineru"
                    figs.append((img_bgr, meta))
                    if progress:
                        progress(len(figs), "mineru")
            except Exception as e:
                print(f"[pdf] MinerU PP-DocLayoutV2 failed: {e}")

        # 0b) DocLayout-YOLO figure location (offline, no API).
        if use_doclayout:
            try:
                if progress:
                    progress(0, "doclayout-start")
                for img_bgr, meta in extract_figures_via_doclayout(
                    pdf_path, dpi=page_dpi, min_size=200
                ):
                    meta = dict(meta)
                    meta["source"] = "doclayout"
                    figs.append((img_bgr, meta))
                    if progress:
                        progress(len(figs), "doclayout")
            except Exception as e:
                print(f"[pdf] DocLayout-YOLO failed: {e}")

        # 1) VLM page-render (per-panel chart isolation, vector-aware).
        if use_vlm and not figs:
            try:
                if progress:
                    progress(0, "vlm-start")
                screener = VLMScreener(model=vlm_model)
                for img_bgr, meta in extract_figures_via_page_render(
                    pdf_path, screener, dpi=page_dpi, min_size=200,
                    refine=refine, refine_model=refine_model,
                ):
                    meta = dict(meta)
                    meta["source"] = "vlm"
                    figs.append((img_bgr, meta))
                    if progress:
                        progress(len(figs), "vlm")
            except Exception as e:
                print(f"[pdf] VLM page-render failed: {e}")

        # 2) Fallback: embedded raster bitmaps (no API key, or VLM found none).
        if not figs:
            if use_vlm:
                print("[pdf] VLM found no charts; falling back to raster bitmaps.")
            try:
                for img_bgr, meta in extract_figures(pdf_path, min_size=200):
                    meta = dict(meta)
                    meta["source"] = "raster"
                    figs.append((img_bgr, meta))
                    if progress:
                        progress(len(figs), "raster")
            except Exception as e:
                print(f"[pdf] raster extraction failed: {e}")

        self.pdf_path = pdf_path
        self.pdf_figures = figs
        return figs

    def load_lineformer_model(self, model_key=None, progress_callback=None):
        import infer
        if model_key is not None:
            self.current_model_key = model_key
        model_info = LINEFORMER_MODELS[self.current_model_key]
        ckpt = os.path.join(get_models_dir(), model_info["checkpoint"])
        if not os.path.exists(ckpt) and "huggingface_filename" in model_info:
            download_finetuned_model(model_info, ckpt, progress_callback)
        CONFIG = os.path.join(LINEFORMER_DIR, "lineformer_swin_t_config.py")
        DEVICE = "cpu"
        infer.load_model(CONFIG, ckpt, DEVICE)
        self.infer_module = infer

    def load_chartdete_model(self):
        import chartdete_infer
        models_dir = get_models_dir()
        config_path = os.path.join(SCRIPT_DIR, 'config', 'chartdete_config.py')
        checkpoint_path = os.path.join(models_dir, 'checkpoint.pth')
        chartdete_infer.load_chartdete_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device='cpu',
        )
        self.chartdete_module = chartdete_infer

    def detect_axis_calibration(self, img):
        if self.chartdete_module is None:
            return None, None
        detections = self.chartdete_module.detect_chart_elements(img, score_thr=0.3)
        axis_info = self.chartdete_module.get_axis_info(detections, img=img, with_ocr=True)
        plot_area = axis_info.get('plot_area')
        ocr_results = axis_info.get('ocr_results', {})
        self.cached_plot_area = plot_area
        if plot_area is None:
            print("[axis] No plot_area detected; cannot calibrate.")
            return None, ocr_results
        candidate_labels = self._collect_text_boxes(ocr_results)
        print(f"[axis] plot_area={plot_area}, collected {len(candidate_labels)} text boxes")
        result = self.smart_axis.extract(img, plot_area=plot_area, candidate_labels=candidate_labels)
        print(f"[axis] confidence={result.confidence}, x_ticks={len(result.x_labels)}, y_ticks={len(result.y_labels)}")
        if result.x_fit:
            print(f"[axis] X fit: residual={result.x_fit.rms_residual_px:.2f}px, dropped {result.x_fit.n_ticks_dropped}, log={result.x_fit.is_log}")
        if result.y_fit:
            print(f"[axis] Y fit: residual={result.y_fit.rms_residual_px:.2f}px, dropped {result.y_fit.n_ticks_dropped}, log={result.y_fit.is_log}")
        for w in result.warnings:
            print(f"[axis] warning: {w}")
        return result.axis_config, ocr_results

    def _collect_text_boxes(self, ocr_results):
        if not hasattr(self, '_axis_debug_printed'):
            try:
                preview = json.dumps(ocr_results, default=str, indent=2)[:2000]
            except Exception:
                preview = repr(ocr_results)[:2000]
            print(f"[axis-debug] ocr_results structure:\n{preview}")
            self._axis_debug_printed = True
        boxes = []
        if not isinstance(ocr_results, dict):
            return boxes
        for category, items in ocr_results.items():
            if not isinstance(items, (list, tuple)):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = item.get('text') or item.get('label') or item.get('value')
                bbox = item.get('bbox') or item.get('box') or item.get('coords')
                if text is None or bbox is None or len(bbox) < 4:
                    continue
                a, b, c, d = bbox[:4]
                if c > a and d > b and (c - a) < 1000 and (d - b) < 200:
                    x, y, w, h = a, b, c - a, d - b
                else:
                    x, y, w, h = a, b, c, d
                boxes.append((str(text), (int(x), int(y), int(w), int(h))))
        return boxes

    def _detect_plot_area_only(self, img):
        if self.chartdete_module is None:
            return None
        try:
            detections = self.chartdete_module.detect_chart_elements(img, score_thr=0.3)
            axis_info = self.chartdete_module.get_axis_info(detections, img=img, with_ocr=False)
            return axis_info.get('plot_area')
        except Exception:
            return None

    def apply_vlm_axis_correction(self, img, model=None):
        """
        Use the VLM to re-read the axis tick labels (scientific notation, log
        scales) and rebuild axis_config. Returns (axis_config, notes). Updates
        self.axis_config on success. Raises on hard failures.
        """
        from vlm_verifier import VLMVerifier
        if self.vlm is None:
            self.vlm = VLMVerifier(verify_ssl=True)
        plot_area = self.cached_plot_area or self._detect_plot_area_only(img)
        if plot_area is None:
            raise RuntimeError("No plot area detected; cannot recalibrate axes.")
        parsed = self.vlm.verify_axis_calibration(
            img, plot_area, axis_config=self.axis_config, model=model)
        cfg, notes = self._axis_config_from_vlm_ticks(parsed, plot_area)
        if cfg is not None:
            self.axis_config = cfg
            self.cached_plot_area = plot_area
        return cfg, notes

    def _axis_config_from_vlm_ticks(self, parsed, plot_area):
        """Convert VLM {value, frac} ticks into the app's axis_config via the
        existing least-squares fitter. frac is position along the axis inside
        the plot box (x: 0=left..1=right, y: 0=bottom..1=top)."""
        from smart_axis_extractor import TickLabel, SmartAxisExtractor
        px0, py0, px1, py1 = [float(v) for v in plot_area]

        def to_ticks(axis_obj, is_x):
            out = []
            for t in (axis_obj or {}).get("ticks", []) or []:
                try:
                    val = float(t["value"])
                    frac = max(0.0, min(1.0, float(t["frac"])))
                except (KeyError, TypeError, ValueError):
                    continue
                pix = (px0 + frac * (px1 - px0)) if is_x else (py1 - frac * (py1 - py0))
                out.append(TickLabel(value=val, raw_text=str(t.get("value")),
                                     pixel=float(pix), box=(0, 0, 0, 0)))
            return out

        x_obj = parsed.get("x_axis") or {}
        y_obj = parsed.get("y_axis") or {}
        x_ticks = to_ticks(x_obj, True)
        y_ticks = to_ticks(y_obj, False)
        if len(x_ticks) < 2 or len(y_ticks) < 2:
            return None, "VLM returned too few ticks to calibrate."

        def fit(ticks, is_log_hint):
            # Honor the VLM's log/linear call when it's consistent with the data.
            if is_log_hint is not None:
                if not is_log_hint or all(t.value > 0 for t in ticks):
                    return SmartAxisExtractor._fit_linear(ticks, log=bool(is_log_hint))
            return self.smart_axis._fit_axis(ticks)

        x_fit = fit(x_ticks, x_obj.get("is_log"))
        y_fit = fit(y_ticks, y_obj.get("is_log"))
        cfg = SmartAxisExtractor._build_axis_config(x_fit, y_fit, plot_area)
        # Snap floating-point "almost zero" edge values to 0 for clean display.
        for axis_keys, span in ((("x1_val", "x2_val"), not cfg["xIsLogScale"]),
                                (("y1_val", "y2_val"), not cfg["yIsLogScale"])):
            if not span:
                continue
            scale = max(abs(cfg[axis_keys[0]]), abs(cfg[axis_keys[1]]), 1e-12)
            for k in axis_keys:
                if abs(cfg[k]) < 1e-6 * scale:
                    cfg[k] = 0.0
        return cfg, str(parsed.get("notes", "")).strip()

    def downsample_points(self, points):
        if len(points) <= 1:
            return points
        if self.downsample_mode == "none":
            return points
        elif self.downsample_mode == "fixed":
            return points[::self.fixed_step]
        elif self.downsample_mode == "max_points":
            if len(points) <= self.max_points:
                return points
            step = max(1, len(points) // self.max_points)
            return points[::step]
        elif self.downsample_mode == "arc_length":
            if len(points) <= self.max_points:
                return points
            return self._arc_length_resample(points, self.max_points)
        return points

    def _arc_length_resample(self, points, n_points):
        pts = np.array(points, dtype=float)
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        cum_arc = np.zeros(len(pts))
        cum_arc[1:] = np.cumsum(seg_lengths)
        total_length = cum_arc[-1]
        if total_length == 0:
            return [points[0]]
        target_distances = np.linspace(0, total_length, n_points)
        result = []
        seg_idx = 0
        for d in target_distances:
            while seg_idx < len(seg_lengths) - 1 and cum_arc[seg_idx + 1] < d:
                seg_idx += 1
            seg_span = cum_arc[seg_idx + 1] - cum_arc[seg_idx]
            t = 0.0 if seg_span == 0 else (d - cum_arc[seg_idx]) / seg_span
            x = pts[seg_idx, 0] + t * (pts[seg_idx + 1, 0] - pts[seg_idx, 0])
            y = pts[seg_idx, 1] + t * (pts[seg_idx + 1, 1] - pts[seg_idx, 1])
            result.append([int(round(x)), int(round(y))])
        return result

    def sort_data_series(self, data_series):
        if self.sort_mode == "original" or len(data_series) == 0:
            return data_series
        if self.sort_mode == "mean_y_desc":
            return sorted(data_series, key=lambda s: np.mean([pt[1] for pt in s["points"]]) if s["points"] else 0)
        elif self.sort_mode == "mean_y_asc":
            return sorted(data_series, key=lambda s: np.mean([pt[1] for pt in s["points"]]) if s["points"] else 0, reverse=True)
        return data_series

    def extract_lines(self, img):
        line_dataseries = self.infer_module.get_dataseries(img, to_clean=False)
        raw_from_model = []
        for line in line_dataseries:
            if len(line) == 0:
                continue
            raw_from_model.append([[int(pt['x']), int(pt['y'])] for pt in line])
        if self.use_color_refinement and raw_from_model:
            try:
                plot_box = self.cached_plot_area
                if plot_box is None:
                    plot_box = self._detect_plot_area_only(img)
                extractor = ColorLineExtractor(img, plot_area=plot_box)
                refined, _ = extractor.refine_lines(raw_from_model)
                self.raw_lines = refined if refined else raw_from_model
            except Exception as e:
                print(f"Color refinement skipped ({e}); using raw LineFormer output.")
                self.raw_lines = raw_from_model
        else:
            self.raw_lines = raw_from_model
        return self.apply_downsample_and_sort()

    def apply_downsample_and_sort(self):
        if self.raw_lines is None:
            return []
        data_series = []
        for all_points in self.raw_lines:
            points = self.downsample_points(all_points)
            data_series.append({"points": points})
        return self.sort_data_series(data_series)

    # --- Manual editing helpers ---
    def spline_interpolate(self, anchors, spacing_px=8):
        pts = np.array(anchors, dtype=float)
        if len(pts) < 2:
            return [[int(round(p[0])), int(round(p[1]))] for p in pts]
        keep = [0]
        for i in range(1, len(pts)):
            dx = pts[i, 0] - pts[keep[-1], 0]
            dy = pts[i, 1] - pts[keep[-1], 1]
            if dx * dx + dy * dy > 1.0:
                keep.append(i)
        pts = pts[keep]
        if len(pts) < 2:
            return [[int(round(pts[0, 0])), int(round(pts[0, 1]))]]
        diffs = np.diff(pts, axis=0)
        seg = np.sqrt((diffs ** 2).sum(axis=1))
        t = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(t[-1])
        if total <= 0:
            return [[int(round(pts[0, 0])), int(round(pts[0, 1]))]]
        t = t / total
        n_out = max(30, int(total / float(spacing_px)))
        t_new = np.linspace(0.0, 1.0, n_out)
        try:
            from scipy.interpolate import interp1d
            if len(pts) >= 4:
                kind = "cubic"
            elif len(pts) == 3:
                kind = "quadratic"
            else:
                kind = "linear"
            fx = interp1d(t, pts[:, 0], kind=kind)
            fy = interp1d(t, pts[:, 1], kind=kind)
            xs, ys = fx(t_new), fy(t_new)
        except Exception:
            xs = np.interp(t_new, t, pts[:, 0])
            ys = np.interp(t_new, t, pts[:, 1])
        return [[int(round(float(x))), int(round(float(y)))] for x, y in zip(xs, ys)]

    def erase_near(self, line_idx, x, y, radius):
        if line_idx is None or self.data_series is None:
            return 0
        if not (0 <= line_idx < len(self.data_series)):
            return 0
        pts = self.data_series[line_idx]["points"]
        if not pts:
            return 0
        r2 = radius * radius
        kept = [p for p in pts if (p[0] - x) ** 2 + (p[1] - y) ** 2 > r2]
        removed = len(pts) - len(kept)
        self.data_series[line_idx]["points"] = kept
        return removed

    def add_spline_to_line(self, line_idx, anchors):
        if line_idx is None or self.data_series is None:
            return
        if not (0 <= line_idx < len(self.data_series)):
            return
        new_pts = self.spline_interpolate(anchors)
        pts = self.data_series[line_idx]["points"]
        pts.extend(new_pts)
        pts.sort(key=lambda p: (p[0], p[1]))
        self.data_series[line_idx]["points"] = pts

    def delete_line(self, line_idx):
        """Remove an entire line. Keeps raw_lines in sync so it doesn't reappear."""
        if line_idx is None or self.data_series is None:
            return False
        if not (0 <= line_idx < len(self.data_series)):
            return False
        del self.data_series[line_idx]
        if isinstance(self.raw_lines, list) and 0 <= line_idx < len(self.raw_lines):
            del self.raw_lines[line_idx]
        return True

    # --- Axis calibration + CSV (NEW) ---
    def pixel_to_data(self, px, py):
        """Convert pixel (px, py) to axis-calibrated (x, y). Raw pixels if no axis."""
        cfg = self.axis_config
        if cfg is None:
            return float(px), float(py)
        x1_px = float(cfg.get('x1_px', 0.0))
        x2_px = float(cfg.get('x2_px', 1.0))
        x1_val = float(cfg.get('x1_val', 0.0))
        x2_val = float(cfg.get('x2_val', 1.0))
        is_log_x = bool(cfg.get('xIsLogScale', False))
        y1_py = float(cfg.get('y1_py', 0.0))
        y2_py = float(cfg.get('y2_py', 1.0))
        y1_val = float(cfg.get('y1_val', 0.0))
        y2_val = float(cfg.get('y2_val', 1.0))
        is_log_y = bool(cfg.get('yIsLogScale', False))
        if x2_px == x1_px:
            x_val = x1_val
        else:
            t = (float(px) - x1_px) / (x2_px - x1_px)
            if is_log_x and x1_val > 0 and x2_val > 0:
                x_val = math.exp(math.log(x1_val) + t * (math.log(x2_val) - math.log(x1_val)))
            else:
                x_val = x1_val + t * (x2_val - x1_val)
        if y2_py == y1_py:
            y_val = y1_val
        else:
            t = (float(py) - y1_py) / (y2_py - y1_py)
            if is_log_y and y1_val > 0 and y2_val > 0:
                y_val = math.exp(math.log(y1_val) + t * (math.log(y2_val) - math.log(y1_val)))
            else:
                y_val = y1_val + t * (y2_val - y1_val)
        return float(x_val), float(y_val)

    def get_axis_titles(self):
        """Try to pull X/Y axis titles from cached ocr_results. Returns (x, y)."""
        x_name, y_name = "", ""
        if not isinstance(self.ocr_results, dict):
            return x_name, y_name
        x_keys = ('x_axis_title', 'xaxis_title', 'x_title', 'x-axis-title')
        y_keys = ('y_axis_title', 'yaxis_title', 'y_title', 'y-axis-title')

        def _first_text(items):
            if not isinstance(items, (list, tuple)) or not items:
                return ""
            for it in items:
                if isinstance(it, dict):
                    t = it.get('text') or it.get('label') or it.get('value')
                    if t:
                        return str(t).strip()
            return ""

        for k in x_keys:
            if k in self.ocr_results:
                x_name = _first_text(self.ocr_results[k])
                if x_name:
                    break
        for k in y_keys:
            if k in self.ocr_results:
                y_name = _first_text(self.ocr_results[k])
                if y_name:
                    break
        return x_name, y_name

    def line_name(self, idx):
        """Display name for a detected line: its VLM legend label if present,
        else 'Line N'."""
        if self.data_series and 0 <= idx < len(self.data_series):
            lab = self.data_series[idx].get("label")
            if lab:
                return str(lab)
        return f"Line {idx + 1}"

    def to_wide_csv(self, x_name="X", y_name="Y"):
        """Wide CSV: per-line (X, Y) column pairs, axis-calibrated."""
        if not self.data_series:
            return ""
        lines_data = []
        for series in self.data_series:
            pts = []
            for pt in series.get("points", []):
                xv, yv = self.pixel_to_data(pt[0], pt[1])
                pts.append((xv, yv))
            lines_data.append(pts)
        max_len = max((len(L) for L in lines_data), default=0)
        if max_len == 0:
            return ""
        buf = io.StringIO()
        writer = csv.writer(buf)
        header = []
        for i in range(len(lines_data)):
            nm = self.line_name(i)
            header.append(f"{nm} {x_name}".strip())
            header.append(f"{nm} {y_name}".strip())
        writer.writerow(header)
        for r in range(max_len):
            row = []
            for L in lines_data:
                if r < len(L):
                    row.append(f"{L[r][0]:.6g}")
                    row.append(f"{L[r][1]:.6g}")
                else:
                    row.append("")
                    row.append("")
            writer.writerow(row)
        return buf.getvalue()

    def draw_points_on_image(self, img, data_series, axis_config=None, highlight_line_idx=None):
        import line_utils
        result_img = img.copy()
        num_lines = len(data_series)
        colors = list(line_utils.get_distinct_colors(num_lines))
        markers = [cv2.MARKER_CROSS, cv2.MARKER_DIAMOND, cv2.MARKER_SQUARE,
                   cv2.MARKER_TRIANGLE_UP, cv2.MARKER_TRIANGLE_DOWN, cv2.MARKER_STAR]
        for line_idx, series in enumerate(data_series):
            color = colors[line_idx]
            marker = markers[line_idx % len(markers)]
            if highlight_line_idx is None:
                draw_color, size, thickness = color, 8, 2
            elif line_idx == highlight_line_idx:
                draw_color, size, thickness = color, 14, 3
            else:
                draw_color = tuple(int(c + (255 - c) * 0.75) for c in color)
                size, thickness = 5, 1
            for pt in series["points"]:
                x, y = int(pt[0]), int(pt[1])
                cv2.drawMarker(result_img, (x, y), draw_color, marker, markerSize=size, thickness=thickness)
            # Legend label (from "Label Lines (AI)") drawn at the line's left end.
            lab = series.get("label")
            if lab and series["points"]:
                lx, ly = min(series["points"], key=lambda p: p[0])
                tx, ty = int(lx) + 6, int(ly) - 6
                (tw, thh), bl = cv2.getTextSize(str(lab), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(result_img, (tx - 2, ty - thh - 3),
                              (tx + tw + 2, ty + bl), (255, 255, 255), -1)
                cv2.putText(result_img, str(lab), (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if axis_config is not None:
            calib_color = (255, 0, 255)
            outline_color = (0, 0, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            def draw_text_with_bg(img, text, pos, color):
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                x, y = pos
                padding = 3
                cv2.rectangle(img, (x - padding, y - text_h - padding),
                             (x + text_w + padding, y + baseline + padding), (255, 255, 255), -1)
                cv2.rectangle(img, (x - padding, y - text_h - padding),
                             (x + text_w + padding, y + baseline + padding), outline_color, 1)
                cv2.putText(img, text, (x, y), font, font_scale, color, thickness)

            def draw_calib_point(img, x, y, color, label, label_offset):
                cv2.circle(img, (x, y), 12, outline_color, 3)
                cv2.circle(img, (x, y), 10, color, -1)
                cv2.circle(img, (x, y), 10, outline_color, 2)
                cv2.line(img, (x - 6, y), (x + 6, y), outline_color, 2)
                cv2.line(img, (x, y - 6), (x, y + 6), outline_color, 2)
                draw_text_with_bg(img, label, (x + label_offset[0], y + label_offset[1]), color)

            x1_x, x1_y = int(axis_config['x1_px']), int(axis_config['x1_py'])
            x2_x, x2_y = int(axis_config['x2_px']), int(axis_config['x2_py'])
            y1_x, y1_y = int(axis_config['y1_px']), int(axis_config['y1_py'])
            y2_x, y2_y = int(axis_config['y2_px']), int(axis_config['y2_py'])
            dash_length = 10
            gap_length = 5
            dx, dy = x2_x - x1_x, x2_y - x1_y
            dist = max(1, int(np.sqrt(dx*dx + dy*dy)))
            for i in range(0, dist, dash_length + gap_length):
                sx = int(x1_x + dx * i / dist); sy = int(x1_y + dy * i / dist)
                ei = min(i + dash_length, dist)
                ex = int(x1_x + dx * ei / dist); ey = int(x1_y + dy * ei / dist)
                cv2.line(result_img, (sx, sy), (ex, ey), calib_color, 2)
            dx, dy = y2_x - y1_x, y2_y - y1_y
            dist = max(1, int(np.sqrt(dx*dx + dy*dy)))
            for i in range(0, dist, dash_length + gap_length):
                sx = int(y1_x + dx * i / dist); sy = int(y1_y + dy * i / dist)
                ei = min(i + dash_length, dist)
                ex = int(y1_x + dx * ei / dist); ey = int(y1_y + dy * ei / dist)
                cv2.line(result_img, (sx, sy), (ex, ey), calib_color, 2)
            draw_calib_point(result_img, x1_x, x1_y, calib_color, f"X1={axis_config['x1_val']}", (15, -5))
            draw_calib_point(result_img, x2_x, x2_y, calib_color, f"X2={axis_config['x2_val']}", (-100, -5))
            draw_calib_point(result_img, y1_x, y1_y, calib_color, f"Y1={axis_config['y1_val']}", (15, 20))
            draw_calib_point(result_img, y2_x, y2_y, calib_color, f"Y2={axis_config['y2_val']}", (15, -5))
        return result_img

    def convert_to_starry_digitizer_format(self, data_series, img_shape, axis_config=None):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        if axis_config is None:
            axis_set = {
                "id": 1, "name": "XY Axes 1",
                "x1": {"name": "x1", "value": 0, "coord": {"xPx": 0, "yPx": float(img_shape[0])}},
                "x2": {"name": "x2", "value": 100, "coord": {"xPx": float(img_shape[1]), "yPx": float(img_shape[0])}},
                "y1": {"name": "y1", "value": 0, "coord": {"xPx": 0, "yPx": float(img_shape[0])}},
                "y2": {"name": "y2", "value": 100, "coord": {"xPx": 0, "yPx": 0}},
                "xIsLogScale": False, "yIsLogScale": False,
                "considerGraphTilt": False, "pointMode": 0, "isVisible": True
            }
        else:
            axis_set = {
                "id": 1, "name": "XY Axes 1",
                "x1": {"name": "x1", "value": axis_config["x1_val"], "coord": {"xPx": axis_config["x1_px"], "yPx": axis_config["x1_py"]}},
                "x2": {"name": "x2", "value": axis_config["x2_val"], "coord": {"xPx": axis_config["x2_px"], "yPx": axis_config["x2_py"]}},
                "y1": {"name": "y1", "value": axis_config["y1_val"], "coord": {"xPx": axis_config["y1_px"], "yPx": axis_config["y1_py"]}},
                "y2": {"name": "y2", "value": axis_config["y2_val"], "coord": {"xPx": axis_config["y2_px"], "yPx": axis_config["y2_py"]}},
                "xIsLogScale": axis_config.get("xIsLogScale", False),
                "yIsLogScale": axis_config.get("yIsLogScale", False),
                "considerGraphTilt": False, "pointMode": 0, "isVisible": True
            }
        datasets = [{"id": 1, "name": "dataset 1", "axisSetId": 1, "points": [],
                     "visiblePointIds": [], "manuallyAddedPointIds": []}]
        for idx, series in enumerate(data_series):
            points = []; visible_ids = []
            for pt_idx, pt in enumerate(series["points"]):
                pt_id = pt_idx + 1
                points.append({"id": pt_id, "xPx": float(pt[0]), "yPx": float(pt[1])})
                visible_ids.append(pt_id)
            datasets.append({"id": idx + 2, "name": f"Line {idx + 1}", "axisSetId": 1,
                             "points": points, "visiblePointIds": visible_ids, "manuallyAddedPointIds": []})
        return {"version": "1.11.2", "timestamp": timestamp,
                "axisSets": [axis_set], "activeAxisSetId": 1,
                "datasets": datasets, "activeDatasetId": len(datasets),
                "canvasHandler": {"scale": 1.0, "manualMode": 0}}

    def convert_to_wpd_format(self, data_series, img_shape, axis_config=None):
        if axis_config is None:
            calibration_points = [
                {"px": 0.0, "py": float(img_shape[0]), "dx": "0", "dy": "0", "dz": None},
                {"px": float(img_shape[1]), "py": float(img_shape[0]), "dx": "100", "dy": "0", "dz": None},
                {"px": 0.0, "py": float(img_shape[0]), "dx": "0", "dy": "0", "dz": None},
                {"px": 0.0, "py": 0.0, "dx": "0", "dy": "100", "dz": None}
            ]
            is_log_x = False; is_log_y = False
        else:
            calibration_points = [
                {"px": axis_config["x1_px"], "py": axis_config["x1_py"], "dx": str(axis_config["x1_val"]), "dy": str(axis_config["y1_val"]), "dz": None},
                {"px": axis_config["x2_px"], "py": axis_config["x2_py"], "dx": str(axis_config["x2_val"]), "dy": str(axis_config["y1_val"]), "dz": None},
                {"px": axis_config["y1_px"], "py": axis_config["y1_py"], "dx": str(axis_config["x1_val"]), "dy": str(axis_config["y1_val"]), "dz": None},
                {"px": axis_config["y2_px"], "py": axis_config["y2_py"], "dx": str(axis_config["x1_val"]), "dy": str(axis_config["y2_val"]), "dz": None}
            ]
            is_log_x = axis_config.get("xIsLogScale", False)
            is_log_y = axis_config.get("yIsLogScale", False)
        axes_coll = [{"name": "XY", "type": "XYAxes", "isLogX": is_log_x, "isLogY": is_log_y,
                      "noRotation": False, "calibrationPoints": calibration_points}]
        dataset_coll = [{"name": "Default Dataset", "axesName": "XY", "colorRGB": [200, 0, 0, 255],
                         "metadataKeys": [], "data": [], "autoDetectionData": None}]
        for idx, series in enumerate(data_series):
            data_points = [{"x": float(pt[0]), "y": float(pt[1]), "value": None} for pt in series["points"]]
            dataset_coll.append({"name": f"Dataset {idx + 1}", "axesName": "XY",
                                 "colorRGB": [200, 0, 0, 255], "metadataKeys": [],
                                 "data": data_points, "autoDetectionData": None})
        return {"version": [4, 2], "axesColl": axes_coll, "datasetColl": dataset_coll, "measurementColl": []}

    def create_starry_digitizer_zip(self, img, project_json):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            _, img_encoded = cv2.imencode('.png', img)
            zf.writestr('image.png', img_encoded.tobytes())
            zf.writestr('project.json', json.dumps(project_json, indent=2, ensure_ascii=False).encode('utf-8'))
        zip_buffer.seek(0)
        return zip_buffer

    def create_wpd_tar(self, img, wpd_json, project_name="project"):
        import time
        tar_buffer = io.BytesIO()
        mtime = time.time()
        with tarfile.open(fileobj=tar_buffer, mode='w') as tf:
            folder_info = tarfile.TarInfo(name=f'{project_name}/')
            folder_info.type = tarfile.DIRTYPE
            folder_info.mtime = mtime
            tf.addfile(folder_info)
            info_json = {"version": [4, 0], "json": "wpd.json", "images": ["image.png"]}
            info_bytes = json.dumps(info_json, ensure_ascii=False).encode('utf-8')
            info_tarinfo = tarfile.TarInfo(name=f'{project_name}/info.json')
            info_tarinfo.size = len(info_bytes); info_tarinfo.mtime = mtime
            tf.addfile(info_tarinfo, io.BytesIO(info_bytes))
            wpd_bytes = json.dumps(wpd_json, ensure_ascii=False).encode('utf-8')
            wpd_tarinfo = tarfile.TarInfo(name=f'{project_name}/wpd.json')
            wpd_tarinfo.size = len(wpd_bytes); wpd_tarinfo.mtime = mtime
            tf.addfile(wpd_tarinfo, io.BytesIO(wpd_bytes))
            _, img_encoded = cv2.imencode('.png', img)
            img_bytes = img_encoded.tobytes()
            img_tarinfo = tarfile.TarInfo(name=f'{project_name}/image.png')
            img_tarinfo.size = len(img_bytes); img_tarinfo.mtime = mtime
            tf.addfile(img_tarinfo, io.BytesIO(img_bytes))
        tar_buffer.seek(0)
        return tar_buffer


def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def main(page: ft.Page):
    # ---- Editorial palette (mirrors the KMDS paper viewer) ----
    ACCENT = "#B5402F"      # terracotta — primary actions / accents
    ACCENT_2 = "#1B4D3E"    # deep green — secondary
    GOLD = "#C08A2D"
    PAPER = "#F7F2E9"       # warm page background
    PAPER_2 = "#EFE7D6"     # subtle panel tint
    SURFACE = "#FFFFFF"     # cards
    INK = "#2A2622"
    INK_3 = "#6B6258"
    RULE = "#D8CDB8"

    page.title = "AutoLineDigitizer"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.theme = ft.Theme(color_scheme_seed=ACCENT, use_material3=True)
    page.bgcolor = PAPER
    page.window.width = 1340
    page.window.height = 920
    page.window.min_width = 1040
    page.window.min_height = 680
    page.padding = 0

    def _section_label(txt):
        return ft.Text(txt.upper(), size=11, weight=ft.FontWeight.BOLD, color=ACCENT)

    def _soft_divider():
        return ft.Divider(height=14, thickness=1, color=RULE)

    app = LineFormerApp()

    CANVAS_W = 460  # fixed display width of each image; click->pixel math depends on it
    MAX_TABLE_ROWS = 40  # display cap; CSV export gets full data

    # ---- UI components ----
    status_text = ft.Text("Loading...", size=12, no_wrap=False, expand=True)
    progress_ring = ft.ProgressRing(visible=True, width=16, height=16)
    axis_status_text = ft.Text("Loading...", size=12, no_wrap=False)

    input_image = ft.Image(visible=False, width=CANVAS_W, fit=ft.ImageFit.FIT_WIDTH)
    result_image = ft.Image(visible=False, width=CANVAS_W, fit=ft.ImageFit.FIT_WIDTH)

    # ---- PDF figure gallery (populated when a PDF is opened) ----
    THUMB_W = 150
    pdf_gallery_title = ft.Text("", size=13, weight=ft.FontWeight.BOLD)
    pdf_gallery_hint = ft.Text("Click a figure to extract and edit it.",
                               size=11, color=ft.colors.GREY_600)
    pdf_gallery_row = ft.Row(spacing=8, scroll=ft.ScrollMode.AUTO, wrap=False)
    pdf_gallery = ft.Container(
        content=ft.Column([
            pdf_gallery_title, pdf_gallery_hint,
            ft.Container(content=pdf_gallery_row, height=THUMB_W + 30,
                         padding=6, border=ft.border.all(1, RULE),
                         border_radius=10, bgcolor="#FFFFFF"),
        ], spacing=6),
        visible=False, padding=14, bgcolor=SURFACE, border_radius=14,
        border=ft.border.all(1, RULE),
    )

    # Recrop button (enabled only for figures sourced from a PDF page).
    recrop_btn = ft.OutlinedButton(
        "Recrop", icon=ft.icons.CROP, disabled=True,
        tooltip="Re-crop this figure from its source PDF page (drag a new box). "
                "Use when the auto-crop is too tight or too loose.",
    )

    # PDF extraction strategy. AI page detection (render every page + Claude
    # bbox detection) is the high-quality default; uncheck for fast offline
    # raster-bitmap extraction (no API key, but misses vector figures).
    _vlm_ok = VLM_SCREENER_AVAILABLE and VLM_SCREENER_SDK_AVAILABLE
    _detector_opts = []
    if MINERU_OK:
        _detector_opts.append(ft.dropdown.Option("mineru", "MinerU PP-DocLayoutV2 (best)"))
    if DOCLAYOUT_OK:
        _detector_opts.append(ft.dropdown.Option("doclayout", "DocLayout-YOLO (offline)"))
    if _vlm_ok:
        _detector_opts.append(ft.dropdown.Option("vlm", "Claude VLM (per-panel)"))
    _detector_opts.append(ft.dropdown.Option("raster", "Embedded raster"))
    _default_detector = ("mineru" if MINERU_OK else
                         ("doclayout" if DOCLAYOUT_OK else ("vlm" if _vlm_ok else "raster")))
    pdf_detector_dropdown = ft.Dropdown(
        label="Figure detector", width=230,
        value=_default_detector,
        options=_detector_opts,
        tooltip="How figures are located in the PDF. MinerU PP-DocLayoutV2: best — "
                "separates charts from photos and splits composite panels (offline). "
                "DocLayout-YOLO: whole figures (offline). Claude VLM: costs API. "
                "Raster: embedded bitmaps only.",
    )

    # Re-run the VLM detection + refinement agent on the current PDF.
    review_figures_btn = ft.OutlinedButton(
        "AI Review Figures", icon=ft.icons.AUTO_FIX_HIGH, disabled=True,
        tooltip="Re-run AI detection and the box-refinement agent on the open PDF: "
                "tightens crops, drops non-charts, and splits merged panels. "
                "Rebuilds the figure gallery."
                if (VLM_SCREENER_AVAILABLE and VLM_SCREENER_SDK_AVAILABLE)
                else "Needs ANTHROPIC_API_KEY + anthropic SDK.",
    )

    # KMDS structured-metadata extraction (whole-PDF -> EN + JA JSON).
    kmds_btn = ft.OutlinedButton(
        "Extract Metadata (KMDS)", icon=ft.icons.DATASET, disabled=True,
        tooltip="Run the parallel KMDS extraction on the open PDF: bibliography, "
                "materials, process, properties, figures -> structured EN + JA JSON. "
                "Takes ~60-90s and uses the Anthropic API."
                if KMDS_AVAILABLE else "Needs ANTHROPIC_API_KEY + anthropic SDK.",
    )

    # Paper-record handoff to the KMDS viewer.
    save_fig_btn = ft.OutlinedButton(
        "Save figure → record", icon=ft.icons.PLAYLIST_ADD,
        tooltip="Stage the currently digitized figure (calibrated points + axes) "
                "into the paper record.",
    )
    open_record_btn = ft.OutlinedButton(
        "Open Paper Record", icon=ft.icons.MENU_BOOK, disabled=True,
        tooltip="Build the KMDS paper record with your staged figure digitizations "
                "and open it in the paper viewer (browser). Enabled once KMDS is "
                "extracted or a figure is staged.",
    )

    # KMDS results render INLINE (not a dialog) — updating inline controls from
    # a worker thread is the pattern that already works elsewhere in this app.
    kmds_paths = {"en": None, "ja": None, "dir": None}
    kmds_title = ft.Text("KMDS metadata", size=16, weight=ft.FontWeight.BOLD)
    kmds_summary_text = ft.Text("", size=12, selectable=True)
    kmds_en_btn = ft.TextButton("English")
    kmds_ja_btn = ft.TextButton("日本語")
    kmds_openfolder_btn = ft.TextButton("Open folder", icon=ft.icons.FOLDER_OPEN)
    kmds_close_btn = ft.TextButton("Hide", icon=ft.icons.CLOSE)
    kmds_json_view = ft.TextField(
        value="", read_only=True, multiline=True, min_lines=16, max_lines=22,
        text_size=11,
    )
    kmds_panel = ft.Container(
        content=ft.Column([
            ft.Row([kmds_title, ft.Container(expand=True),
                    kmds_en_btn, kmds_ja_btn, kmds_openfolder_btn, kmds_close_btn]),
            kmds_summary_text,
            kmds_json_view,
        ], spacing=6),
        visible=False, padding=14,
        border=ft.border.all(1, RULE), border_radius=14,
        bgcolor=SURFACE,
    )

    info_text = ft.Text("", size=14)
    axis_info_text = ft.Text("", size=12, color=ft.colors.GREY_700)

    axis_model_dropdown = ft.Dropdown(
        label="Axis Model", value="chartdete", width=200,
        options=[ft.dropdown.Option("chartdete", "ChartDete")],
    )

    model_dropdown = ft.Dropdown(
        label="Line Model", value="general", width=200,
        options=[ft.dropdown.Option(key, info["name"]) for key, info in LINEFORMER_MODELS.items()],
    )

    sort_dropdown = ft.Dropdown(
        label="Sort Lines", value="mean_y_desc", width=200,
        options=[
            ft.dropdown.Option("original", "Detection Order"),
            ft.dropdown.Option("mean_y_desc", "Mean Y (High → Low)"),
            ft.dropdown.Option("mean_y_asc", "Mean Y (Low → High)"),
        ],
    )

    downsample_dropdown = ft.Dropdown(
        label="Downsampling", value="max_points", width=200,
        options=[
            ft.dropdown.Option("max_points", "Max Points"),
            ft.dropdown.Option("arc_length", "Arc Length"),
            ft.dropdown.Option("fixed", "Fixed Step"),
            ft.dropdown.Option("none", "None"),
        ],
    )

    max_points_label = ft.Text("Max points: 20", size=12)
    max_points_slider = ft.Slider(min=10, max=100, value=20, divisions=9, label="{value}", width=200)
    fixed_step_label = ft.Text("Step: 10", size=12, visible=False)
    fixed_step_slider = ft.Slider(min=1, max=50, value=10, divisions=49, label="{value}", width=200, visible=False)

    color_refine_checkbox = ft.Checkbox(
        label="Color refinement", value=True,
        tooltip="Re-trace each line by its color. Resolves crossings.",
    )

    detected_lines_title = ft.Text("Detected Lines", size=14, weight=ft.FontWeight.BOLD, visible=False)
    detected_lines_hint = ft.Text("Click a line to edit it", size=11, color=ft.colors.GREY_600, visible=False)
    detected_lines_column = ft.Column(spacing=2, tight=True)

    # ---- Manual editing panel ----
    edit_status = ft.Text("", size=12, weight=ft.FontWeight.W_500)
    erase_btn = ft.OutlinedButton("Erase", icon=ft.icons.CLEAR)
    add_btn = ft.OutlinedButton("Add", icon=ft.icons.TIMELINE)
    eraser_label = ft.Text("Eraser radius: 18", size=11, visible=False)
    eraser_slider = ft.Slider(min=6, max=40, value=18, divisions=34, label="{value}", width=200, visible=False)
    add_hint = ft.Text("", size=11, color=ft.colors.GREY_600, visible=False)
    apply_btn = ft.OutlinedButton("Apply spline", icon=ft.icons.CHECK, visible=False)
    undo_anchor_btn = ft.TextButton("Undo last point", visible=False)
    delete_line_btn = ft.OutlinedButton(
        "Delete Line", icon=ft.icons.DELETE_OUTLINE,
        style=ft.ButtonStyle(color=ft.colors.RED_700),
    )
    done_btn = ft.OutlinedButton("Done editing", icon=ft.icons.DONE_ALL)
    edit_panel = ft.Column([
        edit_status,
        ft.Row([erase_btn, add_btn], spacing=6),
        eraser_label, eraser_slider,
        add_hint,
        ft.Row([apply_btn, undo_anchor_btn], spacing=6),
        ft.Divider(height=4, color=ft.colors.GREY_300),
        delete_line_btn,
        done_btn,
    ], spacing=6, visible=False)

    # ---- Data table section (NEW) ----
    data_table_title = ft.Text("Data Table", size=16, weight=ft.FontWeight.BOLD)
    x_axis_name_field = ft.TextField(label="X axis name", value="", width=200, dense=True,
                                     hint_text="e.g. Cycle number")
    y_axis_name_field = ft.TextField(label="Y axis name", value="", width=200, dense=True,
                                     hint_text="e.g. Voltage (V)")
    table_line_picker = ft.Dropdown(
        label="Show", value="all", width=180,
        options=[ft.dropdown.Option("all", "All lines (combined)")],
    )
    export_csv_btn = ft.OutlinedButton("Export CSV", icon=ft.icons.TABLE_VIEW, disabled=True)
    data_table = ft.DataTable(
        columns=[ft.DataColumn(ft.Text("(load an image to see data)"))],
        rows=[],
        column_spacing=18, heading_row_height=36,
        data_row_min_height=28, data_row_max_height=32,
        divider_thickness=0.5,
    )
    data_table_scroll = ft.Container(
        content=ft.Row([data_table], scroll=ft.ScrollMode.AUTO),
        height=320, padding=4,
        border=ft.border.all(1, ft.colors.GREY_300),
        border_radius=6,
    )
    table_row_count_text = ft.Text("", size=11, color=ft.colors.GREY_600)

    def _bgr_to_hex(bgr):
        b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
        return f"#{r:02x}{g:02x}{b:02x}"

    def _current_axis_names():
        x = (x_axis_name_field.value or "").strip() or "X"
        y = (y_axis_name_field.value or "").strip() or "Y"
        return x, y

    def redraw_with_highlight(highlight_idx):
        if app.current_image is None or app.data_series is None:
            return
        img = app.draw_points_on_image(app.current_image, app.data_series, app.axis_config,
                                       highlight_line_idx=highlight_idx)
        result_image.src_base64 = image_to_base64(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        page.update()

    def redraw_canvas():
        if app.current_image is None or app.data_series is None:
            return
        img = app.draw_points_on_image(app.current_image, app.data_series, app.axis_config,
                                       highlight_line_idx=app.selected_line_idx)
        if app.edit_mode == "add" and app.add_anchors:
            for (axp, ayp) in app.add_anchors:
                cv2.circle(img, (int(axp), int(ayp)), 5, (0, 0, 255), -1)
                cv2.circle(img, (int(axp), int(ayp)), 6, (255, 255, 255), 1)
            if len(app.add_anchors) >= 2:
                preview = app.spline_interpolate(app.add_anchors)
                for j in range(len(preview) - 1):
                    cv2.line(img, tuple(preview[j]), tuple(preview[j + 1]), (0, 0, 255), 2)
        result_image.src_base64 = image_to_base64(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result_image.visible = True
        page.update()

    def update_data_table():
        x_name, y_name = _current_axis_names()
        n_lines = len(app.data_series) if app.data_series else 0

        # Refresh line picker options
        options = [ft.dropdown.Option("all", "All lines (combined)")]
        for i in range(n_lines):
            options.append(ft.dropdown.Option(str(i), f"Line {i+1}"))
        table_line_picker.options = options
        valid_keys = {opt.key for opt in options}
        if table_line_picker.value not in valid_keys:
            table_line_picker.value = "all"

        if not app.data_series:
            data_table.columns = [ft.DataColumn(ft.Text("(load an image to see data)"))]
            data_table.rows = []
            table_row_count_text.value = ""
            export_csv_btn.disabled = True
            page.update()
            return

        export_csv_btn.disabled = False
        pick = table_line_picker.value or "all"

        if pick == "all":
            cols = [ft.DataColumn(ft.Text("#"))]
            for i in range(n_lines):
                cols.append(ft.DataColumn(ft.Text(f"{app.line_name(i)} {x_name}")))
                cols.append(ft.DataColumn(ft.Text(f"{app.line_name(i)} {y_name}")))
            data_table.columns = cols
            max_len = max((len(s["points"]) for s in app.data_series), default=0)
            shown = min(max_len, MAX_TABLE_ROWS)
            new_rows = []
            for r in range(shown):
                cells = [ft.DataCell(ft.Text(str(r + 1), size=11))]
                for s in app.data_series:
                    if r < len(s["points"]):
                        px, py = s["points"][r]
                        xv, yv = app.pixel_to_data(px, py)
                        cells.append(ft.DataCell(ft.Text(f"{xv:.4g}", size=11)))
                        cells.append(ft.DataCell(ft.Text(f"{yv:.4g}", size=11)))
                    else:
                        cells.append(ft.DataCell(ft.Text("", size=11)))
                        cells.append(ft.DataCell(ft.Text("", size=11)))
                new_rows.append(ft.DataRow(cells))
            data_table.rows = new_rows
            extra = max_len - shown
            if extra > 0:
                table_row_count_text.value = f"Showing first {shown} of {max_len} rows. Export CSV for full data."
            else:
                table_row_count_text.value = f"{max_len} rows."
        else:
            try:
                idx = int(pick)
            except Exception:
                idx = 0
            idx = max(0, min(idx, n_lines - 1))
            data_table.columns = [
                ft.DataColumn(ft.Text("#")),
                ft.DataColumn(ft.Text(x_name)),
                ft.DataColumn(ft.Text(y_name)),
            ]
            pts = app.data_series[idx].get("points", [])
            shown = min(len(pts), MAX_TABLE_ROWS)
            new_rows = []
            for r in range(shown):
                px, py = pts[r]
                xv, yv = app.pixel_to_data(px, py)
                new_rows.append(ft.DataRow([
                    ft.DataCell(ft.Text(str(r + 1), size=11)),
                    ft.DataCell(ft.Text(f"{xv:.4g}", size=11)),
                    ft.DataCell(ft.Text(f"{yv:.4g}", size=11)),
                ]))
            data_table.rows = new_rows
            extra = len(pts) - shown
            if extra > 0:
                table_row_count_text.value = f"Line {idx+1}: showing first {shown} of {len(pts)} rows. Export CSV for full data."
            else:
                table_row_count_text.value = f"Line {idx+1}: {len(pts)} rows."

        page.update()

    def refresh_after_edit():
        if app.data_series is None:
            return
        app.raw_lines = [list(s["points"]) for s in app.data_series]
        total = sum(len(s["points"]) for s in app.data_series)
        line_pts = [len(s["points"]) for s in app.data_series]
        info_text.value = (f"{len(app.data_series)} lines ({total} points)\n"
                           f"Points per line: {', '.join(map(str, line_pts))}")
        populate_detected_lines()
        update_data_table()
        redraw_canvas()

    def _hide_edit_subcontrols():
        eraser_label.visible = False
        eraser_slider.visible = False
        apply_btn.visible = False
        undo_anchor_btn.visible = False
        add_hint.visible = False

    def set_mode(mode):
        app.edit_mode = mode
        app.add_anchors = []
        _hide_edit_subcontrols()
        if mode == "erase":
            eraser_label.visible = True
            eraser_slider.visible = True
        elif mode == "add":
            apply_btn.visible = True
            undo_anchor_btn.visible = True
            add_hint.visible = True
            add_hint.value = "Click 2+ points along the curve, then Apply spline."
        if app.selected_line_idx is not None:
            edit_status.value = f"Editing Line {app.selected_line_idx + 1} — {mode} tool"
        page.update()
        redraw_canvas()

    def select_line(idx):
        if app.selected_line_idx == idx:
            on_done_click(None)
            return
        app.selected_line_idx = idx
        app.edit_mode = None
        app.add_anchors = []
        edit_panel.visible = True
        _hide_edit_subcontrols()
        edit_status.value = f"Editing Line {idx + 1} — pick a tool"
        populate_detected_lines()
        redraw_canvas()

    def on_erase_click(_):
        set_mode("erase")

    def on_add_click(_):
        set_mode("add")

    def on_apply_click(_):
        if app.selected_line_idx is not None and len(app.add_anchors) >= 2:
            app.add_spline_to_line(app.selected_line_idx, app.add_anchors)
            app.add_anchors = []
            refresh_after_edit()
            add_hint.value = "Section added. Click more points, or press Done editing."
            page.update()
        else:
            add_hint.value = "Need at least 2 points to make a spline."
            page.update()

    def on_undo_anchor_click(_):
        if app.add_anchors:
            app.add_anchors.pop()
            redraw_canvas()

    def on_done_click(_):
        app.selected_line_idx = None
        app.edit_mode = None
        app.add_anchors = []
        edit_panel.visible = False
        _hide_edit_subcontrols()
        populate_detected_lines()
        update_data_table()
        redraw_canvas()

    def on_eraser_change(e):
        app.eraser_radius = int(eraser_slider.value)
        eraser_label.value = f"Eraser radius: {int(eraser_slider.value)}"
        page.update()

    # ---- Delete Line: confirm dialog ----
    confirm_delete_dialog = ft.AlertDialog(modal=True, title=ft.Text("Delete this line?"))

    def _close_dialog():
        confirm_delete_dialog.open = False
        page.update()

    def _confirm_delete_yes(_):
        idx = app.selected_line_idx
        _close_dialog()
        if idx is None:
            return
        if app.delete_line(idx):
            app.selected_line_idx = None
            app.edit_mode = None
            app.add_anchors = []
            edit_panel.visible = False
            _hide_edit_subcontrols()
            total = sum(len(s["points"]) for s in app.data_series)
            line_pts = [len(s["points"]) for s in app.data_series]
            info_text.value = (f"{len(app.data_series)} lines ({total} points)\n"
                               f"Points per line: {', '.join(map(str, line_pts))}")
            populate_detected_lines()
            update_data_table()
            redraw_canvas()
            process_status_text.value = f"Deleted Line {idx + 1}."
            page.update()

    def on_delete_line_click(_):
        if app.selected_line_idx is None or not app.data_series:
            return
        idx = app.selected_line_idx
        n_pts = len(app.data_series[idx].get("points", []))
        confirm_delete_dialog.content = ft.Text(
            f"Line {idx + 1} has {n_pts} points. This cannot be undone "
            f"(re-load the image or re-run detection to reset)."
        )
        confirm_delete_dialog.actions = [
            ft.TextButton("Cancel", on_click=lambda e: _close_dialog()),
            ft.TextButton("Delete", on_click=_confirm_delete_yes,
                          style=ft.ButtonStyle(color=ft.colors.RED_700)),
        ]
        page.dialog = confirm_delete_dialog
        confirm_delete_dialog.open = True
        page.update()

    erase_btn.on_click = on_erase_click
    add_btn.on_click = on_add_click
    apply_btn.on_click = on_apply_click
    undo_anchor_btn.on_click = on_undo_anchor_click
    done_btn.on_click = on_done_click
    delete_line_btn.on_click = on_delete_line_click
    eraser_slider.on_change = on_eraser_change

    def populate_detected_lines():
        import line_utils
        detected_lines_column.controls.clear()
        if not app.data_series:
            detected_lines_title.visible = False
            detected_lines_hint.visible = False
            edit_panel.visible = False
            page.update()
            return
        colors = list(line_utils.get_distinct_colors(len(app.data_series)))
        for idx, series in enumerate(app.data_series):
            swatch = ft.Container(width=14, height=14,
                                  bgcolor=_bgr_to_hex(colors[idx]),
                                  border_radius=3,
                                  border=ft.border.all(1, ft.colors.GREY_400))
            label = ft.Text(f"{app.line_name(idx)} ({len(series['points'])} pts)", size=12)
            selected = (idx == app.selected_line_idx)

            def on_hover(e, _idx=idx):
                if app.selected_line_idx is None:
                    redraw_with_highlight(_idx if e.data == "true" else None)

            def on_click_row(e, _idx=idx):
                select_line(_idx)

            row = ft.Container(
                content=ft.Row([swatch, label], spacing=8, tight=True),
                padding=ft.padding.symmetric(horizontal=6, vertical=4),
                border_radius=4,
                bgcolor=ft.colors.BLUE_50 if selected else None,
                on_hover=on_hover, on_click=on_click_row, ink=True,
            )
            detected_lines_column.controls.append(row)
        detected_lines_title.visible = True
        detected_lines_hint.visible = True
        page.update()

    def reprocess_lines():
        if app.current_image is None or app.raw_lines is None:
            return
        app.selected_line_idx = None
        app.edit_mode = None
        app.add_anchors = []
        edit_panel.visible = False
        _hide_edit_subcontrols()
        app.data_series = app.apply_downsample_and_sort()
        app.result_image = app.draw_points_on_image(app.current_image, app.data_series, app.axis_config)
        result_image.src_base64 = image_to_base64(cv2.cvtColor(app.result_image, cv2.COLOR_BGR2RGB))
        result_image.visible = True
        total_points = sum(len(s['points']) for s in app.data_series)
        line_pts = [len(s['points']) for s in app.data_series]
        info_text.value = f"{len(app.data_series)} lines detected ({total_points} points)\nPoints per line: {', '.join(map(str, line_pts))}"
        populate_detected_lines()
        update_data_table()
        page.update()

    def on_downsample_change(e):
        app.downsample_mode = downsample_dropdown.value
        show_max_points = downsample_dropdown.value in ("max_points", "arc_length")
        max_points_label.visible = show_max_points
        max_points_slider.visible = show_max_points
        fixed_step_label.visible = downsample_dropdown.value == "fixed"
        fixed_step_slider.visible = downsample_dropdown.value == "fixed"
        page.update()
        reprocess_lines()

    def on_sort_change(e):
        app.sort_mode = sort_dropdown.value
        reprocess_lines()

    def on_max_points_change(e):
        app.max_points = int(max_points_slider.value)
        max_points_label.value = f"Max points: {int(max_points_slider.value)}"
        page.update()
        reprocess_lines()

    def on_fixed_step_change(e):
        app.fixed_step = int(fixed_step_slider.value)
        fixed_step_label.value = f"Step: {int(fixed_step_slider.value)}"
        page.update()
        reprocess_lines()

    def on_color_refine_change(e):
        app.use_color_refinement = color_refine_checkbox.value
        app.raw_lines = None
        if app.current_image is not None and app.infer_module is not None:
            page.run_thread(lambda: process_image(skip_axis=app.axis_config is not None))

    def on_axis_name_change(e):
        app.x_axis_name = (x_axis_name_field.value or "").strip()
        app.y_axis_name = (y_axis_name_field.value or "").strip()
        update_data_table()

    def on_table_pick_change(e):
        update_data_table()

    x_axis_name_field.on_change = on_axis_name_change
    y_axis_name_field.on_change = on_axis_name_change
    table_line_picker.on_change = on_table_pick_change

    # ---- Canvas gesture handlers ----
    def to_pixel_coords(local_x, local_y):
        if app.current_image is None:
            return None
        oh, ow = app.current_image.shape[:2]
        if CANVAS_W <= 0:
            return None
        scale = ow / float(CANVAS_W)
        return local_x * scale, local_y * scale

    def on_canvas_tap(e):
        if app.selected_line_idx is None or app.edit_mode is None:
            return
        try:
            lx, ly = e.local_x, e.local_y
        except Exception:
            return
        c = to_pixel_coords(lx, ly)
        if c is None:
            return
        px, py = c
        if app.edit_mode == "erase":
            app.erase_near(app.selected_line_idx, px, py, app.eraser_radius)
            refresh_after_edit()
        elif app.edit_mode == "add":
            app.add_anchors.append([px, py])
            redraw_canvas()

    def on_canvas_pan(e):
        if app.selected_line_idx is None or app.edit_mode != "erase":
            return
        try:
            lx, ly = e.local_x, e.local_y
        except Exception:
            return
        c = to_pixel_coords(lx, ly)
        if c is None:
            return
        px, py = c
        removed = app.erase_near(app.selected_line_idx, px, py, app.eraser_radius)
        if removed:
            redraw_canvas()

    sort_dropdown.on_change = on_sort_change
    downsample_dropdown.on_change = on_downsample_change
    max_points_slider.on_change = on_max_points_change
    fixed_step_slider.on_change = on_fixed_step_change
    color_refine_checkbox.on_change = on_color_refine_change

    export_sd_btn = None
    export_wpd_btn = None
    verify_btn = None
    detect_markers_btn = None

    def process_image(skip_axis=False):
        nonlocal export_sd_btn, export_wpd_btn, verify_btn, detect_markers_btn
        if app.current_image is None or app.infer_module is None:
            return
        app.selected_line_idx = None
        app.edit_mode = None
        app.add_anchors = []
        edit_panel.visible = False
        _hide_edit_subcontrols()
        process_status_text.value = "Extracting lines..."
        process_progress_ring.visible = True
        page.update()
        try:
            app.data_series = app.extract_lines(app.current_image)
            app.result_image = app.draw_points_on_image(app.current_image, app.data_series, app.axis_config)
            result_image.src_base64 = image_to_base64(cv2.cvtColor(app.result_image, cv2.COLOR_BGR2RGB))
            result_image.visible = True
            total_points = sum(len(s['points']) for s in app.data_series)
            line_pts = [len(s['points']) for s in app.data_series]
            info_text.value = f"{len(app.data_series)} lines detected ({total_points} points)\nPoints per line: {', '.join(map(str, line_pts))}"
            populate_detected_lines()
            update_data_table()
            page.update()

            if not skip_axis:
                app.axis_config = None
                app.ocr_results = None
                if app.auto_axis and app.chartdete_module is not None:
                    process_status_text.value = "Detecting axis labels..."
                    page.update()
                    app.axis_config, app.ocr_results = app.detect_axis_calibration(app.current_image)
                    app.result_image = app.draw_points_on_image(app.current_image, app.data_series, app.axis_config)
                    result_image.src_base64 = image_to_base64(cv2.cvtColor(app.result_image, cv2.COLOR_BGR2RGB))
                    # Auto-fill axis names from OCR
                    x_name, y_name = app.get_axis_titles()
                    if x_name and not (x_axis_name_field.value or "").strip():
                        x_axis_name_field.value = x_name
                        app.x_axis_name = x_name
                    if y_name and not (y_axis_name_field.value or "").strip():
                        y_axis_name_field.value = y_name
                        app.y_axis_name = y_name

            if app.axis_config is not None:
                axis_info_text.value = f"Axis: X=[{app.axis_config['x1_val']} → {app.axis_config['x2_val']}], Y=[{app.axis_config['y1_val']} → {app.axis_config['y2_val']}]"
            elif app.auto_axis:
                axis_info_text.value = "Axis detection failed. Manual calibration needed."
            else:
                axis_info_text.value = ""

            update_data_table()

            process_status_text.value = ""
            process_progress_ring.visible = False
            export_sd_btn.disabled = False
            export_wpd_btn.disabled = False
            verify_btn.disabled = not (VLM_VERIFIER_AVAILABLE and ANTHROPIC_AVAILABLE)
            detect_markers_btn.disabled = not MARKER_DETECTOR_AVAILABLE
            axis_fix_btn.disabled = not (VLM_VERIFIER_AVAILABLE and ANTHROPIC_AVAILABLE)
            label_lines_btn.disabled = not (VLM_VERIFIER_AVAILABLE and ANTHROPIC_AVAILABLE)
        except Exception as e:
            process_status_text.value = f"Error: {e}"
            process_progress_ring.visible = False
        page.update()

    def load_image(img, image_path=None, figure_idx=None):
        app.current_image = img
        app.current_image_path = image_path
        app.cached_plot_area = None
        app.current_figure_idx = figure_idx
        # Recrop only makes sense for a figure that came from the current PDF.
        recrop_btn.disabled = not (figure_idx is not None and app.pdf_path
                                   and PDF_SUPPORT)
        input_image.src_base64 = image_to_base64(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        input_image.visible = True
        page.update()
        page.run_thread(process_image)

    def pick_files_result(e: ft.FilePickerResultEvent):
        if e.files and len(e.files) > 0:
            file_path = e.files[0].path
            img = cv2.imread(file_path)
            if img is not None:
                load_image(img, file_path)
            else:
                process_status_text.value = "Failed to read image"
                page.update()

    def on_keyboard(e: ft.KeyboardEvent):
        if e.key == "V" and (e.meta or e.ctrl):
            try:
                from PIL import ImageGrab
                pil_img = ImageGrab.grabclipboard()
                if pil_img is not None:
                    img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
                    load_image(img)
            except Exception:
                pass

    page.on_keyboard_event = on_keyboard
    file_picker = ft.FilePicker(on_result=pick_files_result)
    page.overlay.append(file_picker)

    # ---- PDF -> figure gallery ----
    def make_thumb_b64(img_bgr, max_w=THUMB_W):
        h, w = img_bgr.shape[:2]
        scale = max_w / float(max(1, w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        thumb = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image_to_base64(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))

    def on_thumbnail_click(idx):
        if not (0 <= idx < len(app.pdf_figures)):
            return
        img_bgr, meta = app.pdf_figures[idx]
        process_status_text.value = (
            f"Loading figure {idx + 1} (page {meta.get('page', '?')})..."
        )
        # Reset axis names so the new figure's OCR can repopulate them.
        x_axis_name_field.value = ""
        y_axis_name_field.value = ""
        app.x_axis_name = ""
        app.y_axis_name = ""
        build_gallery(selected_idx=idx)
        page.update()
        load_image(img_bgr.copy(), image_path=app.pdf_path, figure_idx=idx)

    def build_gallery(selected_idx=None):
        pdf_gallery_row.controls.clear()
        n = len(app.pdf_figures)
        if n == 0:
            pdf_gallery.visible = False
            page.update()
            return
        pdf_name = os.path.basename(app.pdf_path) if app.pdf_path else "PDF"
        pdf_gallery_title.value = f"{n} figure(s) from {pdf_name}"
        for idx, (img_bgr, meta) in enumerate(app.pdf_figures):
            thumb_b64 = make_thumb_b64(img_bgr)
            selected = (idx == selected_idx)
            caption = (meta.get("caption") or "").strip()
            label = caption[:22] if caption else f"p{meta.get('page', '?')} #{meta.get('img_idx_on_page', idx + 1)}"
            tile = ft.Container(
                content=ft.Column([
                    ft.Image(src_base64=thumb_b64, width=THUMB_W,
                             fit=ft.ImageFit.FIT_WIDTH),
                    ft.Text(label, size=10, no_wrap=True,
                            color=ft.colors.GREY_700),
                ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER, tight=True),
                padding=4, border_radius=6,
                bgcolor=ft.colors.BLUE_50 if selected else None,
                border=ft.border.all(2, ft.colors.BLUE_400 if selected
                                     else ft.colors.GREY_300),
                on_click=lambda e, _i=idx: on_thumbnail_click(_i),
                ink=True, tooltip=caption or label,
            )
            pdf_gallery_row.controls.append(tile)
        pdf_gallery.visible = True
        page.update()

    def run_pdf_extraction(pdf_path):
        process_progress_ring.visible = True
        process_status_text.value = "Scanning PDF for figures..."
        app.pdf_path = pdf_path
        app.pdf_figures = []
        app.current_figure_idx = None
        pdf_gallery.visible = False
        page.update()

        def _progress(count, phase):
            if phase == "mineru-start":
                process_status_text.value = "Locating charts with MinerU PP-DocLayoutV2 (offline)…"
            elif phase == "mineru":
                process_status_text.value = f"Found {count} chart(s) via PP-DocLayoutV2…"
            elif phase == "doclayout-start":
                process_status_text.value = "Locating figures with DocLayout-YOLO (offline)…"
            elif phase == "doclayout":
                process_status_text.value = f"Found {count} figure(s) via DocLayout-YOLO…"
            elif phase == "vlm-start":
                process_status_text.value = ("Rendering pages and asking Claude to "
                                             "find charts (this can take a moment)…")
            elif phase == "vlm":
                process_status_text.value = f"Found {count} chart(s) via AI page detection…"
            else:
                process_status_text.value = f"Found {count} embedded figure(s)…"
            page.update()

        detector = pdf_detector_dropdown.value or _default_detector
        prefer_vlm = (detector == "vlm") and app._vlm_screener_available()
        try:
            figs = app.extract_pdf_figures(pdf_path, prefer_vlm=prefer_vlm,
                                           detector=detector, progress=_progress)
        except Exception as ex:
            process_status_text.value = f"PDF scan failed: {ex}"
            process_progress_ring.visible = False
            page.update()
            return

        process_progress_ring.visible = False
        if not figs:
            process_status_text.value = (
                "No extractable figures found in this PDF "
                "(no embedded charts; VLM found none either)."
            )
            pdf_gallery.visible = False
            page.update()
            return

        process_status_text.value = (
            f"{len(figs)} figure(s) found. Click one to extract its data."
        )
        # A PDF is now open: enable whole-PDF KMDS metadata + AI re-review.
        kmds_btn.disabled = not (KMDS_AVAILABLE and str(app.pdf_path).lower().endswith(".pdf"))
        review_figures_btn.disabled = not (app._vlm_screener_available()
                                           and str(app.pdf_path).lower().endswith(".pdf"))
        build_gallery()
        # Auto-load the first figure so the user sees results immediately.
        on_thumbnail_click(0)

    def pick_pdf_result(e: ft.FilePickerResultEvent):
        if e.files and len(e.files) > 0:
            pdf_path = e.files[0].path
            if not PDF_SUPPORT:
                process_status_text.value = "PDF support unavailable (PyMuPDF not installed)."
                page.update()
                return
            page.run_thread(lambda: run_pdf_extraction(pdf_path))

    pdf_picker = ft.FilePicker(on_result=pick_pdf_result)
    page.overlay.append(pdf_picker)

    def on_review_figures_click(_):
        if not (app.pdf_path and str(app.pdf_path).lower().endswith(".pdf")
                and app._vlm_screener_available()):
            return
        # Re-run detection + the refinement agent (VLM path) and rebuild gallery.
        pdf_detector_dropdown.value = "vlm"
        page.run_thread(lambda: run_pdf_extraction(app.pdf_path))

    review_figures_btn.on_click = on_review_figures_click

    # ---- Recrop: adjust a figure's crop from its source PDF page ----
    recrop_state = {"page_img": None, "full_page": None, "zoom": 1.0,
                    "disp_scale": 1.0, "start": None, "box": None}
    RECROP_DISP_W = 760

    recrop_image = ft.Image(width=RECROP_DISP_W, fit=ft.ImageFit.FIT_WIDTH)
    recrop_status = ft.Text("Drag a rectangle over the chart, then Apply crop.",
                            size=12, color=ft.colors.GREY_700)
    recrop_dialog = ft.AlertDialog(modal=True, title=ft.Text("Recrop figure"))

    def _render_recrop_overlay():
        base = recrop_state["page_img"]
        if base is None:
            return
        disp = base.copy()
        box = recrop_state["box"]
        if box is not None:
            x0, y0, x1, y1 = box
            cv2.rectangle(disp, (int(x0), int(y0)), (int(x1), int(y1)),
                          (0, 0, 255), 2)
        recrop_image.src_base64 = image_to_base64(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        page.update()

    def on_recrop_pan_start(e):
        recrop_state["start"] = (e.local_x, e.local_y)
        recrop_state["box"] = None

    def on_recrop_pan_update(e):
        if recrop_state["start"] is None:
            return
        base = recrop_state["page_img"]
        if base is None:
            return
        H, W = base.shape[:2]
        sx, sy = recrop_state["start"]
        x0, x1 = sorted((sx, e.local_x))
        y0, y1 = sorted((sy, e.local_y))
        x0 = max(0, min(W - 1, x0)); x1 = max(0, min(W - 1, x1))
        y0 = max(0, min(H - 1, y0)); y1 = max(0, min(H - 1, y1))
        recrop_state["box"] = (x0, y0, x1, y1)
        _render_recrop_overlay()

    def _close_recrop():
        recrop_dialog.open = False
        page.update()

    def on_recrop_apply(_):
        full_page = recrop_state.get("full_page")
        box = recrop_state["box"]
        if full_page is None or box is None:
            recrop_status.value = "Draw a rectangle first."
            page.update()
            return
        # Box is in display coords; map back to the high-res page for the crop.
        ds = recrop_state.get("disp_scale", 1.0) or 1.0
        dx0, dy0, dx1, dy1 = box
        if (dx1 - dx0) < 10 or (dy1 - dy0) < 10:
            recrop_status.value = "Selection too small."
            page.update()
            return
        H, W = full_page.shape[:2]
        x0 = max(0, min(W, int(round(dx0 / ds))))
        y0 = max(0, min(H, int(round(dy0 / ds))))
        x1 = max(0, min(W, int(round(dx1 / ds))))
        y1 = max(0, min(H, int(round(dy1 / ds))))
        crop = full_page[y0:y1, x0:x1].copy()
        idx = app.current_figure_idx
        # Replace the figure in the gallery so the thumbnail updates too.
        if idx is not None and 0 <= idx < len(app.pdf_figures):
            _, meta = app.pdf_figures[idx]
            meta = dict(meta)
            meta["width"], meta["height"] = crop.shape[1], crop.shape[0]
            meta["recropped"] = True
            app.pdf_figures[idx] = (crop, meta)
            build_gallery(selected_idx=idx)
        _close_recrop()
        process_status_text.value = "Recropped. Re-extracting..."
        page.update()
        load_image(crop, image_path=app.pdf_path, figure_idx=idx)

    def on_recrop_click(_):
        idx = app.current_figure_idx
        if idx is None or not app.pdf_path or not PDF_SUPPORT:
            return
        if not (0 <= idx < len(app.pdf_figures)):
            return
        _, meta = app.pdf_figures[idx]
        page_no = meta.get("page")
        if not page_no:
            process_status_text.value = "Source page unknown; cannot recrop."
            page.update()
            return
        process_status_text.value = "Rendering source page..."
        page.update()

        def _open_recrop():
            # Render the page at high DPI for the *actual* crop quality, and
            # keep a separate downscaled copy just for the on-screen gesture.
            recrop_dpi = 200
            full_page, zoom = render_pdf_page(app.pdf_path, page_no, dpi=recrop_dpi)
            if full_page is None:
                process_status_text.value = "Could not render source page."
                page.update()
                return
            H, W = full_page.shape[:2]
            disp_scale = RECROP_DISP_W / float(W)
            disp_img = cv2.resize(full_page, (RECROP_DISP_W, max(1, int(H * disp_scale))),
                                  interpolation=cv2.INTER_AREA)
            recrop_state["full_page"] = full_page       # high-res, cropped from
            recrop_state["page_img"] = disp_img          # display + gesture overlay
            recrop_state["zoom"] = zoom
            recrop_state["disp_scale"] = disp_scale
            recrop_state["start"] = None
            # Pre-seed the box (in display coords) from the figure's known bbox.
            seed = None
            if meta.get("source") == "raster" and meta.get("bbox_pdf"):
                bx0, by0, bx1, by1 = meta["bbox_pdf"]
                seed = (bx0 * zoom * disp_scale, by0 * zoom * disp_scale,
                        bx1 * zoom * disp_scale, by1 * zoom * disp_scale)
            elif meta.get("source") == "vlm" and meta.get("bbox_px") and meta.get("render_dpi"):
                rdpi = meta["render_dpi"]
                k = (float(recrop_dpi) / rdpi) * disp_scale
                bx0, by0, bx1, by1 = meta["bbox_px"]
                seed = (bx0 * k, by0 * k, bx1 * k, by1 * k)
            recrop_state["box"] = seed
            _render_recrop_overlay()
            process_status_text.value = ""
            page.dialog = recrop_dialog
            recrop_dialog.open = True
            page.update()

        page.run_thread(_open_recrop)

    recrop_gesture = ft.GestureDetector(
        content=recrop_image,
        on_pan_start=on_recrop_pan_start,
        on_pan_update=on_recrop_pan_update,
        drag_interval=20,
        mouse_cursor=ft.MouseCursor.PRECISE,
    )
    recrop_dialog.content = ft.Column([
        recrop_status,
        ft.Container(content=recrop_gesture, alignment=ft.alignment.center),
    ], tight=True, scroll=ft.ScrollMode.AUTO, width=RECROP_DISP_W + 20, height=620)
    recrop_dialog.actions = [
        ft.TextButton("Cancel", on_click=lambda e: _close_recrop()),
        ft.ElevatedButton("Apply crop", icon=ft.icons.CHECK, on_click=on_recrop_apply),
    ]
    recrop_btn.on_click = on_recrop_click

    # ---- KMDS metadata extraction (whole PDF -> EN + JA structured JSON) ----
    def _kmds_load_json(path, fallback):
        try:
            with open(path, "r", encoding="utf-8") as f:
                kmds_json_view.value = f.read()
        except Exception:
            kmds_json_view.value = fallback
        page.update()

    def _on_kmds_show_en(_):
        if kmds_paths["en"]:
            kmds_title.value = "KMDS metadata (English)"
            _kmds_load_json(kmds_paths["en"], "(could not read EN JSON file)")

    def _on_kmds_show_ja(_):
        if kmds_paths["ja"]:
            kmds_title.value = "KMDS metadata (日本語)"
            _kmds_load_json(kmds_paths["ja"], "(could not read JA JSON file)")

    def _on_kmds_open_folder(_):
        folder = kmds_paths["dir"]
        if not folder:
            return
        try:
            import subprocess
            if sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            elif sys.platform == "win32":
                os.startfile(folder)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as ex:
            process_status_text.value = f"Could not open folder: {ex}"
            page.update()

    def _on_kmds_hide(_):
        kmds_panel.visible = False
        page.update()

    kmds_en_btn.on_click = _on_kmds_show_en
    kmds_ja_btn.on_click = _on_kmds_show_ja
    kmds_openfolder_btn.on_click = _on_kmds_open_folder
    kmds_close_btn.on_click = _on_kmds_hide

    kmds_clock = {"running": False, "t0": 0.0}

    def on_kmds_click(_):
        if not (KMDS_AVAILABLE and app.pdf_path and str(app.pdf_path).lower().endswith(".pdf")):
            process_status_text.value = "Open a PDF first (KMDS runs on the whole PDF)."
            page.update()
            return
        prompt_path = os.path.join(SCRIPT_DIR, "extraction_prompt.md")
        if not os.path.exists(prompt_path):
            process_status_text.value = "extraction_prompt.md not found; cannot run KMDS."
            page.update()
            return
        kmds_btn.disabled = True
        open_record_btn.disabled = True   # gate the viewer until KMDS is done
        process_progress_ring.visible = True
        page.update()

        # Live elapsed-time loader: ticks once a second until KMDS completes.
        import time as _time
        kmds_clock["running"] = True
        kmds_clock["t0"] = _time.time()

        def _kmds_ticker():
            while kmds_clock.get("running"):
                el = _time.time() - kmds_clock["t0"]
                if not kmds_clock.get("running"):
                    break
                process_status_text.value = (
                    f"⏳ Extracting KMDS metadata… {el:0.0f}s elapsed "
                    f"(typically 60–90s). Please wait — do not close."
                )
                page.update()
                _time.sleep(1.0)

        page.run_thread(_kmds_ticker)

        def run_kmds():
            # Whole body guarded: a worker-thread exception in flet is otherwise
            # swallowed silently (the executor future is never awaited), which
            # looks exactly like "nothing happened".
            try:
                import asyncio
                base = Path(app.pdf_path).stem
                out_dir = os.path.join(os.path.dirname(app.pdf_path) or ".",
                                       f"{base}_kmds")
                os.makedirs(out_dir, exist_ok=True)
                try:
                    summary = asyncio.run(kmds_parallel.extract_kmds_parallel(
                        app.pdf_path, out_dir, base_name=base, prompt_path=prompt_path,
                    ))
                except Exception as ex:
                    summary = {"_error": f"{type(ex).__name__}: {ex}"}

                kmds_clock["running"] = False   # stop the elapsed-time loader
                process_progress_ring.visible = False
                kmds_btn.disabled = False

                if summary.get("_error"):
                    process_status_text.value = f"KMDS failed: {summary['_error']}"
                    open_record_btn.disabled = not bool(app.fig_digitizations)
                    page.update()
                    return

                kmds_paths["en"] = summary.get("en_path")
                kmds_paths["ja"] = summary.get("ja_path")
                kmds_paths["dir"] = out_dir
                try:
                    with open(summary.get("en_path"), "r", encoding="utf-8") as _rf:
                        app.kmds_record = json.load(_rf)
                except Exception:
                    app.kmds_record = None
                n_ok = summary.get("n_sections_ok", 0)
                n_tot = summary.get("n_sections", 0)
                elapsed = summary.get("elapsed_sec", 0)
                out_tok = summary.get("output_tokens", 0)
                tr_ok = (summary.get("translation") or {}).get("ok")

                # Surface WHY sections failed — the UI used to hide this, so a
                # fast all-fail (bad API key, no model access, rate limit, SSL)
                # looked like "saved but empty".
                section_errs = []
                for _sk, _sv in (summary.get("sections") or {}).items():
                    if not _sv.get("ok") and _sv.get("error"):
                        section_errs.append(f"{_sk}: {_sv['error']}")
                err_line = ("\n⚠ " + section_errs[0]) if section_errs else ""

                nv = summary.get("n_schema_violations")
                conf = ("" if nv is None else
                        ("   |   Schema: ✓ valid" if nv == 0
                         else f"   |   Schema: {nv} violation(s)"))
                kmds_summary_text.value = (
                    f"Sections OK: {n_ok}/{n_tot}   |   Translation (JA): "
                    f"{'OK' if tr_ok else 'failed'}{conf}   |   {elapsed:.0f}s, "
                    f"{out_tok:,} output tokens\nSaved to: {out_dir}{err_line}"
                )
                kmds_ja_btn.disabled = not kmds_paths["ja"]
                kmds_title.value = "KMDS metadata (English)"
                _kmds_load_json(kmds_paths["en"], "(could not read EN JSON file)")
                kmds_panel.visible = True
                if n_ok == 0:
                    # Nothing extracted — don't clobber a prior good record, and
                    # tell the user the actual cause.
                    app.kmds_record = None
                    open_record_btn.disabled = not bool(app.fig_digitizations)
                    process_status_text.value = (
                        f"KMDS got 0/{n_tot} sections — extraction failed. "
                        + (section_errs[0] if section_errs else "See terminal for details.")
                    )
                else:
                    open_record_btn.disabled = False   # KMDS ready → can open viewer
                    process_status_text.value = (
                        f"✓ KMDS done: {n_ok}/{n_tot} sections, {elapsed:.0f}s. "
                        f"Click “Open Paper Record” to view."
                        + (f"  ({len(section_errs)} section(s) failed)" if section_errs else "")
                    )
                page.update()
            except Exception as ex:
                import traceback
                traceback.print_exc()
                kmds_clock["running"] = False
                process_progress_ring.visible = False
                kmds_btn.disabled = False
                open_record_btn.disabled = not bool(app.fig_digitizations)
                process_status_text.value = f"KMDS error: {type(ex).__name__}: {ex}"
                page.update()

        page.run_thread(run_kmds)

    kmds_btn.on_click = on_kmds_click

    # ---- Paper-record handoff (native digitizer -> KMDS viewer) ----
    def on_save_fig_to_record(_):
        if app.current_image is None or not app.data_series:
            process_status_text.value = "Digitize a figure first, then save it to the record."
            page.update()
            return
        x_name = (x_axis_name_field.value or "").strip() or "X"
        y_name = (y_axis_name_field.value or "").strip() or "Y"
        series = []
        for s in app.data_series:
            series.append([list(app.pixel_to_data(p[0], p[1])) for p in s.get("points", [])])
        series_names = [app.line_name(i) for i in range(len(app.data_series))]
        total = sum(len(s["points"]) for s in app.data_series)
        idx = app.current_figure_idx
        meta = {}
        if idx is not None and 0 <= idx < len(app.pdf_figures):
            meta = app.pdf_figures[idx][1] or {}
        key = idx if idx is not None else f"img_{len(app.fig_digitizations)}"
        label = (meta.get("caption") or "").strip() or f"Figure {len(app.fig_digitizations) + 1}"
        app.fig_digitizations[key] = {
            "label": label, "page": meta.get("page"),
            "x_name": x_name, "y_name": y_name,
            "is_log_x": bool(app.axis_config and app.axis_config.get("xIsLogScale")),
            "is_log_y": bool(app.axis_config and app.axis_config.get("yIsLogScale")),
            "n_lines": len(app.data_series), "n_points": total, "series": series,
            "series_names": series_names,
        }
        if not kmds_clock.get("running"):
            open_record_btn.disabled = False   # something to view now
        process_status_text.value = (
            f"Staged “{label}” → record ({len(app.fig_digitizations)} figure(s) staged). "
            f"Press “Open Paper Record” when done."
        )
        page.update()

    def _build_paper_record_html():
        import copy, re
        record = copy.deepcopy(app.kmds_record) if app.kmds_record else {"metadata": {"publication": {"figures": []}}}
        pub = record.setdefault("metadata", {}).setdefault("publication", {})
        figs = pub.get("figures")
        if not isinstance(figs, list):
            figs = []
            pub["figures"] = figs

        def _first_int(s):
            m = re.search(r"\d+", str(s or ""))
            return int(m.group()) if m else None

        for _key, dig in app.fig_digitizations.items():
            n = _first_int(dig["label"])
            target = None
            if n is not None:
                for f in figs:
                    if (_first_int(f.get("figure local id")) == n or
                            _first_int(f.get("figure name")) == n or
                            _first_int(f.get("structure")) == n):
                        target = f
                        break
            if target is None:
                target = {"figure local id": dig["label"], "figure name": dig["label"],
                          "structure": "", "description": "(digitized in AutoLineDigitizer)",
                          "graphs": [], "comments": []}
                figs.append(target)
            summary = (f"Digitized: {dig['n_lines']} lines, {dig['n_points']} pts — "
                       f"X:{dig['x_name']}{' log' if dig['is_log_x'] else ''}, "
                       f"Y:{dig['y_name']}{' log' if dig['is_log_y'] else ''}")
            target["digitization"] = (target.get("digitization") + " · " + summary) if target.get("digitization") else summary
            dd = {
                "x_axis": {"name": dig["x_name"], "is_log": dig["is_log_x"]},
                "y_axis": {"name": dig["y_name"], "is_log": dig["is_log_y"]},
                "series": [{"label": (dig.get("series_names") or [])[i] if i < len(dig.get("series_names") or []) else f"Line {i + 1}",
                            "points_data": s} for i, s in enumerate(dig["series"])],
                "provenance": {"page": dig["page"], "n_lines": dig["n_lines"], "n_points": dig["n_points"]},
            }
            target["digitization_data"] = dd
            # The viewer renders digitization inside a GRAPH (with axis cards), so
            # ensure the figure has a graph carrying it — create one if needed.
            if not isinstance(target.get("graphs"), list):
                target["graphs"] = []
            graph = target["graphs"][0] if target["graphs"] else None
            if graph is None:
                graph = {"graph local id": dig["label"], "graph name": dig["label"],
                         "caption summary": "", "samples": [], "description": ""}
                target["graphs"].append(graph)
            if not graph.get("axes"):
                graph["axes"] = [
                    {"axis": "X", "quantity": {"term": dig["x_name"]}, "unit": "",
                     "scale": "log" if dig["is_log_x"] else "linear", "reference ticks": []},
                    {"axis": "Y", "quantity": {"term": dig["y_name"]}, "unit": "",
                     "scale": "log" if dig["is_log_y"] else "linear", "reference ticks": []},
                ]
            graph["digitization"] = summary
            graph["digitization_data"] = dd

        tmpl_path = os.path.join(SCRIPT_DIR, "kmds_paper_viewer_claude.html")
        with open(tmpl_path, "r", encoding="utf-8") as f:
            tmpl = f.read()
        data_js = json.dumps(record, ensure_ascii=False).replace("</", "<\\/")
        inject = ("<script>window.__STUDIO_RECORD = " + data_js + ";\n"
                  "try{ if(window.__STUDIO_RECORD) validateAndRender(window.__STUDIO_RECORD); }"
                  "catch(e){ console.error(e); }</script>\n")
        html_out = tmpl.replace("</body>", inject + "</body>")
        out_dir = kmds_paths.get("dir") or (os.path.dirname(app.pdf_path) if app.pdf_path else SCRIPT_DIR)
        out_path = os.path.join(out_dir, "paper_record_view.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_out)
        return out_path

    def on_open_paper_record(_):
        if not app.kmds_record and not app.fig_digitizations:
            process_status_text.value = "Run KMDS and/or stage a figure first."
            page.update()
            return
        try:
            import webbrowser
            path = _build_paper_record_html()
            webbrowser.open("file://" + path)
            process_status_text.value = f"Paper record opened in browser: {path}"
        except Exception as ex:
            import traceback
            traceback.print_exc()
            process_status_text.value = f"Open record failed: {ex}"
        page.update()

    save_fig_btn.on_click = on_save_fig_to_record
    open_record_btn.on_click = on_open_paper_record

    def save_sd_result(e: ft.FilePickerResultEvent):
        if e.path and app.data_series:
            project_json = app.convert_to_starry_digitizer_format(app.data_series, app.current_image.shape, app.axis_config)
            zip_buffer = app.create_starry_digitizer_zip(app.current_image, project_json)
            with open(e.path, 'wb') as f:
                f.write(zip_buffer.getvalue())
            process_status_text.value = f"Saved: {e.path}"
            page.update()

    def save_wpd_result(e: ft.FilePickerResultEvent):
        if e.path and app.data_series:
            wpd_json = app.convert_to_wpd_format(app.data_series, app.current_image.shape, app.axis_config)
            base_name = Path(app.current_image_path).stem if app.current_image_path else "project"
            tar_buffer = app.create_wpd_tar(app.current_image, wpd_json, project_name=base_name)
            with open(e.path, 'wb') as f:
                f.write(tar_buffer.getvalue())
            process_status_text.value = f"Saved: {e.path}"
            page.update()

    save_sd_picker = ft.FilePicker(on_result=save_sd_result)
    save_wpd_picker = ft.FilePicker(on_result=save_wpd_result)

    # CSV export picker
    def save_csv_result(e: ft.FilePickerResultEvent):
        if e.path and app.data_series:
            x_name, y_name = _current_axis_names()
            csv_str = app.to_wide_csv(x_name=x_name, y_name=y_name)
            try:
                with open(e.path, 'w', encoding='utf-8', newline='') as f:
                    f.write(csv_str)
                process_status_text.value = f"CSV saved: {e.path}"
            except Exception as ex:
                process_status_text.value = f"CSV save failed: {ex}"
            page.update()

    save_csv_picker = ft.FilePicker(on_result=save_csv_result)

    def on_export_csv_click(_):
        if not app.data_series:
            return
        save_csv_picker.save_file(
            file_name=f"data-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
            allowed_extensions=["csv"],
        )

    export_csv_btn.on_click = on_export_csv_click

    import shutil
    _model_import_context = {"mode": None}
    _download_help_controls = []

    def clear_download_help():
        sidebar_controls = settings_panel.content.controls
        for ctrl in _download_help_controls:
            if ctrl in sidebar_controls:
                sidebar_controls.remove(ctrl)
            if ctrl in main_content.controls:
                main_content.controls.remove(ctrl)
        _download_help_controls.clear()

    def import_model_result(e: ft.FilePickerResultEvent):
        if not e.files:
            return
        models_dir = get_models_dir()
        mode = _model_import_context.get("mode")
        if mode == "startup":
            for f in e.files:
                src = f.path
                fname = os.path.basename(src)
                if fname in MODEL_FILES:
                    ok, _ = verify_model_hash(src, fname)
                    if not ok:
                        status_text.value = f"Invalid file: '{fname}' is not the expected model."
                        status_text.visible = True
                        page.update()
                        return
                    shutil.copy2(src, os.path.join(models_dir, fname))
            missing = check_models_exist()
            if not missing:
                status_text.value = "Models imported. Loading..."
                status_text.visible = True
                clear_download_help()
                page.update()
                page.run_thread(load_models_async)
            else:
                status_text.value = f"Still missing: {', '.join(missing)}"
                status_text.visible = True
                page.update()
        elif mode and mode.startswith("finetune:"):
            model_key = mode.split(":", 1)[1]
            model_info = LINEFORMER_MODELS[model_key]
            if e.files:
                src = e.files[0].path
                dest = os.path.join(models_dir, model_info["checkpoint"])
                ok, _ = verify_model_hash(src, model_info["checkpoint"])
                if not ok:
                    status_text.value = f"Invalid file. Expected '{model_info['checkpoint']}'."
                    status_text.visible = True
                    page.update()
                    return
                shutil.copy2(src, dest)
                status_text.value = f"Model imported. Loading {model_info['name']}..."
                status_text.visible = True
                clear_download_help()
                page.update()

                def load_imported():
                    try:
                        app.load_lineformer_model(model_key)
                        app.raw_lines = None
                        status_text.value = "Line model loaded."
                        status_text.color = ft.colors.GREEN_700
                        progress_ring.visible = False
                        page.update()
                        if app.current_image is not None:
                            process_image(skip_axis=app.axis_config is not None)
                    except Exception as ex:
                        status_text.value = f"Model load failed: {ex}"
                        progress_ring.visible = False
                        page.update()
                page.run_thread(load_imported)

    model_import_picker = ft.FilePicker(on_result=import_model_result)
    page.overlay.extend([save_sd_picker, save_wpd_picker, save_csv_picker, model_import_picker])

    def on_model_change(e):
        key = model_dropdown.value
        if key == app.current_model_key:
            return
        status_text.value = f"Loading {LINEFORMER_MODELS[key]['name']} model..."
        status_text.color = None
        progress_ring.visible = True
        page.update()

        def switch_model():
            try:
                model_info = LINEFORMER_MODELS[key]
                ckpt_path = os.path.join(get_models_dir(), model_info["checkpoint"])
                needs_download = not os.path.exists(ckpt_path) and "huggingface_filename" in model_info
                if needs_download:
                    download_progress.visible = True
                    download_progress.value = 0
                    page.update()

                    def on_progress(downloaded, total):
                        download_progress.value = downloaded / total
                        status_text.value = f"Downloading {model_info['name']}... {downloaded // (1024*1024)}MB / {total // (1024*1024)}MB"
                        page.update()

                    app.load_lineformer_model(key, progress_callback=on_progress)
                    download_progress.visible = False
                else:
                    app.load_lineformer_model(key)
                app.raw_lines = None
                status_text.value = "Line model loaded."
                status_text.color = ft.colors.GREEN_700
                progress_ring.visible = False
                page.update()
                if app.current_image is not None:
                    process_image(skip_axis=app.axis_config is not None)
            except Exception as ex:
                model_info = LINEFORMER_MODELS[key]
                ckpt_path = os.path.join(get_models_dir(), model_info["checkpoint"])
                download_progress.visible = False
                progress_ring.visible = False
                if not os.path.exists(ckpt_path):
                    hf_filename = model_info.get("huggingface_filename", model_info["checkpoint"])

                    def on_import_finetune(_, _key=key):
                        _model_import_context["mode"] = f"finetune:{_key}"
                        model_import_picker.pick_files(
                            allowed_extensions=["pth"],
                            dialog_title=f"Select {hf_filename}",
                        )

                    download_help = ft.Column([
                        ft.Text("Download failed.", color=ft.colors.RED),
                        ft.Text(f"Download '{hf_filename}' from:", no_wrap=False),
                        ft.TextButton("HuggingFace", url=HUGGINGFACE_MODELS_URL),
                        ft.ElevatedButton("Import Model", icon=ft.icons.FILE_UPLOAD,
                                          on_click=on_import_finetune),
                    ], spacing=5)
                    clear_download_help()
                    _download_help_controls.append(download_help)
                    sidebar_controls = settings_panel.content.controls
                    idx = sidebar_controls.index(model_dropdown) + 1
                    sidebar_controls.insert(idx, download_help)
                else:
                    status_text.value = f"Model load failed: {ex}"
                page.update()
        page.run_thread(switch_model)

    model_dropdown.on_change = on_model_change

    upload_btn = ft.ElevatedButton(
        "Open Image (or Cmd+V)", icon=ft.icons.FOLDER_OPEN,
        on_click=lambda _: file_picker.pick_files(allowed_extensions=["png", "jpg", "jpeg", "bmp", "tiff"]),
    )

    open_pdf_btn = ft.ElevatedButton(
        "Open PDF", icon=ft.icons.PICTURE_AS_PDF, disabled=not PDF_SUPPORT,
        tooltip="Extract all chart figures from a PDF and edit each one."
                if PDF_SUPPORT else "PDF support requires PyMuPDF (pip install pymupdf).",
        on_click=lambda _: pdf_picker.pick_files(allowed_extensions=["pdf"]),
    )

    export_sd_btn = ft.OutlinedButton(
        "Export .zip", icon=ft.icons.DOWNLOAD, disabled=True,
        on_click=lambda _: save_sd_picker.save_file(
            file_name=f"sd-{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip",
            allowed_extensions=["zip"]),
    )

    export_wpd_btn = ft.OutlinedButton(
        "Export .tar", icon=ft.icons.DOWNLOAD, disabled=True,
        on_click=lambda _: save_wpd_picker.save_file(
            file_name=f"wpd-{datetime.now().strftime('%Y%m%d-%H%M%S')}.tar",
            allowed_extensions=["tar"]),
    )

    verify_btn = ft.OutlinedButton(
        "Verify with AI", icon=ft.icons.AUTO_FIX_HIGH, disabled=True,
        tooltip="Send current extraction to Claude to fix missing/stray points. Needs ANTHROPIC_API_KEY.",
    )

    detect_markers_btn = ft.OutlinedButton(
        "Detect Markers", icon=ft.icons.RADIO_BUTTON_CHECKED, disabled=True,
        tooltip="Replace each line's points with actual marker positions, detected via color-thickness peaks along the LineFormer trace. Best for sparse marker charts (e.g. thermoelectric data).",
    )

    axis_fix_btn = ft.OutlinedButton(
        "Fix Axis (AI)", icon=ft.icons.STRAIGHTEN, disabled=True,
        tooltip="Use Claude to read the axis tick labels — including scientific "
                "notation (10^-4) and log scales — and recalibrate the axes. "
                "Needs ANTHROPIC_API_KEY.",
    )

    label_lines_btn = ft.OutlinedButton(
        "Label Lines (AI)", icon=ft.icons.LABEL, disabled=True,
        tooltip="Use Claude to read the chart legend and name each detected line "
                "(which line is which series). Needs ANTHROPIC_API_KEY.",
    )

    def on_label_lines_click(_):
        if app.current_image is None or not app.data_series:
            return
        process_status_text.value = "Reading the legend with Claude…"
        process_progress_ring.visible = True
        page.update()

        def run_label():
            try:
                import line_utils
                if app.vlm is None:
                    app.vlm = VLMVerifier(verify_ssl=True)
                colors = list(line_utils.get_distinct_colors(len(app.data_series)))
                labels = app.vlm.label_lines_by_legend(
                    app.current_image, app.data_series, colors=colors,
                    model="claude-sonnet-4-6")
                n_named = 0
                for i, lab in enumerate(labels):
                    if i < len(app.data_series):
                        if lab:
                            app.data_series[i]["label"] = lab
                            n_named += 1
                        else:
                            app.data_series[i].pop("label", None)
                app.result_image = app.draw_points_on_image(
                    app.current_image, app.data_series, app.axis_config)
                result_image.src_base64 = image_to_base64(
                    cv2.cvtColor(app.result_image, cv2.COLOR_BGR2RGB))
                result_image.visible = True
                populate_detected_lines()
                update_data_table()
                names = ", ".join(app.line_name(i) for i in range(len(app.data_series)))
                process_status_text.value = (
                    f"Labeled {n_named}/{len(app.data_series)} lines: {names}"[:200])
            except Exception as ex:
                import traceback
                traceback.print_exc()
                process_status_text.value = f"Label lines failed: {type(ex).__name__}: {ex}"
            finally:
                process_progress_ring.visible = False
                page.update()

        page.run_thread(run_label)

    label_lines_btn.on_click = on_label_lines_click

    def on_axis_fix_click(_):
        if app.current_image is None:
            return
        process_status_text.value = "Reading axis ticks with Claude..."
        process_progress_ring.visible = True
        page.update()

        def run_axis_fix():
            try:
                cfg, notes = app.apply_vlm_axis_correction(app.current_image)
                if cfg is None:
                    process_status_text.value = f"Axis AI: {notes or 'could not recalibrate.'}"
                else:
                    # Redraw overlay + refresh the calibrated data table.
                    app.result_image = app.draw_points_on_image(
                        app.current_image, app.data_series or [], app.axis_config)
                    result_image.src_base64 = image_to_base64(
                        cv2.cvtColor(app.result_image, cv2.COLOR_BGR2RGB))
                    result_image.visible = True
                    axis_info_text.value = (
                        f"Axis (AI): X=[{cfg['x1_val']} → {cfg['x2_val']}]"
                        f"{' log' if cfg.get('xIsLogScale') else ''}, "
                        f"Y=[{cfg['y1_val']} → {cfg['y2_val']}]"
                        f"{' log' if cfg.get('yIsLogScale') else ''}")
                    update_data_table()
                    process_status_text.value = (
                        f"Axis recalibrated by AI. {notes}".strip())
            except Exception as ex:
                import traceback
                traceback.print_exc()
                process_status_text.value = f"Axis AI failed: {type(ex).__name__}: {ex}"
            finally:
                process_progress_ring.visible = False
                page.update()

        page.run_thread(run_axis_fix)

    axis_fix_btn.on_click = on_axis_fix_click

    def on_verify_click(_):
        if app.current_image is None or not app.data_series:
            return
        app.selected_line_idx = None
        app.edit_mode = None
        app.add_anchors = []
        edit_panel.visible = False
        _hide_edit_subcontrols()
        process_status_text.value = "Verifying with Claude (this can take ~10-30s)..."
        process_progress_ring.visible = True
        page.update()

        def run_verify():
            try:
                import line_utils
                if app.vlm is None:
                    app.vlm = VLMVerifier(verify_ssl=True)
                colors = list(line_utils.get_distinct_colors(len(app.data_series)))
                corrected, debug = app.vlm.verify_and_correct(
                    app.current_image, app.data_series,
                    axis_config=app.axis_config, colors=colors,
                )
                assessment = (debug.get("parsed") or {}).get("overall_assessment", "")
                print(f"[vlm] {assessment}")
                app.data_series = corrected
                app.raw_lines = [list(s["points"]) for s in corrected]
                app.result_image = app.draw_points_on_image(app.current_image, app.data_series, app.axis_config)
                result_image.src_base64 = image_to_base64(cv2.cvtColor(app.result_image, cv2.COLOR_BGR2RGB))
                result_image.visible = True
                total_points = sum(len(s["points"]) for s in app.data_series)
                line_pts = [len(s["points"]) for s in app.data_series]
                info_text.value = (f"{len(app.data_series)} lines after AI verify ({total_points} points)\n"
                                   f"Points per line: {', '.join(map(str, line_pts))}")
                if assessment:
                    info_text.value += f"\nClaude: {assessment}"
                populate_detected_lines()
                update_data_table()
                process_status_text.value = "AI verification complete."
            except Exception as ex:
                process_status_text.value = f"Verify failed: {ex}"
            finally:
                process_progress_ring.visible = False
                page.update()

        page.run_thread(run_verify)

    verify_btn.on_click = on_verify_click

    def on_detect_markers_click(_):
        if app.current_image is None or not app.data_series:
            return
        if not MARKER_DETECTOR_AVAILABLE:
            process_status_text.value = "line_marker_detector module not installed."
            page.update()
            return
        app.selected_line_idx = None
        app.edit_mode = None
        app.add_anchors = []
        edit_panel.visible = False
        _hide_edit_subcontrols()
        process_status_text.value = "Detecting markers along each line..."
        process_progress_ring.visible = True
        page.update()

        def run_detect():
            try:
                new_series_unsorted = []
                total_markers = 0
                no_marker_lines = 0
                traces = app.raw_lines or []

                for curve in traces:
                    if len(curve) < 3:
                        new_series_unsorted.append({"points": []})
                        continue
                    markers, _color_lab, _thick = find_markers_on_traced_line(
                        app.current_image, curve,
                        color_tol_lab=12.0,
                        min_marker_thickness=3.5,
                        min_peak_distance_px=10,
                    )
                    if markers:
                        pts = [[int(round(x)), int(round(y))] for (x, y) in markers]
                        new_series_unsorted.append({"points": pts})
                        total_markers += len(pts)
                    else:
                        # Fallback: keep downsampled points for lines without clear markers
                        pts = app.downsample_points(curve)
                        new_series_unsorted.append({"points": pts})
                        no_marker_lines += 1

                app.data_series = app.sort_data_series(new_series_unsorted)
                app.result_image = app.draw_points_on_image(
                    app.current_image, app.data_series, app.axis_config
                )
                result_image.src_base64 = image_to_base64(
                    cv2.cvtColor(app.result_image, cv2.COLOR_BGR2RGB)
                )
                result_image.visible = True

                total_pts = sum(len(s["points"]) for s in app.data_series)
                line_pts = [len(s["points"]) for s in app.data_series]
                msg = (f"{len(app.data_series)} lines, {total_pts} marker points\n"
                       f"Points per line: {', '.join(map(str, line_pts))}")
                if no_marker_lines:
                    msg += f"\n({no_marker_lines} lines had no clear markers — using downsampled points instead)"
                info_text.value = msg
                populate_detected_lines()
                update_data_table()
                process_status_text.value = (
                    f"Detected {total_markers} markers across "
                    f"{len(app.data_series) - no_marker_lines} lines."
                )
            except Exception as ex:
                import traceback
                traceback.print_exc()
                process_status_text.value = f"Marker detection failed: {ex}"
            finally:
                process_progress_ring.visible = False
                page.update()

        page.run_thread(run_detect)

    detect_markers_btn.on_click = on_detect_markers_click

    settings_panel = ft.Container(
        content=ft.Column([
            _section_label("Models"),
            model_dropdown,
            ft.Row([progress_ring, status_text], spacing=5),
            color_refine_checkbox,
            axis_model_dropdown,
            axis_status_text,
            sort_dropdown,
            _soft_divider(),
            _section_label("Sampling"),
            downsample_dropdown,
            max_points_label, max_points_slider,
            fixed_step_label, fixed_step_slider,
            _soft_divider(),
            detected_lines_title,
            detected_lines_hint,
            detected_lines_column,
            edit_panel,
            verify_btn,
            detect_markers_btn,
            axis_fix_btn,
            label_lines_btn,
            _soft_divider(),
            _section_label("Adjust in digitizer"),
            ft.Text("StarryDigitizer", size=13, weight=ft.FontWeight.W_500, color=INK),
            ft.Row([export_sd_btn,
                    ft.IconButton(icon=ft.icons.OPEN_IN_NEW, tooltip="Open StarryDigitizer",
                                  url="https://starrydigitizer.vercel.app/")], spacing=0),
            ft.Text("WebPlotDigitizer", size=13, weight=ft.FontWeight.W_500, color=INK),
            ft.Row([export_wpd_btn,
                    ft.IconButton(icon=ft.icons.OPEN_IN_NEW, tooltip="Open WebPlotDigitizer",
                                  url="https://apps.automeris.io/wpd/")], spacing=0),
            ft.Container(expand=True),
            ft.Text(f"v{APP_VERSION}", size=11, color=INK_3),
        ], spacing=9, expand=True, scroll=ft.ScrollMode.AUTO),
        width=280, padding=16,
        bgcolor=SURFACE, border_radius=14,
        border=ft.border.all(1, RULE),
        margin=ft.margin.only(right=4),
    )

    process_status_text = ft.Text("", size=12)
    process_progress_ring = ft.ProgressRing(visible=False, width=16, height=16)

    result_canvas = ft.GestureDetector(
        content=result_image,
        on_tap_down=on_canvas_tap,
        on_pan_update=on_canvas_pan,
        mouse_cursor=ft.MouseCursor.PRECISE,
    )

    def _card(content, pad=14):
        return ft.Container(content=content, padding=pad, bgcolor=SURFACE,
                            border_radius=14, border=ft.border.all(1, RULE))

    data_table_section = _card(ft.Column([
        ft.Row([data_table_title, ft.Container(expand=True), export_csv_btn]),
        ft.Row([x_axis_name_field, y_axis_name_field, table_line_picker], spacing=10),
        table_row_count_text,
        data_table_scroll,
    ], spacing=8))

    toolbar_card = _card(
        ft.Row([upload_btn, open_pdf_btn, pdf_detector_dropdown, review_figures_btn,
                recrop_btn, kmds_btn, save_fig_btn, open_record_btn,
                process_progress_ring, process_status_text],
               alignment=ft.MainAxisAlignment.START,
               vertical_alignment=ft.CrossAxisAlignment.CENTER,
               wrap=True, run_spacing=8, spacing=8),
        pad=12)

    def _img_card(title, body):
        return ft.Container(
            content=ft.Column([
                ft.Text(title, size=13, weight=ft.FontWeight.W_600, color=INK_3),
                ft.Container(content=body, bgcolor="#FFFFFF", border_radius=10,
                             border=ft.border.all(1, RULE), padding=4),
            ], spacing=8, width=CANVAS_W + 8),
            padding=10, bgcolor=SURFACE, border_radius=14, border=ft.border.all(1, RULE))

    main_content = ft.Column([
        toolbar_card,
        pdf_gallery,
        # Fixed-width image columns: the click-to-pixel math assumes the canvas
        # renders at exactly CANVAS_W, so the columns must NOT be squeezed.
        ft.Row([
            _img_card("Input image", input_image),
            _img_card("Extracted points", result_canvas),
        ], vertical_alignment=ft.CrossAxisAlignment.START,
           scroll=ft.ScrollMode.AUTO, spacing=12),
        info_text,
        axis_info_text,
        data_table_section,
        kmds_panel,
    ], expand=True, scroll=ft.ScrollMode.AUTO, spacing=12)

    header = ft.Container(
        content=ft.Row([
            ft.Text("●", color=ACCENT, size=15),
            ft.Column([
                ft.Text("AutoLineDigitizer", size=19, weight=ft.FontWeight.BOLD, color=INK),
                ft.Text("chart figures → calibrated data → paper record",
                        size=11, color=INK_3),
            ], spacing=0, tight=True),
            ft.Container(expand=True),
            ft.Text(f"v{APP_VERSION}", size=11, color=INK_3),
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
        padding=ft.padding.symmetric(horizontal=22, vertical=12),
        bgcolor=SURFACE, border=ft.border.only(bottom=ft.BorderSide(1, RULE)),
    )

    body = ft.Container(
        content=ft.Row([settings_panel, main_content], expand=True, spacing=16),
        expand=True, padding=18,
    )

    page.add(ft.Column([header, body], expand=True, spacing=0))

    download_progress = ft.ProgressBar(visible=False, width=200)
    sidebar_controls = settings_panel.content.controls
    _status_row_idx = sidebar_controls.index(axis_model_dropdown)
    sidebar_controls.insert(_status_row_idx, download_progress)

    def load_models_async():
        try:
            missing = check_models_exist()
            if missing:
                models_dir = get_models_dir()
                for filename in missing:
                    model_name = MODEL_FILES[filename]
                    status_text.value = f"Downloading {model_name} model..."
                    download_progress.visible = True
                    download_progress.value = 0
                    page.update()

                    def on_progress(downloaded, total, _name=model_name):
                        download_progress.value = downloaded / total
                        status_text.value = f"Downloading {_name}... {downloaded // (1024*1024)}MB / {total // (1024*1024)}MB"
                        page.update()

                    try:
                        download_model(filename, models_dir, progress_callback=on_progress)
                    except Exception as e:
                        download_progress.visible = False
                        progress_ring.visible = False

                        def on_import_startup(_):
                            _model_import_context["mode"] = "startup"
                            model_import_picker.pick_files(
                                allow_multiple=True, allowed_extensions=["pth"],
                                dialog_title="Select model files (iter_3000.pth, checkpoint.pth)",
                            )

                        download_help = ft.Column([
                            ft.Text("Auto-download failed.", color=ft.colors.RED),
                            ft.Text("1. Download from:", no_wrap=False),
                            ft.TextButton("GitHub Releases", url=GITHUB_MODELS_URL),
                            ft.Text("2. Import models:", no_wrap=False),
                            ft.ElevatedButton("Import Models", icon=ft.icons.FILE_UPLOAD,
                                              on_click=on_import_startup),
                            ft.Text("(select iter_3000.pth and checkpoint.pth)",
                                    size=11, no_wrap=False, color=ft.colors.GREY_600),
                        ], spacing=5)
                        _download_help_controls.append(download_help)
                        sidebar_controls = settings_panel.content.controls
                        idx = sidebar_controls.index(model_dropdown) + 1
                        sidebar_controls.insert(idx, download_help)
                        page.update()
                        return
                download_progress.visible = False
                page.update()

            status_text.value = "Loading..."
            axis_status_text.value = ""
            page.update()

            try:
                app.load_lineformer_model()
                status_text.value = "Line model loaded."
                status_text.color = ft.colors.GREEN_700
            except Exception as e:
                import traceback
                traceback.print_exc()
                status_text.value = f"Failed: {e}"
                status_text.color = ft.colors.RED_700
            progress_ring.visible = False
            page.update()

            axis_status_text.value = "Loading..."
            page.update()

            if not CHARTDETE_AVAILABLE:
                app.auto_axis = False
                axis_status_text.value = "Not available"
                axis_status_text.color = ft.colors.ORANGE_700
            else:
                try:
                    app.load_chartdete_model()
                    axis_status_text.value = "Axis model loaded."
                    axis_status_text.color = ft.colors.GREEN_700
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    app.auto_axis = False
                    axis_status_text.value = f"Failed: {e}"
                    axis_status_text.color = ft.colors.RED_700
        except Exception as e:
            status_text.value = f"Failed: {e}"
            status_text.color = ft.colors.RED_700
            progress_ring.visible = False
        page.update()

    page.run_thread(load_models_async)


if __name__ == "__main__":
    ft.app(target=main)