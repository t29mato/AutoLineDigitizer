# -*- coding: utf-8 -*-
"""
LineFormer Desktop App
Chart line data extraction using LineFormer with Flet UI.
Automatic axis detection using ChartDete + OCR.
"""

import sys
import os

# Ensure the script's directory is in sys.path for local imports
# Handle both normal execution and PyInstaller bundled execution
if getattr(sys, 'frozen', False):
    # Running as PyInstaller bundle
    SCRIPT_DIR = sys._MEIPASS
else:
    # Running as normal script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CHARTDETE_DIR = os.path.join(SCRIPT_DIR, "submodules", "chartdete")
LINEFORMER_DIR = os.path.join(SCRIPT_DIR, "submodules", "lineformer")
MMDET_DIR = os.path.join(LINEFORMER_DIR, "mmdetection")
SRC_DIR = os.path.join(SCRIPT_DIR, "src")

# Add paths to sys.path
# Order matters: sys.path.insert(0, x) puts x at the FRONT of the list
# So we insert ChartDete LAST to make it highest priority
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, MMDET_DIR)
sys.path.insert(0, LINEFORMER_DIR)
sys.path.insert(0, CHARTDETE_DIR)  # Highest priority - inserted last

# Try to import ChartDete's mmdet and register custom models
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
import zipfile
import tarfile
import base64
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# GitHub repository for model downloads
GITHUB_REPO = "t29mato/AutoLineDigitizer"
GITHUB_RELEASE_TAG = "models"

# Model files to download
MODEL_FILES = {
    "iter_3000.pth": "LineFormer",
    "checkpoint.pth": "ChartDete",
}

# HuggingFace repository for fine-tuned models
HUGGINGFACE_REPO = "t29mato/lineformer-battery-finetuned"

# LineFormer model variants
LINEFORMER_MODELS = {
    "general": {
        "name": "General",
        "checkpoint": "iter_3000.pth",
    },
    "battery_iter5000": {
        "name": "Battery (iter_5000)",
        "checkpoint": "battery_iter_5000.pth",
        "huggingface_filename": "iter_5000.pth",
    },
    "battery_best_segm": {
        "name": "Battery (best_segm)",
        "checkpoint": "battery_best_segm_mAP_iter_1300.pth",
        "huggingface_filename": "best_segm_mAP_iter_1300.pth",
    },
}


def get_models_dir():
    """Get the models directory path. Uses user data directory for bundled apps."""
    if getattr(sys, 'frozen', False):
        # Bundled app: store models in user data directory
        if sys.platform == 'darwin':
            data_dir = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'AutoLineDigitizer')
        elif sys.platform == 'win32':
            data_dir = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), 'AutoLineDigitizer')
        else:
            data_dir = os.path.join(os.path.expanduser('~'), '.autolinedigitizer')
        models_dir = os.path.join(data_dir, 'models')
    else:
        # Normal script: use project models directory
        models_dir = os.path.join(SCRIPT_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def check_models_exist():
    """Check if all required model files exist."""
    models_dir = get_models_dir()
    missing = []
    for filename in MODEL_FILES:
        if not os.path.exists(os.path.join(models_dir, filename)):
            missing.append(filename)
    return missing


def download_file(url, dest_path, progress_callback=None):
    """Download a file from a URL."""
    tmp_path = dest_path + '.tmp'

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        block_size = 1024 * 1024  # 1MB

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


def download_model(filename, models_dir, progress_callback=None):
    """Download a model file from GitHub Releases."""
    url = f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}/{filename}"
    download_file(url, os.path.join(models_dir, filename), progress_callback)


def download_huggingface_model(hf_filename, dest_path, progress_callback=None):
    """Download a model file from HuggingFace."""
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
        self.raw_lines = None  # Cached raw points from model inference
        self.current_model_key = "general"

        # Settings
        self.sort_mode = "mean_y_desc"
        self.downsample_mode = "max_points"
        self.fixed_step = 10
        self.max_points = 20
        self.auto_axis = True

    def load_lineformer_model(self, model_key=None, progress_callback=None):
        """Load LineFormer model. Downloads from HuggingFace if needed."""
        import infer
        if model_key is not None:
            self.current_model_key = model_key
        model_info = LINEFORMER_MODELS[self.current_model_key]
        ckpt = os.path.join(get_models_dir(), model_info["checkpoint"])
        # Download from HuggingFace if not present
        if not os.path.exists(ckpt) and "huggingface_filename" in model_info:
            download_huggingface_model(
                model_info["huggingface_filename"], ckpt, progress_callback
            )
        CONFIG = os.path.join(LINEFORMER_DIR, "lineformer_swin_t_config.py")
        DEVICE = "cpu"
        infer.load_model(CONFIG, ckpt, DEVICE)
        self.infer_module = infer

    def load_chartdete_model(self):
        """Load ChartDete model for axis detection."""
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
        """Detect chart elements and extract axis calibration using ChartDete + OCR."""
        if self.chartdete_module is None:
            return None, None

        # Run ChartDete detection
        detections = self.chartdete_module.detect_chart_elements(img, score_thr=0.3)

        # Get axis info with OCR
        axis_info = self.chartdete_module.get_axis_info(detections, img=img, with_ocr=True)

        calibration = axis_info.get('calibration')
        ocr_results = axis_info.get('ocr_results', {})

        if calibration is None:
            return None, ocr_results

        has_x = 'x1_pixel' in calibration and 'x2_pixel' in calibration
        has_y = 'y1_pixel' in calibration and 'y2_pixel' in calibration

        axis_config = None
        if has_x and has_y:
            plot_area = axis_info.get('plot_area')

            if plot_area:
                x_calib_y = plot_area[3]
                y_calib_x = plot_area[0]
            else:
                x_calib_y = img.shape[0] * 0.9
                y_calib_x = img.shape[1] * 0.1

            axis_config = {
                "x1_px": calibration['x1_pixel'],
                "x1_py": x_calib_y,
                "x1_val": calibration['x1_value'],
                "x2_px": calibration['x2_pixel'],
                "x2_py": x_calib_y,
                "x2_val": calibration['x2_value'],
                "y1_px": y_calib_x,
                "y1_py": calibration['y2_pixel'],
                "y1_val": calibration['y2_value'],
                "y2_px": y_calib_x,
                "y2_py": calibration['y1_pixel'],
                "y2_val": calibration['y1_value'],
                "xIsLogScale": False,
                "yIsLogScale": False,
            }

        return axis_config, ocr_results

    def downsample_points(self, points):
        """Downsample points based on mode."""
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

        return points

    def sort_data_series(self, data_series):
        """Sort data series based on mode."""
        if self.sort_mode == "original" or len(data_series) == 0:
            return data_series

        if self.sort_mode == "mean_y_desc":
            return sorted(data_series, key=lambda s: np.mean([pt[1] for pt in s["points"]]))
        elif self.sort_mode == "mean_y_asc":
            return sorted(data_series, key=lambda s: np.mean([pt[1] for pt in s["points"]]), reverse=True)

        return data_series

    def extract_lines(self, img):
        """Extract line data from image using model inference."""
        line_dataseries = self.infer_module.get_dataseries(img, to_clean=False)

        self.raw_lines = []
        for line in line_dataseries:
            if len(line) == 0:
                continue
            self.raw_lines.append([[int(pt['x']), int(pt['y'])] for pt in line])

        return self.apply_downsample_and_sort()

    def apply_downsample_and_sort(self):
        """Apply downsampling and sorting to cached raw line data."""
        if self.raw_lines is None:
            return []

        data_series = []
        for all_points in self.raw_lines:
            points = self.downsample_points(all_points)
            data_series.append({"points": points})

        return self.sort_data_series(data_series)

    def draw_points_on_image(self, img, data_series, axis_config=None):
        """Draw extracted points on image with optional axis calibration markers."""
        import line_utils

        result_img = img.copy()
        num_lines = len(data_series)
        colors = list(line_utils.get_distinct_colors(num_lines))

        markers = [
            cv2.MARKER_CROSS,
            cv2.MARKER_DIAMOND,
            cv2.MARKER_SQUARE,
            cv2.MARKER_TRIANGLE_UP,
            cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_STAR,
        ]

        for line_idx, series in enumerate(data_series):
            color = colors[line_idx]
            marker = markers[line_idx % len(markers)]

            for pt in series["points"]:
                x, y = int(pt[0]), int(pt[1])
                cv2.drawMarker(result_img, (x, y), color, marker, markerSize=8, thickness=2)

        # Draw axis calibration points if available
        if axis_config is not None:
            calib_color = (255, 0, 255)  # Magenta
            outline_color = (0, 0, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            def draw_text_with_bg(img, text, pos, color):
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                x, y = pos
                padding = 3
                cv2.rectangle(img, (x - padding, y - text_h - padding),
                             (x + text_w + padding, y + baseline + padding),
                             (255, 255, 255), -1)
                cv2.rectangle(img, (x - padding, y - text_h - padding),
                             (x + text_w + padding, y + baseline + padding),
                             outline_color, 1)
                cv2.putText(img, text, (x, y), font, font_scale, color, thickness)

            def draw_calib_point(img, x, y, color, label, label_offset):
                cv2.circle(img, (x, y), 12, outline_color, 3)
                cv2.circle(img, (x, y), 10, color, -1)
                cv2.circle(img, (x, y), 10, outline_color, 2)
                cv2.line(img, (x - 6, y), (x + 6, y), outline_color, 2)
                cv2.line(img, (x, y - 6), (x, y + 6), outline_color, 2)
                label_x = x + label_offset[0]
                label_y = y + label_offset[1]
                draw_text_with_bg(img, label, (label_x, label_y), color)

            x1_x, x1_y = int(axis_config['x1_px']), int(axis_config['x1_py'])
            x2_x, x2_y = int(axis_config['x2_px']), int(axis_config['x2_py'])
            y1_x, y1_y = int(axis_config['y1_px']), int(axis_config['y1_py'])
            y2_x, y2_y = int(axis_config['y2_px']), int(axis_config['y2_py'])

            # Draw dashed lines
            dash_length = 10
            gap_length = 5

            # X-axis line
            dx, dy = x2_x - x1_x, x2_y - x1_y
            dist = max(1, int(np.sqrt(dx*dx + dy*dy)))
            for i in range(0, dist, dash_length + gap_length):
                start_x = int(x1_x + dx * i / dist)
                start_y = int(x1_y + dy * i / dist)
                end_i = min(i + dash_length, dist)
                end_x = int(x1_x + dx * end_i / dist)
                end_y = int(x1_y + dy * end_i / dist)
                cv2.line(result_img, (start_x, start_y), (end_x, end_y), calib_color, 2)

            # Y-axis line
            dx, dy = y2_x - y1_x, y2_y - y1_y
            dist = max(1, int(np.sqrt(dx*dx + dy*dy)))
            for i in range(0, dist, dash_length + gap_length):
                start_x = int(y1_x + dx * i / dist)
                start_y = int(y1_y + dy * i / dist)
                end_i = min(i + dash_length, dist)
                end_x = int(y1_x + dx * end_i / dist)
                end_y = int(y1_y + dy * end_i / dist)
                cv2.line(result_img, (start_x, start_y), (end_x, end_y), calib_color, 2)

            # Draw calibration points
            draw_calib_point(result_img, x1_x, x1_y, calib_color, f"X1={axis_config['x1_val']}", (15, -5))
            draw_calib_point(result_img, x2_x, x2_y, calib_color, f"X2={axis_config['x2_val']}", (-100, -5))
            draw_calib_point(result_img, y1_x, y1_y, calib_color, f"Y1={axis_config['y1_val']}", (15, 20))
            draw_calib_point(result_img, y2_x, y2_y, calib_color, f"Y2={axis_config['y2_val']}", (15, -5))

        return result_img

    def convert_to_starry_digitizer_format(self, data_series, img_shape, axis_config=None):
        """Convert to starry-digitizer format."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        if axis_config is None:
            axis_set = {
                "id": 1,
                "name": "XY Axes 1",
                "x1": {"name": "x1", "value": 0, "coord": {"xPx": 0, "yPx": float(img_shape[0])}},
                "x2": {"name": "x2", "value": 100, "coord": {"xPx": float(img_shape[1]), "yPx": float(img_shape[0])}},
                "y1": {"name": "y1", "value": 0, "coord": {"xPx": 0, "yPx": float(img_shape[0])}},
                "y2": {"name": "y2", "value": 100, "coord": {"xPx": 0, "yPx": 0}},
                "xIsLogScale": False,
                "yIsLogScale": False,
                "considerGraphTilt": False,
                "pointMode": 0,
                "isVisible": True
            }
        else:
            axis_set = {
                "id": 1,
                "name": "XY Axes 1",
                "x1": {"name": "x1", "value": axis_config["x1_val"], "coord": {"xPx": axis_config["x1_px"], "yPx": axis_config["x1_py"]}},
                "x2": {"name": "x2", "value": axis_config["x2_val"], "coord": {"xPx": axis_config["x2_px"], "yPx": axis_config["x2_py"]}},
                "y1": {"name": "y1", "value": axis_config["y1_val"], "coord": {"xPx": axis_config["y1_px"], "yPx": axis_config["y1_py"]}},
                "y2": {"name": "y2", "value": axis_config["y2_val"], "coord": {"xPx": axis_config["y2_px"], "yPx": axis_config["y2_py"]}},
                "xIsLogScale": axis_config.get("xIsLogScale", False),
                "yIsLogScale": axis_config.get("yIsLogScale", False),
                "considerGraphTilt": False,
                "pointMode": 0,
                "isVisible": True
            }

        datasets = [{
            "id": 1, "name": "dataset 1", "axisSetId": 1,
            "points": [], "visiblePointIds": [], "manuallyAddedPointIds": []
        }]

        for idx, series in enumerate(data_series):
            points = []
            visible_ids = []
            for pt_idx, pt in enumerate(series["points"]):
                pt_id = pt_idx + 1
                points.append({"id": pt_id, "xPx": float(pt[0]), "yPx": float(pt[1])})
                visible_ids.append(pt_id)

            datasets.append({
                "id": idx + 2, "name": f"Line {idx + 1}", "axisSetId": 1,
                "points": points, "visiblePointIds": visible_ids, "manuallyAddedPointIds": []
            })

        return {
            "version": "1.11.2", "timestamp": timestamp,
            "axisSets": [axis_set], "activeAxisSetId": 1,
            "datasets": datasets, "activeDatasetId": len(datasets),
            "canvasHandler": {"scale": 1.0, "manualMode": 0}
        }

    def convert_to_wpd_format(self, data_series, img_shape, axis_config=None):
        """Convert to WebPlotDigitizer format."""
        if axis_config is None:
            calibration_points = [
                {"px": 0.0, "py": float(img_shape[0]), "dx": "0", "dy": "0", "dz": None},
                {"px": float(img_shape[1]), "py": float(img_shape[0]), "dx": "100", "dy": "0", "dz": None},
                {"px": 0.0, "py": float(img_shape[0]), "dx": "0", "dy": "0", "dz": None},
                {"px": 0.0, "py": 0.0, "dx": "0", "dy": "100", "dz": None}
            ]
            is_log_x = False
            is_log_y = False
        else:
            calibration_points = [
                {"px": axis_config["x1_px"], "py": axis_config["x1_py"],
                 "dx": str(axis_config["x1_val"]), "dy": str(axis_config["y1_val"]), "dz": None},
                {"px": axis_config["x2_px"], "py": axis_config["x2_py"],
                 "dx": str(axis_config["x2_val"]), "dy": str(axis_config["y1_val"]), "dz": None},
                {"px": axis_config["y1_px"], "py": axis_config["y1_py"],
                 "dx": str(axis_config["x1_val"]), "dy": str(axis_config["y1_val"]), "dz": None},
                {"px": axis_config["y2_px"], "py": axis_config["y2_py"],
                 "dx": str(axis_config["x1_val"]), "dy": str(axis_config["y2_val"]), "dz": None}
            ]
            is_log_x = axis_config.get("xIsLogScale", False)
            is_log_y = axis_config.get("yIsLogScale", False)

        axes_coll = [{
            "name": "XY", "type": "XYAxes",
            "isLogX": is_log_x, "isLogY": is_log_y, "noRotation": False,
            "calibrationPoints": calibration_points
        }]

        dataset_coll = [{
            "name": "Default Dataset", "axesName": "XY",
            "colorRGB": [200, 0, 0, 255], "metadataKeys": [],
            "data": [], "autoDetectionData": None
        }]

        for idx, series in enumerate(data_series):
            data_points = [{"x": float(pt[0]), "y": float(pt[1]), "value": None} for pt in series["points"]]
            dataset_coll.append({
                "name": f"Dataset {idx + 1}", "axesName": "XY",
                "colorRGB": [200, 0, 0, 255], "metadataKeys": [],
                "data": data_points, "autoDetectionData": None
            })

        return {"version": [4, 2], "axesColl": axes_coll, "datasetColl": dataset_coll, "measurementColl": []}

    def create_starry_digitizer_zip(self, img, project_json):
        """Create ZIP for starry-digitizer."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            _, img_encoded = cv2.imencode('.png', img)
            zf.writestr('image.png', img_encoded.tobytes())
            zf.writestr('project.json', json.dumps(project_json, indent=2, ensure_ascii=False).encode('utf-8'))
        zip_buffer.seek(0)
        return zip_buffer

    def create_wpd_tar(self, img, wpd_json, project_name="project"):
        """Create TAR for WebPlotDigitizer."""
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
            info_tarinfo.size = len(info_bytes)
            info_tarinfo.mtime = mtime
            tf.addfile(info_tarinfo, io.BytesIO(info_bytes))

            wpd_bytes = json.dumps(wpd_json, ensure_ascii=False).encode('utf-8')
            wpd_tarinfo = tarfile.TarInfo(name=f'{project_name}/wpd.json')
            wpd_tarinfo.size = len(wpd_bytes)
            wpd_tarinfo.mtime = mtime
            tf.addfile(wpd_tarinfo, io.BytesIO(wpd_bytes))

            _, img_encoded = cv2.imencode('.png', img)
            img_bytes = img_encoded.tobytes()
            img_tarinfo = tarfile.TarInfo(name=f'{project_name}/image.png')
            img_tarinfo.size = len(img_bytes)
            img_tarinfo.mtime = mtime
            tf.addfile(img_tarinfo, io.BytesIO(img_bytes))

        tar_buffer.seek(0)
        return tar_buffer


def image_to_base64(img):
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def main(page: ft.Page):
    # Page settings
    page.title = "AutoLineDigitizer - Chart Data Extraction"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.width = 1200
    page.window.height = 800
    page.padding = 20

    app = LineFormerApp()

    # UI Components
    status_text = ft.Text("Loading models...", size=14)
    progress_ring = ft.ProgressRing(visible=True, width=20, height=20)

    input_image = ft.Image(visible=False, fit=ft.ImageFit.FIT_WIDTH)
    result_image = ft.Image(visible=False, fit=ft.ImageFit.FIT_WIDTH)

    info_text = ft.Text("", size=14)
    axis_info_text = ft.Text("", size=12, color=ft.colors.GREY_700)

    # Axis detection status text (no checkbox - always on when available)
    axis_status_text = ft.Text("", size=12)

    model_dropdown = ft.Dropdown(
        label="Line Model",
        value="general",
        width=200,
        options=[
            ft.dropdown.Option(key, info["name"])
            for key, info in LINEFORMER_MODELS.items()
        ],
    )

    sort_dropdown = ft.Dropdown(
        label="Sort Lines",
        value="mean_y_desc",
        width=200,
        options=[
            ft.dropdown.Option("original", "Detection Order"),
            ft.dropdown.Option("mean_y_desc", "Mean Y (High → Low)"),
            ft.dropdown.Option("mean_y_asc", "Mean Y (Low → High)"),
        ],
    )

    downsample_dropdown = ft.Dropdown(
        label="Downsampling",
        value="max_points",
        width=200,
        options=[
            ft.dropdown.Option("max_points", "Max Points"),
            ft.dropdown.Option("fixed", "Fixed Step"),
            ft.dropdown.Option("none", "None"),
        ],
    )

    max_points_label = ft.Text("Max points: 20", size=12)
    max_points_slider = ft.Slider(
        min=10, max=100, value=20, divisions=9,
        label="{value}", width=200,
    )

    fixed_step_label = ft.Text("Step: 10", size=12, visible=False)
    fixed_step_slider = ft.Slider(
        min=1, max=50, value=10, divisions=49,
        label="{value}", width=200, visible=False,
    )

    def reprocess_lines():
        """Re-apply downsampling/sorting to cached line data (no re-inference)."""
        if app.current_image is None or app.raw_lines is None:
            return
        app.data_series = app.apply_downsample_and_sort()
        app.result_image = app.draw_points_on_image(app.current_image, app.data_series, app.axis_config)
        result_image.src_base64 = image_to_base64(cv2.cvtColor(app.result_image, cv2.COLOR_BGR2RGB))
        result_image.visible = True
        total_points = sum(len(s['points']) for s in app.data_series)
        line_pts = [len(s['points']) for s in app.data_series]
        info_text.value = f"{len(app.data_series)} lines detected ({total_points} points)\nPoints per line: {', '.join(map(str, line_pts))}"
        page.update()

    def on_downsample_change(e):
        app.downsample_mode = downsample_dropdown.value
        max_points_label.visible = downsample_dropdown.value == "max_points"
        max_points_slider.visible = downsample_dropdown.value == "max_points"
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

    def on_model_change(e):
        key = model_dropdown.value
        if key == app.current_model_key:
            return
        status_text.value = f"Loading {LINEFORMER_MODELS[key]['name']} model..."
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
                status_text.value = "Model loaded."
                progress_ring.visible = False
                page.update()
                if app.current_image is not None:
                    process_image(skip_axis=app.axis_config is not None)
            except Exception as ex:
                status_text.value = f"Model load failed: {ex}"
                progress_ring.visible = False
                page.update()

        page.run_thread(switch_model)

    model_dropdown.on_change = on_model_change
    sort_dropdown.on_change = on_sort_change
    downsample_dropdown.on_change = on_downsample_change
    max_points_slider.on_change = on_max_points_change
    fixed_step_slider.on_change = on_fixed_step_change

    # Forward declarations for buttons
    export_sd_btn = None
    export_wpd_btn = None

    def process_image(skip_axis=False):
        """Process current image and update display."""
        nonlocal export_sd_btn, export_wpd_btn

        if app.current_image is None or app.infer_module is None:
            return

        status_text.value = "Extracting lines..."
        progress_ring.visible = True
        page.update()

        try:
            # Step 1: Extract lines
            app.data_series = app.extract_lines(app.current_image)

            # Show results immediately after line extraction
            app.result_image = app.draw_points_on_image(app.current_image, app.data_series, app.axis_config)
            result_image.src_base64 = image_to_base64(cv2.cvtColor(app.result_image, cv2.COLOR_BGR2RGB))
            result_image.visible = True
            total_points = sum(len(s['points']) for s in app.data_series)
            line_pts = [len(s['points']) for s in app.data_series]
            info_text.value = f"{len(app.data_series)} lines detected ({total_points} points)\nPoints per line: {', '.join(map(str, line_pts))}"
            page.update()

            # Step 2: Detect axis if enabled (skip if already detected)
            if not skip_axis:
                app.axis_config = None
                app.ocr_results = None
                if app.auto_axis and app.chartdete_module is not None:
                    status_text.value = "Detecting axis labels..."
                    page.update()
                    app.axis_config, app.ocr_results = app.detect_axis_calibration(app.current_image)

                    # Redraw with axis calibration overlay
                    app.result_image = app.draw_points_on_image(app.current_image, app.data_series, app.axis_config)
                    result_image.src_base64 = image_to_base64(cv2.cvtColor(app.result_image, cv2.COLOR_BGR2RGB))

            # Update axis info
            if app.axis_config is not None:
                axis_info_text.value = f"Axis: X=[{app.axis_config['x1_val']} → {app.axis_config['x2_val']}], Y=[{app.axis_config['y1_val']} → {app.axis_config['y2_val']}]"
            elif app.auto_axis:
                axis_info_text.value = "Axis detection failed. Manual calibration needed."
            else:
                axis_info_text.value = ""

            status_text.value = "Ready"
            progress_ring.visible = False

            # Enable export buttons
            export_sd_btn.disabled = False
            export_wpd_btn.disabled = False

        except Exception as e:
            status_text.value = f"Error: {e}"
            progress_ring.visible = False

        page.update()

    def load_image(img, image_path=None):
        """Load an image into the app and trigger processing."""
        app.current_image = img
        app.current_image_path = image_path
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
                status_text.value = "Failed to read image"
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

    def save_sd_result(e: ft.FilePickerResultEvent):
        if e.path and app.data_series:
            project_json = app.convert_to_starry_digitizer_format(app.data_series, app.current_image.shape, app.axis_config)
            zip_buffer = app.create_starry_digitizer_zip(app.current_image, project_json)
            with open(e.path, 'wb') as f:
                f.write(zip_buffer.getvalue())
            status_text.value = f"Saved: {e.path}"
            page.update()

    def save_wpd_result(e: ft.FilePickerResultEvent):
        if e.path and app.data_series:
            wpd_json = app.convert_to_wpd_format(app.data_series, app.current_image.shape, app.axis_config)
            base_name = Path(app.current_image_path).stem if app.current_image_path else "project"
            tar_buffer = app.create_wpd_tar(app.current_image, wpd_json, project_name=base_name)
            with open(e.path, 'wb') as f:
                f.write(tar_buffer.getvalue())
            status_text.value = f"Saved: {e.path}"
            page.update()

    save_sd_picker = ft.FilePicker(on_result=save_sd_result)
    save_wpd_picker = ft.FilePicker(on_result=save_wpd_result)
    page.overlay.extend([save_sd_picker, save_wpd_picker])

    # Buttons
    upload_btn = ft.ElevatedButton(
        "Open Image",
        icon=ft.icons.FOLDER_OPEN,
        on_click=lambda _: file_picker.pick_files(
            allowed_extensions=["png", "jpg", "jpeg", "bmp", "tiff"]
        ),
    )

    export_sd_btn = ft.OutlinedButton(
        "Export .zip",
        icon=ft.icons.DOWNLOAD,
        disabled=True,
        on_click=lambda _: save_sd_picker.save_file(
            file_name=f"sd-{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip",
            allowed_extensions=["zip"]
        ),
    )

    export_wpd_btn = ft.OutlinedButton(
        "Export .tar",
        icon=ft.icons.DOWNLOAD,
        disabled=True,
        on_click=lambda _: save_wpd_picker.save_file(
            file_name=f"wpd-{datetime.now().strftime('%Y%m%d-%H%M%S')}.tar",
            allowed_extensions=["tar"]
        ),
    )

    # Layout
    settings_panel = ft.Container(
        content=ft.Column([
            ft.Text("Settings", size=18, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            model_dropdown,
            sort_dropdown,
            ft.Divider(),
            downsample_dropdown,
            max_points_label,
            max_points_slider,
            fixed_step_label,
            fixed_step_slider,
            ft.Divider(),
            ft.Text("Adjust in Digitizer", size=14, weight=ft.FontWeight.BOLD),
            ft.Text("StarryDigitizer", size=13, weight=ft.FontWeight.W_500),
            ft.Row([
                export_sd_btn,
                ft.IconButton(
                    icon=ft.icons.OPEN_IN_NEW, tooltip="Open StarryDigitizer",
                    url="https://starrydigitizer.vercel.app/",
                ),
            ], spacing=0),
            ft.Text("WebPlotDigitizer", size=13, weight=ft.FontWeight.W_500),
            ft.Row([
                export_wpd_btn,
                ft.IconButton(
                    icon=ft.icons.OPEN_IN_NEW, tooltip="Open WebPlotDigitizer",
                    url="https://apps.automeris.io/wpd/",
                ),
            ], spacing=0),
        ], spacing=10),
        width=250,
        padding=10,
        bgcolor=ft.colors.GREY_100,
        border_radius=10,
    )

    paste_hint = ft.Text("or Cmd+V to paste", size=12, color=ft.colors.GREY_500)

    main_content = ft.Column([
        ft.Row([
            upload_btn,
            paste_hint,
            progress_ring,
            status_text,
            axis_status_text,
        ], alignment=ft.MainAxisAlignment.START),
        ft.Divider(),
        ft.Row([
            ft.Column([
                ft.Text("Input Image", size=16, weight=ft.FontWeight.BOLD),
                input_image,
            ], expand=True),
            ft.Column([
                ft.Text("Extracted Points", size=16, weight=ft.FontWeight.BOLD),
                result_image,
            ], expand=True),
        ], vertical_alignment=ft.CrossAxisAlignment.START),
        info_text,
        axis_info_text,
    ], expand=True)

    page.add(
        ft.Row([
            settings_panel,
            ft.VerticalDivider(),
            main_content,
        ], expand=True)
    )

    # Download progress bar
    download_progress = ft.ProgressBar(visible=False, width=400)

    # Insert progress bar into the layout (after status_text row)
    main_content.controls.insert(1, download_progress)

    # Load models on startup
    def load_models_async():
        try:
            # Check and download missing models
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
                        status_text.value = f"Download failed: {e}"
                        download_progress.visible = False
                        progress_ring.visible = False
                        page.update()
                        return

                download_progress.visible = False
                page.update()

            status_text.value = "Loading LineFormer..."
            page.update()
            app.load_lineformer_model()

            # Check if ChartDete is available
            if not CHARTDETE_AVAILABLE:
                app.auto_axis = False
                axis_status_text.value = "Axis detection: Not available"
                axis_status_text.color = ft.colors.ORANGE_700
            else:
                status_text.value = "Loading ChartDete..."
                page.update()
                try:
                    app.load_chartdete_model()
                    axis_status_text.value = "Axis detection: Ready"
                    axis_status_text.color = ft.colors.GREEN_700
                except Exception as e:
                    app.auto_axis = False
                    axis_status_text.value = f"Axis detection: Failed ({str(e)[:30]})"
                    axis_status_text.color = ft.colors.RED_700

            status_text.value = "Models loaded. Open an image to start."
            progress_ring.visible = False
        except Exception as e:
            status_text.value = f"Failed to load models: {e}"
            progress_ring.visible = False
        page.update()

    page.run_thread(load_models_async)


if __name__ == "__main__":
    ft.app(target=main)
