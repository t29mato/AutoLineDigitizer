# -*- coding: utf-8 -*-
"""
LineFormer Streamlit App
Chart line data extraction using LineFormer.
Automatic axis detection using ChartDete + OCR.
Output compatible with starry-digitizer and WebPlotDigitizer formats.
"""

import sys
import os

# Project root is parent of src/
_src_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_src_dir)

# Add ChartDete submodule to path FIRST (highest priority for custom mmdet models)
sys.path.insert(0, os.path.join(_project_root, 'submodules', 'chartdete'))

# Add lineformer submodule to path
sys.path.insert(0, os.path.join(_project_root, 'submodules', 'lineformer'))

# Import mmdet (uses ChartDete's mmdet which includes custom models like CascadeRoIHead_LGF)
import mmdet  # noqa: F401
from mmdet.models.roi_heads.cascade_roi_head_LGF import CascadeRoIHead_LGF  # noqa: F401

# Add src to path for local imports
sys.path.insert(0, _src_dir)

import streamlit as st
import cv2
import numpy as np
import json
import io
import zipfile
import tarfile
from datetime import datetime

# Page config
st.set_page_config(
    page_title="LineFormer - Chart Data Extraction",
    page_icon="📈",
    layout="wide"
)


@st.cache_resource
def load_lineformer_model():
    """Load LineFormer model (cached)."""
    import infer

    # Model weights in models/, config in submodules/lineformer/
    CKPT = os.path.join(_project_root, "models", "iter_3000.pth")
    CONFIG = os.path.join(_project_root, "submodules", "lineformer", "lineformer_swin_t_config.py")
    DEVICE = "cpu"

    infer.load_model(CONFIG, CKPT, DEVICE)
    return infer


@st.cache_resource
def load_chartdete_model():
    """Load ChartDete model for axis detection (cached)."""
    import chartdete_infer
    chartdete_infer.load_chartdete_model(device='cpu')
    return chartdete_infer


def detect_axis_calibration(chartdete_module, img):
    """
    Detect chart elements and extract axis calibration using ChartDete + OCR.

    Returns:
        axis_config: dict with calibration data or None
        detections: raw detection results
        ocr_results: OCR results for labels
    """
    # Run ChartDete detection
    detections = chartdete_module.detect_chart_elements(img, score_thr=0.3)

    # Get axis info with OCR
    axis_info = chartdete_module.get_axis_info(detections, img=img, with_ocr=True)

    calibration = axis_info.get('calibration')
    ocr_results = axis_info.get('ocr_results', {})

    if calibration is None:
        return None, detections, ocr_results

    # Convert calibration to axis_config format
    # For x-axis: use xlabel center positions (x pixel, fixed y at label position)
    # For y-axis: use ylabel center positions (fixed x at label position, y pixel)
    axis_config = None

    has_x = 'x1_pixel' in calibration and 'x2_pixel' in calibration
    has_y = 'y1_pixel' in calibration and 'y2_pixel' in calibration

    if has_x and has_y:
        # Get plot_area to determine y position for x-axis calibration points
        plot_area = axis_info.get('plot_area')

        if plot_area:
            # x calibration: use bottom of plot area for y
            x_calib_y = plot_area[3]  # y2 of plot_area (bottom)
            # y calibration: use left of plot area for x
            y_calib_x = plot_area[0]  # x1 of plot_area (left)
        else:
            # Fallback: use image dimensions
            x_calib_y = img.shape[0] * 0.9
            y_calib_x = img.shape[1] * 0.1

        # Note: In calibration from OCR:
        #   y1_pixel/y1_value = top label (higher Y pixel, but could be higher or lower value)
        #   y2_pixel/y2_value = bottom label (lower Y pixel)
        # In starry-digitizer/WPD format:
        #   y1 = bottom point (lower Y pixel = higher on screen)
        #   y2 = top point (higher Y pixel = lower on screen)
        # So we swap y1 and y2 from OCR calibration

        axis_config = {
            # X axis calibration points (at bottom of chart)
            "x1_px": calibration['x1_pixel'],
            "x1_py": x_calib_y,
            "x1_val": calibration['x1_value'],
            "x2_px": calibration['x2_pixel'],
            "x2_py": x_calib_y,
            "x2_val": calibration['x2_value'],
            # Y axis calibration points (at left of chart)
            # y1 = bottom (higher pixel Y), y2 = top (lower pixel Y)
            "y1_px": y_calib_x,
            "y1_py": calibration['y2_pixel'],  # bottom label
            "y1_val": calibration['y2_value'],
            "y2_px": y_calib_x,
            "y2_py": calibration['y1_pixel'],  # top label
            "y2_val": calibration['y1_value'],
            "xIsLogScale": False,
            "yIsLogScale": False,
        }

    return axis_config, detections, ocr_results


def downsample_points(points, mode, fixed_step, max_points):
    """Downsample points based on mode."""
    if len(points) <= 1:
        return points

    if mode == "none":
        return points
    elif mode == "fixed":
        return points[::fixed_step]
    elif mode == "max_points":
        if len(points) <= max_points:
            return points
        step = max(1, len(points) // max_points)
        return points[::step]

    return points


def sort_data_series(data_series, sort_mode):
    """Sort data series based on the specified mode."""
    if sort_mode == "original" or len(data_series) == 0:
        return data_series

    if sort_mode == "mean_y_desc":
        # Sort by mean Y descending (higher Y value = lower on screen in image coords)
        # For chart interpretation: lower Y pixel = higher value, so desc means high→low value
        return sorted(data_series, key=lambda s: np.mean([pt[1] for pt in s["points"]]))
    elif sort_mode == "mean_y_asc":
        # Sort by mean Y ascending
        return sorted(data_series, key=lambda s: np.mean([pt[1] for pt in s["points"]]), reverse=True)

    return data_series


def extract_lines(infer_module, img, downsample_mode, fixed_step, max_points):
    """Extract line data from image."""
    line_dataseries = infer_module.get_dataseries(img, to_clean=False)

    data_series = []
    for line in line_dataseries:
        if len(line) == 0:
            continue

        # Extract all points
        all_points = [[int(pt['x']), int(pt['y'])] for pt in line]

        # Downsample
        points = downsample_points(all_points, downsample_mode, fixed_step, max_points)

        data_series.append({"points": points})

    return data_series, line_dataseries


def draw_points_on_image(img, data_series, axis_config=None):
    """Draw extracted points as symbols on image, with optional axis calibration markers."""
    import line_utils

    result_img = img.copy()
    num_lines = len(data_series)
    colors = list(line_utils.get_distinct_colors(num_lines))

    # Marker symbols (using different shapes)
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
        # Single color for all calibration points (Magenta - visible on white backgrounds)
        calib_color = (255, 0, 255)   # Magenta (BGR)
        outline_color = (0, 0, 0)  # Black outline for contrast
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # Helper function to draw text with background
        def draw_text_with_bg(img, text, pos, color):
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x, y = pos
            # Draw background rectangle
            padding = 3
            cv2.rectangle(img, (x - padding, y - text_h - padding),
                         (x + text_w + padding, y + baseline + padding),
                         (255, 255, 255), -1)  # White background
            cv2.rectangle(img, (x - padding, y - text_h - padding),
                         (x + text_w + padding, y + baseline + padding),
                         outline_color, 1)  # Black border
            # Draw text
            cv2.putText(img, text, (x, y), font, font_scale, color, thickness)

        # Helper function to draw calibration point with marker
        def draw_calib_point(img, x, y, color, label, label_offset):
            # Draw filled circle with outline
            cv2.circle(img, (x, y), 12, outline_color, 3)  # Black outline
            cv2.circle(img, (x, y), 10, color, -1)  # Filled circle
            cv2.circle(img, (x, y), 10, outline_color, 2)  # Inner outline
            # Draw crosshair inside circle
            cv2.line(img, (x - 6, y), (x + 6, y), outline_color, 2)
            cv2.line(img, (x, y - 6), (x, y + 6), outline_color, 2)
            # Draw label with background
            label_x = x + label_offset[0]
            label_y = y + label_offset[1]
            draw_text_with_bg(img, label, (label_x, label_y), color)

        # Get calibration points
        x1_x, x1_y = int(axis_config['x1_px']), int(axis_config['x1_py'])
        x2_x, x2_y = int(axis_config['x2_px']), int(axis_config['x2_py'])
        y1_x, y1_y = int(axis_config['y1_px']), int(axis_config['y1_py'])
        y2_x, y2_y = int(axis_config['y2_px']), int(axis_config['y2_py'])

        # Draw dashed lines connecting calibration points
        # X-axis line (X1 to X2)
        dash_length = 10
        gap_length = 5
        # Draw dashed line for X-axis
        dx = x2_x - x1_x
        dy = x2_y - x1_y
        dist = max(1, int(np.sqrt(dx*dx + dy*dy)))
        for i in range(0, dist, dash_length + gap_length):
            start_x = int(x1_x + dx * i / dist)
            start_y = int(x1_y + dy * i / dist)
            end_i = min(i + dash_length, dist)
            end_x = int(x1_x + dx * end_i / dist)
            end_y = int(x1_y + dy * end_i / dist)
            cv2.line(result_img, (start_x, start_y), (end_x, end_y), calib_color, 2)

        # Draw dashed line for Y-axis
        dx = y2_x - y1_x
        dy = y2_y - y1_y
        dist = max(1, int(np.sqrt(dx*dx + dy*dy)))
        for i in range(0, dist, dash_length + gap_length):
            start_x = int(y1_x + dx * i / dist)
            start_y = int(y1_y + dy * i / dist)
            end_i = min(i + dash_length, dist)
            end_x = int(y1_x + dx * end_i / dist)
            end_y = int(y1_y + dy * end_i / dist)
            cv2.line(result_img, (start_x, start_y), (end_x, end_y), calib_color, 2)

        # Draw calibration points with labels
        # X1 point (left on X axis)
        draw_calib_point(result_img, x1_x, x1_y, calib_color,
                        f"X1={axis_config['x1_val']}", (15, -5))

        # X2 point (right on X axis) - label on left side to avoid edge
        draw_calib_point(result_img, x2_x, x2_y, calib_color,
                        f"X2={axis_config['x2_val']}", (-100, -5))

        # Y1 point (bottom on Y axis)
        draw_calib_point(result_img, y1_x, y1_y, calib_color,
                        f"Y1={axis_config['y1_val']}", (15, 20))

        # Y2 point (top on Y axis)
        draw_calib_point(result_img, y2_x, y2_y, calib_color,
                        f"Y2={axis_config['y2_val']}", (15, -5))

    return result_img


def convert_to_starry_digitizer_format(data_series, img_shape, axis_config=None):
    """
    Convert LineFormer output to starry-digitizer project.json format.

    axis_config: dict with x1, x2, y1, y2 pixel coordinates and values
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    # Default axis set (user needs to calibrate in starry-digitizer)
    if axis_config is None:
        # Use image corners as default
        axis_set = {
            "id": 1,
            "name": "XY Axes 1",
            "x1": {
                "name": "x1",
                "value": 0,
                "coord": {"xPx": 0, "yPx": float(img_shape[0])}
            },
            "x2": {
                "name": "x2",
                "value": 100,
                "coord": {"xPx": float(img_shape[1]), "yPx": float(img_shape[0])}
            },
            "y1": {
                "name": "y1",
                "value": 0,
                "coord": {"xPx": 0, "yPx": float(img_shape[0])}
            },
            "y2": {
                "name": "y2",
                "value": 100,
                "coord": {"xPx": 0, "yPx": 0}
            },
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
            "x1": {
                "name": "x1",
                "value": axis_config["x1_val"],
                "coord": {"xPx": axis_config["x1_px"], "yPx": axis_config["x1_py"]}
            },
            "x2": {
                "name": "x2",
                "value": axis_config["x2_val"],
                "coord": {"xPx": axis_config["x2_px"], "yPx": axis_config["x2_py"]}
            },
            "y1": {
                "name": "y1",
                "value": axis_config["y1_val"],
                "coord": {"xPx": axis_config["y1_px"], "yPx": axis_config["y1_py"]}
            },
            "y2": {
                "name": "y2",
                "value": axis_config["y2_val"],
                "coord": {"xPx": axis_config["y2_px"], "yPx": axis_config["y2_py"]}
            },
            "xIsLogScale": axis_config.get("xIsLogScale", False),
            "yIsLogScale": axis_config.get("yIsLogScale", False),
            "considerGraphTilt": False,
            "pointMode": 0,
            "isVisible": True
        }

    # Convert datasets
    datasets = []

    # Add empty dataset 1 (starry-digitizer convention)
    datasets.append({
        "id": 1,
        "name": "dataset 1",
        "axisSetId": 1,
        "points": [],
        "visiblePointIds": [],
        "manuallyAddedPointIds": []
    })

    # Add extracted lines
    for idx, series in enumerate(data_series):
        points = []
        visible_ids = []
        for pt_idx, pt in enumerate(series["points"]):
            pt_id = pt_idx + 1
            points.append({
                "id": pt_id,
                "xPx": float(pt[0]),
                "yPx": float(pt[1])
            })
            visible_ids.append(pt_id)

        datasets.append({
            "id": idx + 2,  # Start from 2 (1 is empty dataset)
            "name": f"Line {idx + 1}",
            "axisSetId": 1,
            "points": points,
            "visiblePointIds": visible_ids,
            "manuallyAddedPointIds": []
        })

    project = {
        "version": "1.11.2",
        "timestamp": timestamp,
        "axisSets": [axis_set],
        "activeAxisSetId": 1,
        "datasets": datasets,
        "activeDatasetId": len(datasets),
        "canvasHandler": {
            "scale": 1.0,
            "manualMode": 0
        }
    }

    return project


def create_starry_digitizer_zip(img, project_json):
    """
    Create a ZIP file containing image.png and project.json
    for starry-digitizer import.
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add image.png
        _, img_encoded = cv2.imencode('.png', img)
        zf.writestr('image.png', img_encoded.tobytes())

        # Add project.json
        json_str = json.dumps(project_json, indent=2, ensure_ascii=False)
        zf.writestr('project.json', json_str.encode('utf-8'))

    zip_buffer.seek(0)
    return zip_buffer


def convert_to_wpd_format(data_series, img_shape, axis_config=None):
    """
    Convert LineFormer output to WebPlotDigitizer JSON format.

    This format can be loaded in WebPlotDigitizer after loading the image.
    """
    # Default calibration points (user needs to recalibrate in WPD)
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
        "name": "XY",
        "type": "XYAxes",
        "isLogX": is_log_x,
        "isLogY": is_log_y,
        "noRotation": False,
        "calibrationPoints": calibration_points
    }]

    # Convert datasets
    dataset_coll = []

    # Add empty default dataset (WPD convention)
    dataset_coll.append({
        "name": "Default Dataset",
        "axesName": "XY",
        "colorRGB": [200, 0, 0, 255],
        "metadataKeys": [],
        "data": [],
        "autoDetectionData": None
    })

    # Add extracted lines
    for idx, series in enumerate(data_series):
        data_points = []
        for pt in series["points"]:
            # WPD format: x, y are pixel coords, value is [realX, realY] (null if not calibrated)
            data_points.append({
                "x": float(pt[0]),
                "y": float(pt[1]),
                "value": None  # Will be calculated by WPD after calibration
            })

        dataset_coll.append({
            "name": f"Dataset {idx + 1}",
            "axesName": "XY",
            "colorRGB": [200, 0, 0, 255],
            "metadataKeys": [],
            "data": data_points,
            "autoDetectionData": None
        })

    wpd_json = {
        "version": [4, 2],
        "axesColl": axes_coll,
        "datasetColl": dataset_coll,
        "measurementColl": []
    }

    return wpd_json


def create_wpd_tar(img, wpd_json, project_name="project"):
    """
    Create a TAR file for WebPlotDigitizer import.

    WPD expects TAR structure:
      projectName/
      projectName/info.json
      projectName/wpd.json
      projectName/image.png
    """
    import time
    tar_buffer = io.BytesIO()
    mtime = time.time()

    with tarfile.open(fileobj=tar_buffer, mode='w') as tf:
        # Add project folder
        folder_info = tarfile.TarInfo(name=f'{project_name}/')
        folder_info.type = tarfile.DIRTYPE
        folder_info.mtime = mtime
        tf.addfile(folder_info)

        # Add info.json
        info_json = {
            "version": [4, 0],
            "json": "wpd.json",
            "images": ["image.png"]
        }
        info_bytes = json.dumps(info_json, ensure_ascii=False).encode('utf-8')
        info_tarinfo = tarfile.TarInfo(name=f'{project_name}/info.json')
        info_tarinfo.size = len(info_bytes)
        info_tarinfo.mtime = mtime
        tf.addfile(info_tarinfo, io.BytesIO(info_bytes))

        # Add wpd.json
        wpd_bytes = json.dumps(wpd_json, ensure_ascii=False).encode('utf-8')
        wpd_tarinfo = tarfile.TarInfo(name=f'{project_name}/wpd.json')
        wpd_tarinfo.size = len(wpd_bytes)
        wpd_tarinfo.mtime = mtime
        tf.addfile(wpd_tarinfo, io.BytesIO(wpd_bytes))

        # Add image.png
        _, img_encoded = cv2.imencode('.png', img)
        img_bytes = img_encoded.tobytes()
        img_tarinfo = tarfile.TarInfo(name=f'{project_name}/image.png')
        img_tarinfo.size = len(img_bytes)
        img_tarinfo.mtime = mtime
        tf.addfile(img_tarinfo, io.BytesIO(img_bytes))

    tar_buffer.seek(0)
    return tar_buffer


def main():
    st.title("📈 LineFormer - Chart Data Extraction")
    st.markdown("""
    Upload a chart image to extract line data automatically.
    Output is compatible with **starry-digitizer** and **WebPlotDigitizer**.

    **[LineFormer Paper (ICDAR 2023)](https://arxiv.org/abs/2305.01837)**
    """)

    # Sidebar settings
    st.sidebar.header("Settings")

    show_visualization = st.sidebar.checkbox("Show visualization", value=True)

    st.sidebar.subheader("Axis Detection")
    auto_axis = st.sidebar.checkbox("Auto-detect axis (ChartDete + OCR)", value=True,
                                    help="Automatically detect axis labels and calibration")

    st.sidebar.subheader("Line Sorting")
    sort_mode = st.sidebar.selectbox(
        "Sort by",
        options=["original", "mean_y_desc", "mean_y_asc"],
        format_func=lambda x: {
            "original": "Original (Detection Order)",
            "mean_y_desc": "Mean Y (High → Low)",
            "mean_y_asc": "Mean Y (Low → High)",
        }[x]
    )

    st.sidebar.subheader("Downsampling")
    downsample_mode = st.sidebar.selectbox(
        "Mode",
        options=["max_points", "fixed", "none"],
        index=0,
        help="max_points: Limit points per line, fixed: Every N points, none: All points"
    )

    if downsample_mode == "fixed":
        fixed_step = st.sidebar.slider("Fixed step (every N points)", 1, 50, 10)
        max_points = 50
    elif downsample_mode == "max_points":
        max_points = st.sidebar.slider("Max points per line", 10, 200, 50)
        fixed_step = 10
    else:
        fixed_step = 10
        max_points = 50

    # Load LineFormer model
    with st.spinner("Loading LineFormer model..."):
        try:
            infer_module = load_lineformer_model()
            st.sidebar.success("LineFormer loaded!")
        except Exception as e:
            st.error(f"Failed to load LineFormer: {e}")
            st.stop()

    # Load ChartDete model if needed
    chartdete_module = None
    if auto_axis:
        with st.spinner("Loading ChartDete model..."):
            try:
                chartdete_module = load_chartdete_model()
                st.sidebar.success("ChartDete loaded!")
            except Exception as e:
                st.warning(f"ChartDete not available: {e}")
                auto_axis = False

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a chart image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"]
    )

    # Status placeholder right after file uploader (below Drag and drop)
    status_placeholder = st.empty()

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Failed to read image")
            st.stop()

        # Initialize axis detection variables
        axis_config = None
        detections = None
        ocr_results = None

        # Display columns - show input image immediately
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Size: {img.shape[1]} x {img.shape[0]} pixels")

        with col2:
            # Create placeholders for dynamic updates
            st.subheader("Extraction Result")
            viz_placeholder = st.empty()
            summary_placeholder = st.empty()
            caption_placeholder = st.empty()

        # Placeholder for axis calibration results (outside columns)
        axis_placeholder = st.empty()

        # Step 1: Extract lines (faster) - with spinner
        with status_placeholder.container():
            with st.spinner("⏳ Extracting lines (LineFormer)..."):
                data_series, raw_lines = extract_lines(
                    infer_module, img, downsample_mode, fixed_step, max_points
                )

                # Sort lines
                data_series = sort_data_series(data_series, sort_mode)

        # Show initial result (without axis calibration) immediately
        if show_visualization:
            result_img = draw_points_on_image(img, data_series, None)
            viz_placeholder.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Show line summary
        total_points = sum(len(s['points']) for s in data_series)
        line_pts = [len(s['points']) for s in data_series]
        summary_placeholder.success(f"**{len(data_series)} lines** detected ({total_points} points total)")
        caption_placeholder.caption(f"Points per line: {', '.join(map(str, line_pts))}")

        # Step 2: Run axis detection (slower, OCR-heavy) - with spinner
        if auto_axis and chartdete_module is not None:
            with status_placeholder.container():
                with st.spinner("🔍 Detecting axis labels (ChartDete + OCR)..."):
                    axis_config, detections, ocr_results = detect_axis_calibration(
                        chartdete_module, img
                    )

            # Update visualization with axis calibration
            if show_visualization:
                result_img = draw_points_on_image(img, data_series, axis_config)
                viz_placeholder.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Clear the status
            status_placeholder.empty()
        else:
            # Clear status if no axis detection
            status_placeholder.empty()

        # Show axis calibration results
        if axis_config is not None:
            with axis_placeholder.container():
                with st.expander("Axis Calibration (Auto-detected)", expanded=True):
                    col_x, col_y = st.columns(2)
                    with col_x:
                        st.markdown("**X-axis:**")
                        st.write(f"  {axis_config['x1_val']} → {axis_config['x2_val']}")
                    with col_y:
                        st.markdown("**Y-axis:**")
                        st.write(f"  {axis_config['y1_val']} → {axis_config['y2_val']}")

                    # Show OCR details
                    if ocr_results:
                        st.markdown("**Detected Labels:**")
                        ocr_text = []
                        if 'xlabels' in ocr_results:
                            x_vals = [f"{l['value']}" for l in ocr_results['xlabels'] if l['value'] is not None]
                            ocr_text.append(f"X: [{', '.join(x_vals)}]")
                        if 'ylabels' in ocr_results:
                            y_vals = [f"{l['value']}" for l in ocr_results['ylabels'] if l['value'] is not None]
                            ocr_text.append(f"Y: [{', '.join(y_vals)}]")
                        st.caption(' | '.join(ocr_text))
        elif auto_axis:
            axis_placeholder.warning("Could not auto-detect axis calibration. Manual calibration needed in WPD/starry-digitizer.")

        # Build starry-digitizer project
        project_json = convert_to_starry_digitizer_format(
            data_series, img.shape, axis_config
        )

        # Build WebPlotDigitizer project
        wpd_json = convert_to_wpd_format(
            data_series, img.shape, axis_config
        )

        # Create ZIP file for starry-digitizer
        zip_buffer = create_starry_digitizer_zip(img, project_json)

        # Create TAR file for WebPlotDigitizer
        base_name = os.path.splitext(uploaded_file.name)[0]
        tar_buffer = create_wpd_tar(img, wpd_json, project_name=base_name)

        # Generate filenames
        timestamp_str = datetime.now().strftime('%Y%m%d-%H%M%S')
        zip_filename = f"sd-{timestamp_str}.zip"
        tar_filename = f"wpd-{timestamp_str}.tar"

        # Download buttons
        st.subheader("Download")
        col_dl1, col_dl2, col_dl3 = st.columns(3)

        with col_dl1:
            st.download_button(
                label="📦 starry-digitizer (.zip)",
                data=zip_buffer.getvalue(),
                file_name=zip_filename,
                mime="application/zip",
                help="ZIP containing image.png + project.json"
            )

        with col_dl2:
            st.download_button(
                label="📊 WebPlotDigitizer (.tar)",
                data=tar_buffer.getvalue(),
                file_name=tar_filename,
                mime="application/x-tar",
                help="TAR with project folder structure"
            )

        with col_dl3:
            if show_visualization:
                _, buffer = cv2.imencode('.png', result_img)
                st.download_button(
                    label="🖼️ Visualization (.png)",
                    data=buffer.tobytes(),
                    file_name=f"{base_name}_result.png",
                    mime="image/png"
                )

        # Show JSON previews
        with st.expander("Preview starry-digitizer project.json"):
            json_str = json.dumps(project_json, indent=2, ensure_ascii=False)
            if len(json_str) > 5000:
                st.code(json_str[:5000] + "\n... (truncated)", language="json")
            else:
                st.code(json_str, language="json")

        with st.expander("Preview WebPlotDigitizer wpd.json"):
            wpd_preview = json.dumps(wpd_json, indent=2, ensure_ascii=False)
            if len(wpd_preview) > 5000:
                st.code(wpd_preview[:5000] + "\n... (truncated)", language="json")
            else:
                st.code(wpd_preview, language="json")

        # Instructions
        st.info("""
        **starry-digitizer:** Download ZIP → Open [starry-digitizer](https://starrydigitizer.vercel.app/) → Load Project

        **WebPlotDigitizer:** Download TAR → Open [WPD](https://apps.automeris.io/wpd4/) → File → Load Project (.tar)
        """)


if __name__ == "__main__":
    main()
