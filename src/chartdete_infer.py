# -*- coding: utf-8 -*-
"""
ChartDete inference module with CPU support.
Uses torchvision NMS fallback for CPU inference.
Includes EasyOCR for reading axis label text.
"""
import sys
import os
import re

# Project paths
_src_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_src_dir)

# Add chartdete submodule to path only if mmdet is not already imported
# This prevents duplicate hook registration when used with LineFormer
_chartdete_path = os.path.join(_project_root, 'submodules', 'chartdete')
if 'mmdet' not in sys.modules:
    sys.path.insert(0, _chartdete_path)

import torch
import numpy as np
import cv2
import torchvision.ops as tv_ops

# EasyOCR (lazy load)
_ocr_reader = None


def _patch_mmcv_ops():
    """Patch mmcv ops to use torchvision for CPU."""
    import mmcv.ops
    import mmcv.ops.nms as mmcv_nms_module

    # ========== Patch NMSop class ==========
    class TorchvisionNMSop(torch.autograd.Function):
        @staticmethod
        def forward(ctx, bboxes, scores, iou_threshold, offset, score_threshold, max_num):
            is_filtering_by_score = score_threshold > 0
            if is_filtering_by_score:
                valid_mask = scores > score_threshold
                bboxes_f, scores_f = bboxes[valid_mask], scores[valid_mask]
                valid_inds = torch.nonzero(valid_mask, as_tuple=False).squeeze(dim=1)
            else:
                bboxes_f, scores_f = bboxes, scores
                valid_inds = None

            if bboxes_f.numel() == 0:
                return torch.zeros(0, dtype=torch.long, device=bboxes.device)

            # Use torchvision NMS
            inds = tv_ops.nms(bboxes_f, scores_f, iou_threshold)

            if max_num > 0:
                inds = inds[:max_num]
            if is_filtering_by_score and valid_inds is not None:
                inds = valid_inds[inds]
            return inds

        @staticmethod
        def backward(ctx, grad_output):
            return None, None, None, None, None, None

    # Replace NMSop in the module
    mmcv_nms_module.NMSop = TorchvisionNMSop

    # ========== Patch batched_nms ==========
    def torchvision_batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
        """Batched NMS using torchvision."""
        nms_cfg_ = nms_cfg.copy()
        class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
        iou_thr = nms_cfg_.get('iou_threshold', nms_cfg_.get('iou_thr', 0.5))

        if class_agnostic:
            boxes_for_nms = boxes
        else:
            if boxes.numel() == 0:
                return boxes.new_zeros((0, 5)), torch.zeros(0, dtype=torch.long, device=boxes.device)
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (max_coordinate + 1)
            boxes_for_nms = boxes + offsets[:, None]

        if boxes_for_nms.numel() == 0:
            return boxes.new_zeros((0, 5)), torch.zeros(0, dtype=torch.long, device=boxes.device)

        keep = tv_ops.nms(boxes_for_nms, scores, iou_thr)

        max_num = nms_cfg_.get('max_num', -1)
        if max_num > 0 and len(keep) > max_num:
            keep = keep[:max_num]

        dets = torch.cat([boxes[keep], scores[keep].unsqueeze(1)], dim=1)
        return dets, keep

    # Replace batched_nms in both places
    mmcv.ops.batched_nms = torchvision_batched_nms
    mmcv_nms_module.batched_nms = torchvision_batched_nms

    # ========== RoIAlign ==========
    def patched_roi_align(input, rois, output_size, spatial_scale=1.0,
                          sampling_ratio=-1, pool_mode='avg', aligned=True):
        """Use torchvision roi_align as fallback."""
        return tv_ops.roi_align(
            input, rois, output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio if sampling_ratio > 0 else 2,
            aligned=aligned
        )

    # Patch roi_align function
    original_roi_align = mmcv.ops.roi_align

    def wrapped_roi_align(input, rois, output_size, spatial_scale=1.0,
                          sampling_ratio=-1, pool_mode='avg', aligned=True):
        try:
            return original_roi_align(input, rois, output_size, spatial_scale,
                                     sampling_ratio, pool_mode, aligned)
        except NotImplementedError:
            return patched_roi_align(input, rois, output_size, spatial_scale,
                                    sampling_ratio, pool_mode, aligned)

    mmcv.ops.roi_align = wrapped_roi_align

    # Patch RoIAlign class forward method
    from mmcv.ops import RoIAlign
    original_roialign_forward = RoIAlign.forward

    def patched_roialign_forward(self, input, rois):
        try:
            return original_roialign_forward(self, input, rois)
        except NotImplementedError:
            return tv_ops.roi_align(
                input, rois, self.output_size,
                spatial_scale=self.spatial_scale,
                sampling_ratio=self.sampling_ratio if self.sampling_ratio > 0 else 2,
                aligned=self.aligned
            )

    RoIAlign.forward = patched_roialign_forward


# Apply patch before importing mmdet
_patch_mmcv_ops()

from mmdet.apis import init_detector, inference_detector

# ChartDete classes
CHARTDETE_CLASSES = [
    'x_title', 'y_title', 'plot_area', 'other', 'xlabel', 'ylabel',
    'chart_title', 'x_tick', 'y_tick', 'legend_patch', 'legend_label',
    'legend_title', 'legend_area', 'mark_label', 'value_label',
    'y_axis_area', 'x_axis_area', 'tick_grouping'
]

# Indices for axis-related classes
AXIS_CLASSES = {
    'x_title': 0,
    'y_title': 1,
    'plot_area': 2,
    'xlabel': 4,
    'ylabel': 5,
    'x_tick': 7,
    'y_tick': 8,
    'y_axis_area': 15,
    'x_axis_area': 16,
}


_model = None


def load_chartdete_model(config_path=None, checkpoint_path=None, device='cpu'):
    """Load ChartDete model."""
    global _model

    if config_path is None:
        config_path = os.path.join(_project_root, 'config', 'chartdete_config.py')

    if checkpoint_path is None:
        checkpoint_path = os.path.join(_project_root, 'models', 'checkpoint.pth')

    _model = init_detector(config_path, checkpoint_path, device=device)
    return _model


def detect_chart_elements(img, score_thr=0.5, model=None):
    """
    Detect chart elements in an image.

    Args:
        img: Image path or numpy array (BGR)
        score_thr: Score threshold for detection
        model: Optional model instance

    Returns:
        dict: Detection results keyed by class name
              Each value is a list of [x1, y1, x2, y2, score]
    """
    global _model

    if model is None:
        model = _model

    if model is None:
        raise ValueError("Model not loaded. Call load_chartdete_model() first.")

    result = inference_detector(model, img)

    detections = {}
    for i, class_result in enumerate(result):
        class_name = CHARTDETE_CLASSES[i]
        if len(class_result) > 0:
            high_conf = class_result[class_result[:, 4] > score_thr]
            if len(high_conf) > 0:
                detections[class_name] = high_conf.tolist()

    return detections


def get_ocr_reader():
    """Get or initialize EasyOCR reader."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader


def parse_numeric_value(text):
    """
    Parse numeric value from OCR text.
    Handles scientific notation, negative numbers, decimals.

    Returns:
        float or None if parsing fails
    """
    if not text:
        return None

    # Clean up text
    text = text.strip()

    # Handle common OCR errors
    text = text.replace('O', '0').replace('o', '0')
    text = text.replace('l', '1').replace('I', '1')
    text = text.replace(',', '.')  # European decimal
    text = text.replace(' ', '')

    # Try to extract number with regex
    # Matches: -123, 12.34, 1.23e-4, 1.23E+4, etc.
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    match = re.search(pattern, text)

    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def infer_axis_scale(values, positions):
    """
    Infer the correct axis scale from a list of values and positions.
    Helps correct OCR errors by detecting linear patterns.

    Args:
        values: List of numeric values (may contain None or errors)
        positions: List of pixel positions

    Returns:
        Corrected list of values
    """
    # Filter valid values
    valid_pairs = [(v, p) for v, p in zip(values, positions) if v is not None]

    if len(valid_pairs) < 2:
        return values

    # Try to find linear relationship
    vs = [v for v, _ in valid_pairs]
    ps = [p for _, p in valid_pairs]

    # Check if values form a linear sequence
    diffs = [vs[i+1] - vs[i] for i in range(len(vs)-1)]
    pos_diffs = [ps[i+1] - ps[i] for i in range(len(ps)-1)]

    # If roughly uniform spacing in both, infer scale
    if len(diffs) >= 2:
        # Calculate expected step size
        avg_val_step = sum(diffs) / len(diffs)
        avg_pos_step = sum(pos_diffs) / len(pos_diffs)

        # Check consistency
        if avg_pos_step != 0:
            scale = avg_val_step / avg_pos_step

            # Correct any outliers
            corrected = list(values)
            for i, (v, p) in enumerate(zip(values, positions)):
                if v is None:
                    # Interpolate from neighbors
                    if i > 0 and values[i-1] is not None:
                        expected = values[i-1] + scale * (positions[i] - positions[i-1])
                        corrected[i] = round(expected)
            return corrected

    return values


def ocr_region(img, bbox, padding=10):
    """
    Run OCR on a specific region of the image.

    Args:
        img: Image (BGR numpy array)
        bbox: [x1, y1, x2, y2] bounding box
        padding: Extra pixels to add around bbox

    Returns:
        str: OCR result text
    """
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    h, w = img.shape[:2]

    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    # Crop region
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return ""

    # Preprocessing for better OCR
    # Convert to grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    # Scale up significantly for small text (decimal points are tiny)
    target_height = 128  # Larger for better decimal point detection
    if gray.shape[0] < target_height:
        scale = target_height / gray.shape[0]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Sharpening to make decimal points more visible
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    gray_sharp = cv2.filter2D(gray, -1, kernel)
    gray_sharp = np.clip(gray_sharp, 0, 255).astype(np.uint8)

    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_contrast = clahe.apply(gray_sharp)

    # Try multiple OCR approaches and pick best result
    reader = get_ocr_reader()

    all_results = []

    # Approach 1: Sharpened + contrast enhanced
    results1 = reader.readtext(gray_contrast, allowlist='0123456789.-eE+')
    for r in results1:
        all_results.append((r[1], r[2], 'sharp'))

    # Approach 2: Binarized with adaptive threshold (better for small dots)
    binary_adapt = cv2.adaptiveThreshold(gray_sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    results2 = reader.readtext(binary_adapt, allowlist='0123456789.-eE+')
    for r in results2:
        all_results.append((r[1], r[2], 'adapt'))

    # Approach 3: Inverted binary
    binary_inv = 255 - binary_adapt
    results3 = reader.readtext(binary_inv, allowlist='0123456789.-eE+')
    for r in results3:
        all_results.append((r[1], r[2], 'inv'))

    # Approach 4: Morphological enhancement (dilate to make dots larger)
    kernel_dilate = np.ones((2, 2), np.uint8)
    gray_dilated = cv2.dilate(gray_sharp, kernel_dilate, iterations=1)
    results4 = reader.readtext(gray_dilated, allowlist='0123456789.-eE+')
    for r in results4:
        all_results.append((r[1], r[2], 'dilate'))

    if not all_results:
        return ""

    # Prefer results that contain a decimal point (more likely to be correct)
    results_with_dot = [(t, c, m) for t, c, m in all_results if '.' in t]
    if results_with_dot:
        results_with_dot.sort(key=lambda x: x[1], reverse=True)
        return results_with_dot[0][0]

    # Otherwise return highest confidence
    all_results.sort(key=lambda x: x[1], reverse=True)
    return all_results[0][0]


def validate_and_correct_axis_values(labels, is_y_axis=False):
    """
    Validate axis values and correct obvious OCR errors using linear interpolation.

    Common OCR errors:
    - Missing decimal point: 45 instead of 4.5
    - Wrong digit: 20 instead of 2.0

    Strategy: If values form a roughly linear sequence, detect outliers.
    """
    if len(labels) < 3:
        return labels

    # Extract values and positions
    values = [l['value'] for l in labels]
    if is_y_axis:
        positions = [(l['bbox'][1] + l['bbox'][3]) / 2 for l in labels]  # y center
    else:
        positions = [(l['bbox'][0] + l['bbox'][2]) / 2 for l in labels]  # x center

    # Filter valid values for analysis
    valid_indices = [i for i, v in enumerate(values) if v is not None]
    if len(valid_indices) < 3:
        return labels

    valid_values = [values[i] for i in valid_indices]
    valid_positions = [positions[i] for i in valid_indices]

    # Check if values form a linear sequence
    # Calculate expected step sizes
    diffs = [valid_values[i+1] - valid_values[i] for i in range(len(valid_values)-1)]
    pos_diffs = [valid_positions[i+1] - valid_positions[i] for i in range(len(valid_positions)-1)]

    # Check for consistent step (allowing some tolerance)
    if len(diffs) >= 2:
        median_diff = sorted(diffs)[len(diffs)//2]

        # If most diffs are similar, we have a linear scale
        consistent_count = sum(1 for d in diffs if abs(d - median_diff) < abs(median_diff) * 0.3)

        if consistent_count >= len(diffs) * 0.6:
            # Linear scale detected - check for outliers
            scale = median_diff / (sum(pos_diffs) / len(pos_diffs)) if sum(pos_diffs) != 0 else 0

            # Detect values that are likely wrong (off by factor of 10)
            for i in range(1, len(valid_indices) - 1):
                idx = valid_indices[i]
                prev_idx = valid_indices[i-1]
                next_idx = valid_indices[i+1]

                expected = (values[prev_idx] + values[next_idx]) / 2
                actual = values[idx]

                if actual is not None and expected != 0:
                    ratio = actual / expected
                    # Check if off by factor of 10
                    if 8 < ratio < 12:  # ~10x too high
                        labels[idx]['value'] = actual / 10
                        labels[idx]['corrected'] = True
                    elif 0.08 < ratio < 0.12:  # ~10x too low
                        labels[idx]['value'] = actual * 10
                        labels[idx]['corrected'] = True

    return labels


def ocr_labels(img, detections, label_type='both'):
    """
    Run OCR on detected xlabel and ylabel regions.

    Args:
        img: Image path or BGR numpy array
        detections: Detection results from detect_chart_elements()
        label_type: 'xlabel', 'ylabel', or 'both'

    Returns:
        dict with 'xlabels' and/or 'ylabels', each containing
        list of {'bbox': [x1,y1,x2,y2], 'text': str, 'value': float}
    """
    if isinstance(img, str):
        img = cv2.imread(img)

    results = {}

    if label_type in ['xlabel', 'both'] and 'xlabel' in detections:
        xlabels = []
        for det in detections['xlabel']:
            bbox = det[:4]
            text = ocr_region(img, bbox)
            value = parse_numeric_value(text)
            xlabels.append({
                'bbox': bbox,
                'text': text,
                'value': value,
                'confidence': det[4]
            })
        # Sort by x position (left to right)
        xlabels.sort(key=lambda x: x['bbox'][0])
        # Validate and correct
        xlabels = validate_and_correct_axis_values(xlabels, is_y_axis=False)
        results['xlabels'] = xlabels

    if label_type in ['ylabel', 'both'] and 'ylabel' in detections:
        ylabels = []
        for det in detections['ylabel']:
            bbox = det[:4]
            text = ocr_region(img, bbox)
            value = parse_numeric_value(text)
            ylabels.append({
                'bbox': bbox,
                'text': text,
                'value': value,
                'confidence': det[4]
            })
        # Sort by y position (top to bottom)
        ylabels.sort(key=lambda x: x['bbox'][1])
        # Validate and correct
        ylabels = validate_and_correct_axis_values(ylabels, is_y_axis=True)
        results['ylabels'] = ylabels

    return results


def get_axis_calibration(img, detections):
    """
    Extract axis calibration data for WebPlotDigitizer/starry-digitizer format.

    Returns:
        dict with:
        - x1_pixel, x1_value: First x calibration point
        - x2_pixel, x2_value: Second x calibration point
        - y1_pixel, y1_value: First y calibration point
        - y2_pixel, y2_value: Second y calibration point
        Or None if calibration cannot be determined
    """
    # Get OCR results for labels
    ocr_results = ocr_labels(img, detections, label_type='both')

    calibration = {}

    # Process x-axis labels
    if 'xlabels' in ocr_results:
        xlabels = [l for l in ocr_results['xlabels'] if l['value'] is not None]
        if len(xlabels) >= 2:
            # Use first and last labels with valid values
            x1_label = xlabels[0]
            x2_label = xlabels[-1]

            # Use center of bbox for pixel position
            calibration['x1_pixel'] = (x1_label['bbox'][0] + x1_label['bbox'][2]) / 2
            calibration['x1_value'] = x1_label['value']
            calibration['x2_pixel'] = (x2_label['bbox'][0] + x2_label['bbox'][2]) / 2
            calibration['x2_value'] = x2_label['value']

    # Process y-axis labels
    if 'ylabels' in ocr_results:
        ylabels = [l for l in ocr_results['ylabels'] if l['value'] is not None]
        if len(ylabels) >= 2:
            # Use first (top) and last (bottom) labels
            y1_label = ylabels[0]  # Top (usually higher value)
            y2_label = ylabels[-1]  # Bottom (usually lower value)

            # Use center of bbox for pixel position
            calibration['y1_pixel'] = (y1_label['bbox'][1] + y1_label['bbox'][3]) / 2
            calibration['y1_value'] = y1_label['value']
            calibration['y2_pixel'] = (y2_label['bbox'][1] + y2_label['bbox'][3]) / 2
            calibration['y2_value'] = y2_label['value']

    return calibration if calibration else None


def get_axis_info(detections, img=None, with_ocr=False):
    """
    Extract axis information from detections.

    Args:
        detections: Detection results from detect_chart_elements()
        img: Image (required if with_ocr=True)
        with_ocr: If True, run OCR on labels to get text values

    Returns:
        dict with plot_area, x_axis, y_axis bounding boxes
        If with_ocr=True, also includes 'ocr_results' and 'calibration'
    """
    info = {}

    if 'plot_area' in detections and len(detections['plot_area']) > 0:
        # Get highest confidence plot area
        plot_areas = sorted(detections['plot_area'], key=lambda x: x[4], reverse=True)
        info['plot_area'] = plot_areas[0][:4]  # x1, y1, x2, y2

    if 'x_axis_area' in detections and len(detections['x_axis_area']) > 0:
        x_axes = sorted(detections['x_axis_area'], key=lambda x: x[4], reverse=True)
        info['x_axis_area'] = x_axes[0][:4]

    if 'y_axis_area' in detections and len(detections['y_axis_area']) > 0:
        y_axes = sorted(detections['y_axis_area'], key=lambda x: x[4], reverse=True)
        info['y_axis_area'] = y_axes[0][:4]

    # Get tick positions
    if 'x_tick' in detections:
        info['x_ticks'] = [d[:4] for d in detections['x_tick']]

    if 'y_tick' in detections:
        info['y_ticks'] = [d[:4] for d in detections['y_tick']]

    # Get labels (bbox only)
    if 'xlabel' in detections:
        info['xlabels'] = detections['xlabel']

    if 'ylabel' in detections:
        info['ylabels'] = detections['ylabel']

    # OCR if requested
    if with_ocr and img is not None:
        info['ocr_results'] = ocr_labels(img, detections)
        info['calibration'] = get_axis_calibration(img, detections)

    return info


def visualize_detections(img, detections, output_path=None):
    """
    Visualize detected chart elements.

    Args:
        img: Image path or numpy array (BGR)
        detections: Detection results from detect_chart_elements()
        output_path: Optional path to save visualization

    Returns:
        Annotated image (BGR numpy array)
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    else:
        img = img.copy()

    # Color map for different classes
    colors = {
        'plot_area': (0, 255, 0),      # Green
        'x_axis_area': (255, 0, 0),    # Blue
        'y_axis_area': (0, 0, 255),    # Red
        'x_tick': (255, 255, 0),       # Cyan
        'y_tick': (0, 255, 255),       # Yellow
        'xlabel': (255, 0, 255),       # Magenta
        'ylabel': (128, 0, 255),       # Purple
        'x_title': (0, 128, 255),      # Orange
        'y_title': (255, 128, 0),      # Light blue
        'chart_title': (128, 255, 0),  # Lime
        'legend_area': (128, 128, 128),# Gray
    }

    for class_name, boxes in detections.items():
        color = colors.get(class_name, (200, 200, 200))
        for box in boxes:
            x1, y1, x2, y2, score = box[:5]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{class_name}: {score:.2f}"
            cv2.putText(img, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if output_path:
        cv2.imwrite(output_path, img)

    return img


if __name__ == '__main__':
    # Test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, default='chartdete_result.png')
    parser.add_argument('--ocr', action='store_true', help='Run OCR on labels')
    args = parser.parse_args()

    print("Loading model...")
    load_chartdete_model(device='cpu')
    print("Model loaded!")

    print(f"Processing {args.image}...")
    detections = detect_chart_elements(args.image, score_thr=0.3)

    print("Detected elements:")
    for class_name, boxes in detections.items():
        print(f"  {class_name}: {len(boxes)}")

    # Load image for OCR
    img = cv2.imread(args.image)

    axis_info = get_axis_info(detections, img=img, with_ocr=args.ocr)
    print("\nAxis info:")
    for key, value in axis_info.items():
        if key == 'ocr_results':
            print(f"  OCR Results:")
            for label_type, labels in value.items():
                print(f"    {label_type}:")
                for label in labels:
                    print(f"      text='{label['text']}' value={label['value']} bbox={label['bbox'][:2]}")
        elif key == 'calibration':
            print(f"  Calibration:")
            if value:
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print("    Could not determine calibration")
        else:
            print(f"  {key}: {value}")

    print(f"\nSaving visualization to {args.output}...")
    visualize_detections(args.image, detections, args.output)
    print("Done!")
