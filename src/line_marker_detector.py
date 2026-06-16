# -*- coding: utf-8 -*-
"""
line_marker_detector.py — Detect markers on LineFormer-traced curves.

Given a polyline trace (Nx2 (x,y) points) and the original chart image,
find marker positions by analyzing color thickness along the trace.
"""

import numpy as np
import cv2
from scipy.signal import find_peaks
from skimage import color as skcolor


def find_markers_on_traced_line(image_bgr, line_points,
                                  color_tol_lab=12.0,
                                  min_marker_thickness=3.5,
                                  min_peak_distance_px=10,
                                  morph_kernel=3):
    """
    Detect marker positions along a traced curve via color-thickness analysis.

    Args:
        image_bgr: chart image (BGR, HxWx3, uint8)
        line_points: Nx2 array of (x, y) points along the curve (pixel coords)
        color_tol_lab: Lab color matching tolerance (smaller = stricter)
        min_marker_thickness: minimum distance-transform value to qualify as marker
        min_peak_distance_px: minimum spacing between markers (in line-index units)
        morph_kernel: morphological closing kernel size (fills hollow markers)

    Returns:
        markers: list of (x, y) marker positions
        line_color_lab: detected line color in Lab space
        thickness_signal: 1D array of distance-transform values along the line
    """
    H, W = image_bgr.shape[:2]
    line_points = np.array(line_points, dtype=float)
    if len(line_points) < 3:
        return [], None, np.array([])

    # 1. Convert image to Lab
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    img_lab = skcolor.rgb2lab(img_rgb)

    # 2. Sample line color from middle of trace
    sample_colors = []
    for x, y in line_points[::max(1, len(line_points)//20)]:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            sample_colors.append(img_lab[yi, xi])
    if not sample_colors:
        return [], None, np.array([])
    line_color_lab = np.median(np.array(sample_colors), axis=0)

    # 3. Build color mask
    color_diff = np.linalg.norm(img_lab - line_color_lab, axis=2)
    color_mask = (color_diff < color_tol_lab).astype(np.uint8)

    # 4. Morphological closing to fill hollow markers
    if morph_kernel > 0:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    # 5. Distance transform — thickness map
    dist = cv2.distanceTransform(color_mask, cv2.DIST_L2, 5)

    # 6. Sample distance ALONG the trace
    line_distances = []
    for x, y in line_points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            line_distances.append(dist[yi, xi])
        else:
            line_distances.append(0.0)
    line_distances = np.array(line_distances)

    # 7. Find peaks
    peaks, _ = find_peaks(line_distances,
                          height=min_marker_thickness,
                          distance=min_peak_distance_px)

    markers = [tuple(line_points[i]) for i in peaks]
    return markers, line_color_lab, line_distances


if __name__ == "__main__":
    print("Module loaded. Call find_markers_on_traced_line() to detect markers.")
