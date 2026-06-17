# -*- coding: utf-8 -*-
"""Tests for line_marker_detector.find_markers_on_traced_line.

Builds a synthetic chart with known marker positions and verifies the
detector recovers them. (Promoted from the old throwaway script that only
printed/wrote a PNG — now with real assertions.)
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from line_marker_detector import find_markers_on_traced_line

TRUE_MARKERS = [(150, 350), (220, 280), (300, 220), (390, 200),
                (480, 230), (570, 290), (640, 340)]
LINE_COLOR_BGR = (0, 100, 200)


def _synthetic_chart():
    """White canvas with one curve and filled circle markers; returns (img, trace)."""
    img = np.full((400, 700, 3), 255, dtype=np.uint8)
    for a, b in zip(TRUE_MARKERS, TRUE_MARKERS[1:]):
        cv2.line(img, a, b, LINE_COLOR_BGR, 2)
    for (x, y) in TRUE_MARKERS:
        cv2.circle(img, (x, y), 7, LINE_COLOR_BGR, -1)

    # Dense trace: linearly interpolate 50 points per segment
    trace = []
    for (x0, y0), (x1, y1) in zip(TRUE_MARKERS, TRUE_MARKERS[1:]):
        for t in np.linspace(0, 1, 50):
            trace.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return img, np.array(trace)


def test_too_few_points_returns_empty():
    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    markers, color, signal = find_markers_on_traced_line(img, [(1, 1), (2, 2)])
    assert markers == []
    assert color is None
    assert signal.size == 0


def test_return_shapes_are_consistent():
    img, trace = _synthetic_chart()
    markers, color, signal = find_markers_on_traced_line(img, trace)
    assert isinstance(markers, list)
    assert color is not None and len(color) == 3        # Lab triple
    assert signal.shape[0] == len(trace)                # one thickness sample per trace point


def test_detects_known_markers():
    img, trace = _synthetic_chart()
    markers, _, _ = find_markers_on_traced_line(img, trace)

    # Should find a meaningful subset of the 7 planted markers...
    assert len(markers) >= 4
    # ...and every detection must sit near a real marker (no wild false positives).
    for mx, my in markers:
        nearest = min(((mx - tx) ** 2 + (my - ty) ** 2) ** 0.5 for tx, ty in TRUE_MARKERS)
        assert nearest <= 25, f"detected marker ({mx:.0f},{my:.0f}) is {nearest:.0f}px from any true marker"
