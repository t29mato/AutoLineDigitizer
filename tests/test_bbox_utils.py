# -*- coding: utf-8 -*-
"""Unit tests for mineru_layout.bbox_utils.normalize_to_int_bbox.

Pure-logic tests (numpy only) — no torch/cv2/flet required.
"""
import numpy as np

from mineru_layout.bbox_utils import normalize_to_int_bbox


def test_none_and_empty_return_none():
    assert normalize_to_int_bbox(None) is None
    assert normalize_to_int_bbox([]) is None


def test_flat_xyxy_floors_min_and_ceils_max():
    # floor(10.2)=10, floor(20.8)=20, ceil(30.1)=31, ceil(40.9)=41
    assert normalize_to_int_bbox([10.2, 20.8, 30.1, 40.9]) == [10, 20, 31, 41]


def test_already_integer_box_is_unchanged():
    assert normalize_to_int_bbox([0, 0, 100, 50]) == [0, 0, 100, 50]


def test_polygon_nx2_reduces_to_bounding_box():
    poly = [[10, 10], [30, 10], [30, 40], [10, 40]]
    assert normalize_to_int_bbox(poly) == [10, 10, 30, 40]


def test_flat_polygon_8_coords_reduces_to_bounding_box():
    # interleaved x,y for a 4-point polygon
    assert normalize_to_int_bbox([0, 0, 10, 0, 10, 5, 0, 5]) == [0, 0, 10, 5]


def test_image_size_clamps_out_of_bounds():
    # image_size is (height, width)
    assert normalize_to_int_bbox([-5, -5, 200, 200], image_size=(100, 120)) == [0, 0, 120, 100]


def test_degenerate_box_returns_none():
    # zero-width box -> xmax <= xmin
    assert normalize_to_int_bbox([10, 10, 10, 40]) is None


def test_malformed_length_returns_none():
    assert normalize_to_int_bbox([1, 2, 3]) is None


def test_accepts_numpy_array_input():
    assert normalize_to_int_bbox(np.array([1.0, 2.0, 3.0, 4.0])) == [1, 2, 3, 4]
