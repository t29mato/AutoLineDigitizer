# -*- coding: utf-8 -*-
"""
Color-aware line extraction that resolves line identity across intersections.

Problem
-------
Instance-segmentation models like LineFormer often merge lines or swap line
identities where two lines cross. The information needed to disambiguate is
already in the image: each line has its own distinct color, and at the
crossing, one line was simply drawn on top of the other - the pixels still
belong to their original colors.

Approach
--------
1. Take LineFormer's initial line detections (lists of (x, y) points).
2. For each line, infer its representative color by sampling pixels along the
   detected centerline, deliberately avoiding regions close to other lines
   (which may be inside intersections, where the wrong color sits on top).
3. For each inferred color, build a binary mask of all pixels matching it
   (LAB distance), restricted to the plot area.
4. Re-trace each line column-by-column through its own mask. At columns where
   the line is occluded by another line, cubic interpolation fills the gap.
5. De-duplicate lines whose inferred colors are nearly identical (handles
   the case where the model produced several segments for one logical line).

Because each color is processed independently, intersections stop being a
problem: line A's color is still line A's color at the crossing.

Usage
-----
    extractor = ColorLineExtractor(img_bgr, plot_area=(x1, y1, x2, y2))
    refined, colors = extractor.refine_lines(lineformer_point_lists)
"""

import numpy as np
import cv2
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d


# LAB distance below which a pixel counts as "this line's color"
DEFAULT_LAB_THRESHOLD = 18.0

# LAB distance below which two inferred colors are considered the same line
DUPLICATE_LAB_THRESHOLD = 12.0

# Pixel mean above which we treat a pixel as near-white background
BACKGROUND_LUMINANCE = 235

# Minimum vertical run length (px) to count as a line segment in a column
MIN_VERTICAL_RUN = 2


class ColorLineExtractor:
    def __init__(self, img_bgr, plot_area=None, lab_threshold=DEFAULT_LAB_THRESHOLD):
        """
        img_bgr      : H x W x 3 BGR image (e.g. from cv2.imread).
        plot_area    : (x1, y1, x2, y2) bounding box of the plot region,
                       or None to use the full image.
        lab_threshold: LAB color distance for "matches this line's color".
                       Larger = more permissive.
        """
        self.img_bgr = img_bgr
        self.img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.h, self.w = img_bgr.shape[:2]
        self.lab_threshold = lab_threshold

        if plot_area is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in plot_area]
            self.plot_box = (
                max(0, x1), max(0, y1),
                min(self.w, x2), min(self.h, y2),
            )
        else:
            self.plot_box = (0, 0, self.w, self.h)

    # ------------------------------------------------------------------ #
    # Step 1: infer each line's representative color
    # ------------------------------------------------------------------ #
    def _sample_line_color(self, points, other_points_sets):
        """Median BGR color along `points`, ignoring samples close to other lines."""
        avoid_radius = 4
        avoid = set()
        for other in other_points_sets:
            for (ox, oy) in other:
                ox_i, oy_i = int(ox), int(oy)
                for dx in range(-avoid_radius, avoid_radius + 1):
                    for dy in range(-avoid_radius, avoid_radius + 1):
                        avoid.add((ox_i + dx, oy_i + dy))

        samples = []
        for (x, y) in points:
            xi, yi = int(x), int(y)
            if not (0 <= xi < self.w and 0 <= yi < self.h):
                continue
            if (xi, yi) in avoid:
                continue
            # 3x3 patch around the seed point
            x0, x1 = max(0, xi - 1), min(self.w, xi + 2)
            y0, y1 = max(0, yi - 1), min(self.h, yi + 2)
            patch = self.img_bgr[y0:y1, x0:x1].reshape(-1, 3).astype(np.int32)
            lum = patch.mean(axis=1)
            keep = patch[lum < BACKGROUND_LUMINANCE]
            if len(keep) > 0:
                samples.extend(keep.tolist())

        # Fallback if avoidance was too aggressive: sample everywhere
        if len(samples) < 5:
            samples = []
            for (x, y) in points:
                xi, yi = int(x), int(y)
                if 0 <= xi < self.w and 0 <= yi < self.h:
                    p = self.img_bgr[yi, xi].astype(np.int32)
                    if p.mean() < BACKGROUND_LUMINANCE:
                        samples.append(p.tolist())

        if not samples:
            return None
        return np.median(np.array(samples), axis=0).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Step 2: build a binary mask of pixels matching a target color
    # ------------------------------------------------------------------ #
    def _color_mask(self, target_bgr, seed_points=None):
        """Binary mask of pixels close to target_bgr in LAB, inside the plot area.

        If seed_points is given, the mask is further filtered to keep only the
        connected components that contain at least one seed. This removes false
        positives from same-color but disconnected elements — axis tick labels,
        end-of-line value labels, chart text — which matter mostly for black /
        near-achromatic line colors where text would otherwise pollute the mask.
        """
        target_lab = cv2.cvtColor(
            np.array([[target_bgr]], dtype=np.uint8), cv2.COLOR_BGR2LAB
        )[0, 0].astype(np.float32)
        diff = self.img_lab - target_lab[None, None, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        mask = (dist < self.lab_threshold).astype(np.uint8)

        # Restrict to plot area
        x1, y1, x2, y2 = self.plot_box
        area = np.zeros_like(mask)
        area[y1:y2, x1:x2] = 1
        mask = mask * area

        # Light morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Connected-component filter: keep only blobs that contain a seed point.
        # Seeds rarely land exactly on the stroke (anti-aliasing offsets the true
        # stroke by 1-2 px), so each seed votes via a 5x5 window's most common
        # non-zero label.
        if seed_points is not None:
            _, labels = cv2.connectedComponents(mask, connectivity=8)
            hi, wi = labels.shape
            keeper_labels = set()
            for (sx, sy) in seed_points:
                sx_i, sy_i = int(sx), int(sy)
                wy0 = max(0, sy_i - 2); wy1 = min(hi, sy_i + 3)
                wx0 = max(0, sx_i - 2); wx1 = min(wi, sx_i + 3)
                window = labels[wy0:wy1, wx0:wx1].ravel()
                nz = window[window > 0]
                if len(nz):
                    keeper_labels.add(int(np.bincount(nz).argmax()))

            # If no seed landed near any mask blob (rare; e.g. very bad color
            # sample), fall back to the unfiltered mask so the caller can still
            # use seed-anchored tracing rather than getting an all-zero mask.
            if keeper_labels:
                mask = np.isin(labels, list(keeper_labels)).astype(np.uint8)

        return mask

    # ------------------------------------------------------------------ #
    # Step 3: trace one line through its color mask
    # ------------------------------------------------------------------ #
    def _trace_through_mask(self, mask, seed_points):
        """
        Walk x-by-x through `mask`, taking the y-centroid at each column.
        Gaps are interpolated. seed_points constrains the x-range and helps
        pick the right vertical branch if the mask has multiple disjoint
        runs at a column.
        """
        x1, y1, x2, y2 = self.plot_box

        if seed_points:
            seed_arr = np.array(seed_points, dtype=float)
            x_start = max(x1, int(seed_arr[:, 0].min()))
            x_end = min(x2, int(seed_arr[:, 0].max()) + 1)
        else:
            seed_arr = None
            x_start, x_end = x1, x2

        if x_end <= x_start:
            return []

        xs_valid, ys_valid, gap_xs = [], [], []

        for x in range(x_start, x_end):
            col = mask[y1:y2, x]
            ys = np.where(col > 0)[0]
            if len(ys) == 0:
                gap_xs.append(x)
                continue

            # Group consecutive y values into runs
            runs = []
            run_start = prev = ys[0]
            for v in ys[1:]:
                if v - prev > 1:
                    runs.append((run_start, prev))
                    run_start = v
                prev = v
            runs.append((run_start, prev))
            runs = [(s, e) for (s, e) in runs if (e - s + 1) >= MIN_VERTICAL_RUN]
            if not runs:
                gap_xs.append(x)
                continue

            # Pick the run closest to the expected y from the seed line
            if seed_arr is not None:
                idx = int(np.argmin(np.abs(seed_arr[:, 0] - x)))
                expected_local_y = seed_arr[idx, 1] - y1
                run = min(runs, key=lambda r: abs(((r[0] + r[1]) / 2) - expected_local_y))
            else:
                run = max(runs, key=lambda r: r[1] - r[0])

            xs_valid.append(x)
            ys_valid.append((run[0] + run[1]) / 2 + y1)

        if len(xs_valid) < 2:
            return []

        xs_arr = np.array(xs_valid, dtype=float)
        ys_arr = np.array(ys_valid, dtype=float)

        # Cubic interpolation across gaps (intersections, occlusions, cleanup holes)
        if gap_xs and len(xs_arr) >= 4:
            try:
                f = interpolate.interp1d(
                    xs_arr, ys_arr, kind='cubic',
                    bounds_error=False, fill_value=np.nan,
                )
                for gx in gap_xs:
                    if xs_arr.min() <= gx <= xs_arr.max():
                        gy = float(f(gx))
                        if np.isfinite(gy):
                            xs_arr = np.append(xs_arr, gx)
                            ys_arr = np.append(ys_arr, gy)
            except Exception:
                pass
            order = np.argsort(xs_arr)
            xs_arr = xs_arr[order]
            ys_arr = ys_arr[order]

        # Light denoise along x to smooth out anti-aliasing jitter
        if len(ys_arr) > 5:
            ys_arr = gaussian_filter1d(ys_arr, sigma=1.0)

        return [[int(xv), int(yv)] for xv, yv in zip(xs_arr, ys_arr)]

    # ------------------------------------------------------------------ #
    # De-duplicate: merge lines whose inferred colors are nearly identical
    # ------------------------------------------------------------------ #
    def _dedupe(self, lines, colors):
        n = len(lines)
        if n < 2:
            return lines, colors

        labs = []
        for c in colors:
            if c is None:
                labs.append(None)
            else:
                lab = cv2.cvtColor(
                    np.array([[c]], dtype=np.uint8), cv2.COLOR_BGR2LAB
                )[0, 0].astype(np.float32)
                labs.append(lab)

        parent = list(range(n))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            if labs[i] is None:
                continue
            for j in range(i + 1, n):
                if labs[j] is None:
                    continue
                if np.linalg.norm(labs[i] - labs[j]) < DUPLICATE_LAB_THRESHOLD:
                    union(i, j)

        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        keep = set()
        for grp in groups.values():
            best = max(grp, key=lambda idx: len(lines[idx]))
            keep.add(best)

        out_lines = [lines[i] for i in range(n) if i in keep]
        out_colors = [colors[i] for i in range(n) if i in keep]
        return out_lines, out_colors

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def refine_lines(self, lineformer_lines, dedupe=True):
        """
        Refine LineFormer line detections using per-line color tracing.

        Parameters
        ----------
        lineformer_lines : list of [(x, y), ...]
            One entry per detected line.
        dedupe : bool
            If True, merge lines whose inferred colors are nearly the same.

        Returns
        -------
        refined_lines : list of [[x, y], ...]
        line_colors   : list of BGR uint8 arrays (or None) - one per refined line
        """
        if not lineformer_lines:
            return [], []

        # 1) Infer each line's color
        colors = []
        for i, pts in enumerate(lineformer_lines):
            others = [lineformer_lines[j] for j in range(len(lineformer_lines)) if j != i]
            colors.append(self._sample_line_color(pts, others))

        # 2) Re-trace each line via its color mask
        refined = []
        for pts, c in zip(lineformer_lines, colors):
            if c is None:
                refined.append([[int(p[0]), int(p[1])] for p in pts])  # fall back
                continue
            mask = self._color_mask(c, seed_points=pts)
            traced = self._trace_through_mask(mask, pts)
            if traced:
                refined.append(traced)
            else:
                refined.append([[int(p[0]), int(p[1])] for p in pts])  # fall back

        # 3) Merge near-identical-color duplicates
        if dedupe:
            refined, colors = self._dedupe(refined, colors)

        return refined, colors