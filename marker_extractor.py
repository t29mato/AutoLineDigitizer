# -*- coding: utf-8 -*-
"""
Marker-based data extraction for charts that plot data points as markers
(circles, squares, triangles, diamonds), with or without connecting lines.

Why this exists
---------------
Line-segmentation models (LineFormer) struggle to keep line identities apart
where curves cross, and colour-based refinement cannot separate two series that
share a colour. But in many battery charts the *data points* are drawn as
discrete markers, and markers carry signal those approaches throw away:

  * They are locally THICK (unlike the thin lines that may connect them), so a
    marker can be isolated from its connecting line.
  * They are the authors' actual measured points - arguably the most faithful
    thing to extract.
  * Open markers enclose a small, compact patch of background, which is a clean
    topological cue independent of any connecting line.

This module finds markers two complementary ways, groups them into series by
colour, and returns one (x, y) point-list per series - the SAME format the rest
of the pipeline already consumes, so it slots in beside ColorLineExtractor.

It degrades gracefully: on a line-only chart (no markers) it returns [], so the
caller can fall back to LineFormer.

Detection strategy
------------------
1. Foreground = non-background pixels inside the plot area. Thin gridlines,
   axes and ticks survive here but are removed in step 2 because they are thin.
2a. FILLED markers - compute a distance transform of the foreground. Marker
    centres sit far from any edge (large distance); thin lines do not. Threshold
    at a multiple of the *median* foreground distance (which approximates the
    line half-width) and the surviving compact blobs are filled-marker cores.
    An optional morphological close first fills small open markers so they are
    caught here too.
2b. HOLLOW markers - an open marker encloses a small background hole. Find
    connected components of the *background* inside the plot area and keep the
    small, compact ones that do not touch the plot border. Their centroids are
    marker centres, regardless of any connecting line.
3. Union the two sets and de-duplicate centres that coincide.
4. Sample each marker's colour from the foreground pixels around its centre.
5. Group markers into series by LAB colour distance (single-linkage).

Usage
-----
    from marker_extractor import MarkerExtractor
    ext = MarkerExtractor(img_bgr, plot_area=(x1, y1, x2, y2))
    series_points, series_colors = ext.extract()
    # series_points: list of [[x, y], ...]  - drop straight into raw_lines
"""

import numpy as np
import cv2


# Pixel mean above which a pixel is treated as near-white background
BACKGROUND_LUMINANCE = 235

# Marker cores are kept where the distance transform exceeds this multiple of
# the median foreground distance (~line half-width). Higher = stricter.
CORE_THRESH_MULT = 1.7

# LAB distance below which two markers are considered the same series
COLOR_MERGE_LAB = 16.0

# Series with fewer markers than this are dropped as noise
MIN_MARKERS_PER_SERIES = 3


class MarkerExtractor:
    def __init__(
        self,
        img_bgr,
        plot_area=None,
        background_luminance=BACKGROUND_LUMINANCE,
        core_thresh_mult=CORE_THRESH_MULT,
        color_merge_lab=COLOR_MERGE_LAB,
        min_markers_per_series=MIN_MARKERS_PER_SERIES,
        fill_hollow=True,
        sat_thresh=35,
        val_thresh=160,
    ):
        """
        img_bgr      : H x W x 3 BGR image (e.g. from cv2.imread).
        plot_area    : (x1, y1, x2, y2) bbox of the plot region, or None for the
                       whole image. Pass it to exclude the legend / axis labels.
        fill_hollow  : also try to fill small open markers so the distance-based
                       detector catches them (the hole detector catches the rest).
        """
        self.img_bgr = img_bgr
        self.img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.h, self.w = img_bgr.shape[:2]
        self.bg_lum = background_luminance
        self.core_thresh_mult = core_thresh_mult
        self.color_merge_lab = color_merge_lab
        self.min_markers = min_markers_per_series
        self.fill_hollow = fill_hollow
        self.sat_thresh = sat_thresh
        self.val_thresh = val_thresh

        if plot_area is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in plot_area]
            self.plot_box = (
                max(0, x1), max(0, y1),
                min(self.w, x2), min(self.h, y2),
            )
        else:
            self.plot_box = (0, 0, self.w, self.h)

    # ------------------------------------------------------------------ #
    # Foreground
    # ------------------------------------------------------------------ #
    def _foreground(self):
        """Binary mask of 'ink' pixels inside the plot area.

        A pixel is ink if it is clearly coloured (high saturation) OR clearly
        dark (low value). This keeps coloured and black markers/lines while
        dropping the white background AND faint grey gridlines - the latter
        would otherwise carve the background into cells that mimic hollow
        markers.
        """
        x1, y1, x2, y2 = self.plot_box
        hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        ink = ((s > self.sat_thresh) | (v < self.val_thresh)).astype(np.uint8)
        area = np.zeros_like(ink)
        area[y1:y2, x1:x2] = 1
        return ink * area

    @staticmethod
    def _is_compact(area, w, h, min_extent=0.30, max_aspect=2.8):
        """True for roughly equidimensional, well-filled blobs.

        Rejects thin line fragments and the slim background wedges that form
        where two lines cross (which would otherwise look like hollow markers).
        """
        if w == 0 or h == 0:
            return False
        extent = area / float(w * h)
        aspect = w / float(h)
        return extent > min_extent and (1.0 / max_aspect) < aspect < max_aspect

    # ------------------------------------------------------------------ #
    # 2a. Filled markers: locally-thick blobs (survives connecting lines)
    # ------------------------------------------------------------------ #
    def _filled_centers(self, fg):
        work = fg
        if self.fill_hollow:
            dt0 = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
            stroke = float(np.median(dt0[fg > 0])) if np.any(fg > 0) else 1.0
            k = int(max(3, round(stroke * 3))) | 1  # odd kernel
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            work = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, ker)

        dt = cv2.distanceTransform(work, cv2.DIST_L2, 5)
        fgp = dt[work > 0]
        if fgp.size == 0:
            return [], 1.0
        stroke_half = float(np.median(fgp))  # ~ line half-width
        cores = (dt > stroke_half * self.core_thresh_mult).astype(np.uint8)

        n, _, stats, cent = cv2.connectedComponentsWithStats(cores, 8)
        out = []
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            w_ = stats[i, cv2.CC_STAT_WIDTH]
            h_ = stats[i, cv2.CC_STAT_HEIGHT]
            if a < 2 or not self._is_compact(a, w_, h_):
                continue
            out.append((float(cent[i][0]), float(cent[i][1]), (w_, h_)))
        return out, stroke_half

    # ------------------------------------------------------------------ #
    # 2b. Hollow markers: small compact enclosed background holes
    # ------------------------------------------------------------------ #
    def _hollow_centers(self, fg, stroke_half):
        x1, y1, x2, y2 = self.plot_box
        bg = np.zeros((self.h, self.w), np.uint8)
        bg[y1:y2, x1:x2] = 1
        bg = ((bg > 0) & (fg == 0)).astype(np.uint8)

        n, _, stats, cent = cv2.connectedComponentsWithStats(bg, 8)
        out = []
        max_area = (0.06 * min(x2 - x1, y2 - y1)) ** 2 * np.pi
        min_area = max(2.0, stroke_half * stroke_half * 0.5)
        for i in range(1, n):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w_ = stats[i, cv2.CC_STAT_WIDTH]
            h_ = stats[i, cv2.CC_STAT_HEIGHT]
            a = stats[i, cv2.CC_STAT_AREA]
            # The big surrounding background touches the plot border - skip it.
            if x <= x1 + 1 or y <= y1 + 1 or x + w_ >= x2 - 1 or y + h_ >= y2 - 1:
                continue
            if a < min_area or a > max_area:
                continue
            # Stricter compactness for holes (wedges at line crossings are slim).
            if not self._is_compact(a, w_, h_, min_extent=0.40, max_aspect=2.2):
                continue
            out.append((float(cent[i][0]), float(cent[i][1]), (w_, h_)))
        return out

    # ------------------------------------------------------------------ #
    # Combine + de-duplicate centres
    # ------------------------------------------------------------------ #
    def _detect_centers(self, fg):
        filled, stroke_half = self._filled_centers(fg)
        hollow = self._hollow_centers(fg, stroke_half)
        cand = filled + hollow
        if not cand:
            return [], stroke_half

        diags = [np.hypot(w_, h_) for (_, _, (w_, h_)) in cand]
        merge_r = max(3.0, 0.6 * float(np.median(diags)))  # < marker spacing

        kept = []
        for (cx, cy, wh) in cand:
            if all(np.hypot(cx - kx, cy - ky) > merge_r for (kx, ky, _) in kept):
                kept.append((cx, cy, wh))
        return kept, stroke_half

    # ------------------------------------------------------------------ #
    # Colour at a marker
    # ------------------------------------------------------------------ #
    def _color_at(self, cx, cy, fg, wh):
        w_, h_ = wh
        r = int(max(3, 0.8 * max(w_, h_)))
        x0, x1 = max(0, int(cx) - r), min(self.w, int(cx) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(self.h, int(cy) + r + 1)
        sub_fg = fg[y0:y1, x0:x1]
        sub = self.img_bgr[y0:y1, x0:x1]
        pix = sub[sub_fg > 0]
        if len(pix) < 3:
            return None
        return np.median(pix.astype(np.int32), axis=0).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Group markers into series by colour (single-linkage union-find)
    # ------------------------------------------------------------------ #
    def _cluster_by_color(self, markers):
        n = len(markers)
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

        idx = [i for i, m in enumerate(markers) if m["color"] is not None]
        labs = {}
        for i in idx:
            c = markers[i]["color"]
            labs[i] = cv2.cvtColor(
                np.array([[c]], np.uint8), cv2.COLOR_BGR2LAB
            )[0, 0].astype(np.float32)

        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                i, j = idx[ii], idx[jj]
                if np.linalg.norm(labs[i] - labs[j]) < self.color_merge_lab:
                    union(i, j)

        groups = {}
        for i in idx:
            groups.setdefault(find(i), []).append(i)
        return groups

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def extract(self):
        """
        Returns
        -------
        series_points : list of [[x, y], ...]   one entry per detected series
        series_colors : list of BGR uint8 arrays (or None), one per series
        """
        fg = self._foreground()
        centers, _ = self._detect_centers(fg)
        if not centers:
            return [], []

        markers = [
            {"xy": (cx, cy), "color": self._color_at(cx, cy, fg, wh)}
            for (cx, cy, wh) in centers
        ]

        groups = self._cluster_by_color(markers)

        series_points, series_colors = [], []
        for members in groups.values():
            if len(members) < self.min_markers:
                continue
            pts = sorted((markers[i]["xy"] for i in members), key=lambda p: p[0])
            cols = [markers[i]["color"] for i in members if markers[i]["color"] is not None]
            med = (
                np.median(np.array(cols, dtype=np.int32), axis=0).astype(np.uint8)
                if cols else None
            )
            series_points.append([[int(x), int(y)] for (x, y) in pts])
            series_colors.append(med)

        # Deterministic order: topmost series (smallest mean y) first.
        order = sorted(
            range(len(series_points)),
            key=lambda k: np.mean([p[1] for p in series_points[k]]),
        )
        series_points = [series_points[k] for k in order]
        series_colors = [series_colors[k] for k in order]
        return series_points, series_colors