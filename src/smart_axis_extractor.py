# -*- coding: utf-8 -*-
"""
SmartAxisExtractor
==================

A robust replacement for the old `get_axis_info` -> calibration logic.

The old approach failed because it would happily accept ANY numeric text
ChartDete + OCR returned as a candidate tick label. In real plots this
means legend entries (e.g. "25℃"), in-plot curve labels (e.g. "3.5",
"3.6" annotations on curves), titles, and units all hijack the X/Y axis
calibration.

This extractor fixes that with a strict geometric pipeline:

  1.  Anchor to the plot area bounding box (from ChartDete).
  2.  Define narrow spatial bands where real tick labels can live:
        - X-tick labels: just BELOW the bottom edge of the plot area,
          aligned horizontally with the plot's X range.
        - Y-tick labels: just LEFT of the left edge of the plot area,
          aligned vertically with the plot's Y range.
  3.  Run OCR ONLY in those bands (or filter ChartDete detections
      that fall inside those bands). Everything else - legends,
      in-plot annotations, titles - is rejected by geometry alone.
  4.  Parse each surviving string with a strict numeric regex.
      Things that don't parse as numbers (like "20℃" or "AmpHrs")
      are discarded. Note "-20℃" *would* parse as -20 if we let it,
      so we also require the box to be inside the tick-label band -
      degree-Celsius legend entries live INSIDE the plot, not below
      the bottom edge, so they're already filtered out by step 2.
  5.  Pair each label to its position on the axis (label center
      projected onto the axis line gives a pixel coordinate).
  6.  Detect log vs linear by checking whether the (pixel, value)
      pairs are better fit by value or by log10(value).
  7.  Fit the calibration with ALL valid pairs via least squares,
      so one bad OCR read is an outlier we can detect and drop,
      not a calibration-breaker.
  8.  Return a confidence score (RMS residual in pixels).

Usage:

    from smart_axis_extractor import SmartAxisExtractor
    extractor = SmartAxisExtractor()
    axis_config, debug = extractor.extract(
        img,
        plot_area=(x0, y0, x1, y1),       # from chartdete
        ocr_fn=my_ocr_function,            # callable: img_crop -> [(text, box), ...]
        tick_label_boxes=None,             # optional: pre-detected boxes from chartdete
    )

If `tick_label_boxes` is provided (list of (text, (x,y,w,h))), OCR is skipped
and the boxes are filtered by geometry. Otherwise, the extractor crops the
two tick-label bands and runs `ocr_fn` on them.
"""

from __future__ import annotations
import re
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


# -------------------------------------------------------------------------
# Numeric parsing
# -------------------------------------------------------------------------

# Matches plain numbers, decimals, scientific notation, with optional sign.
# Examples it accepts: "0", "-3", "3.14", "1.2e-5", "+0.05"
# Examples it rejects: "20℃", "AmpHrs", "1O" (letter O), "3..5"
_NUMBER_RE = re.compile(
    r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"
)

# Common OCR confusions we forgive ONLY for strings that already look
# mostly numeric. We don't apply these blindly because that would let
# "lO" become "10" everywhere.
_OCR_FIXUPS = {
    "O": "0",
    "o": "0",
    "l": "1",
    "I": "1",
    "S": "5",   # only if surrounded by digits
}


def _looks_numeric(s: str) -> bool:
    """Heuristic: at least half the characters are digit-like or known OCR
    confusables for digits (O/o/l/I/S)."""
    if not s:
        return False
    digit_like = sum(
        c.isdigit() or c in ".-+eE" or c in _OCR_FIXUPS
        for c in s
    )
    return digit_like >= max(1, len(s) // 2)


def _try_parse_number(raw: str) -> Optional[float]:
    """Try to parse a string as a number, with mild OCR-confusion repair.

    Returns the float value or None if it doesn't parse.
    """
    s = raw.strip()
    if not s:
        return None

    # Strip surrounding quotes, commas (thousands separator), whitespace
    s = s.replace(",", "").replace(" ", "")

    # Direct attempt
    if _NUMBER_RE.match(s):
        try:
            return float(s)
        except ValueError:
            pass

    # OCR repair attempt - only if it already looks roughly numeric
    if _looks_numeric(s):
        repaired = "".join(_OCR_FIXUPS.get(c, c) for c in s)
        if _NUMBER_RE.match(repaired):
            try:
                return float(repaired)
            except ValueError:
                pass

    return None


# -------------------------------------------------------------------------
# Data classes
# -------------------------------------------------------------------------

@dataclass
class TickLabel:
    """A candidate axis tick label that has passed geometry + numeric filters."""
    value: float           # parsed numeric value
    raw_text: str          # original OCR string
    pixel: float           # position along the axis (x for X-axis, y for Y-axis)
    box: Tuple[int, int, int, int]  # (x, y, w, h) of label in image


@dataclass
class AxisFit:
    """Result of fitting a calibration line through tick labels."""
    is_log: bool
    # For linear: value = a * pixel + b
    # For log:    log10(value) = a * pixel + b
    slope: float
    intercept: float
    rms_residual_px: float   # quality of fit, in pixels
    n_ticks_used: int
    n_ticks_dropped: int

    def pixel_to_value(self, px: float) -> float:
        v = self.slope * px + self.intercept
        return 10 ** v if self.is_log else v

    def value_to_pixel(self, value: float) -> float:
        v = math.log10(value) if self.is_log else value
        return (v - self.intercept) / self.slope


@dataclass
class ExtractionResult:
    """Everything the extractor knows about the axis after running."""
    axis_config: Optional[dict] = None  # The dict shape your app already uses
    x_fit: Optional[AxisFit] = None
    y_fit: Optional[AxisFit] = None
    x_labels: List[TickLabel] = field(default_factory=list)
    y_labels: List[TickLabel] = field(default_factory=list)
    plot_area: Optional[Tuple[int, int, int, int]] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def confidence(self) -> str:
        """Coarse confidence label based on residuals + tick counts."""
        if self.x_fit is None or self.y_fit is None:
            return "failed"
        worst_rms = max(self.x_fit.rms_residual_px, self.y_fit.rms_residual_px)
        min_ticks = min(self.x_fit.n_ticks_used, self.y_fit.n_ticks_used)
        if min_ticks < 2:
            return "failed"
        if min_ticks >= 3 and worst_rms < 2.0:
            return "high"
        if min_ticks >= 2 and worst_rms < 5.0:
            return "medium"
        return "low"


# -------------------------------------------------------------------------
# The extractor
# -------------------------------------------------------------------------

class SmartAxisExtractor:
    """Extract axis calibration from a chart image, robust to legends/annotations.

    Parameters
    ----------
    x_band_height_frac : float
        Vertical thickness of the X-tick-label search band, as a fraction
        of plot area height. Default 0.20 means we look in a strip below
        the plot up to 20% of the plot height tall. Tick labels are
        usually within ~5-10% of the bottom edge.
    y_band_width_frac : float
        Same idea for Y-tick labels, fraction of plot width.
    x_band_h_margin_px : int
        How far horizontally outside the plot area to still accept X-tick
        labels (e.g. the rightmost X label might overhang slightly).
    y_band_v_margin_px : int
        Same idea vertically for Y-tick labels.
    max_residual_drop_px : float
        During fitting, drop labels whose residual exceeds this many pixels.
    min_ticks : int
        Minimum number of valid tick labels required to fit an axis.
    """

    def __init__(
        self,
        x_band_height_frac: float = 0.20,
        y_band_width_frac: float = 0.20,
        x_band_h_margin_px: int = 20,
        y_band_v_margin_px: int = 10,
        max_residual_drop_px: float = 8.0,
        min_ticks: int = 2,
    ):
        self.x_band_height_frac = x_band_height_frac
        self.y_band_width_frac = y_band_width_frac
        self.x_band_h_margin_px = x_band_h_margin_px
        self.y_band_v_margin_px = y_band_v_margin_px
        self.max_residual_drop_px = max_residual_drop_px
        self.min_ticks = min_ticks

    # ---- Geometry: define where tick labels are allowed to live ----------

    def _x_band(self, plot_area):
        """(x0, y0, x1, y1) box where X-tick labels can be."""
        px0, py0, px1, py1 = plot_area
        plot_h = py1 - py0
        band_h = max(20, int(plot_h * self.x_band_height_frac))
        return (
            px0 - self.x_band_h_margin_px,
            py1,
            px1 + self.x_band_h_margin_px,
            py1 + band_h,
        )

    def _y_band(self, plot_area):
        """(x0, y0, x1, y1) box where Y-tick labels can be."""
        px0, py0, px1, py1 = plot_area
        plot_w = px1 - px0
        band_w = max(30, int(plot_w * self.y_band_width_frac))
        return (
            px0 - band_w,
            py0 - self.y_band_v_margin_px,
            px0,
            py1 + self.y_band_v_margin_px,
        )

    @staticmethod
    def _box_center_in(box, region) -> bool:
        """True if box (x,y,w,h) center falls inside region (x0,y0,x1,y1)."""
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        rx0, ry0, rx1, ry1 = region
        return rx0 <= cx <= rx1 and ry0 <= cy <= ry1

    # ---- The main entry point --------------------------------------------

    def extract(
        self,
        img: np.ndarray,
        plot_area: Tuple[int, int, int, int],
        candidate_labels: Optional[List[Tuple[str, Tuple[int, int, int, int]]]] = None,
        ocr_fn: Optional[Callable] = None,
    ) -> ExtractionResult:
        """Extract X and Y axis calibration.

        Parameters
        ----------
        img : the chart image (BGR or grayscale ndarray)
        plot_area : (x0, y0, x1, y1) of the plot's drawing region
        candidate_labels : pre-detected text boxes with OCR strings, optional.
            Each entry is (text, (x, y, w, h)). If you have these from
            ChartDete + OCR, pass them in and we'll filter geometrically.
        ocr_fn : callable to OCR a cropped image region, optional.
            Signature: ocr_fn(crop_ndarray) -> List[(text, (x,y,w,h))]
            where boxes are in the *crop's* coordinates (we offset back).
            Used as a fallback if `candidate_labels` is None or empty.

        Returns
        -------
        ExtractionResult with axis_config dict, fits, and diagnostics.
        """
        result = ExtractionResult(plot_area=plot_area)

        if plot_area is None:
            result.warnings.append("No plot_area provided; cannot calibrate.")
            return result

        x_band = self._x_band(plot_area)
        y_band = self._y_band(plot_area)

        # ---- Collect candidates --------------------------------------
        labels = list(candidate_labels) if candidate_labels else []

        # If we have nothing, fall back to OCR'ing the band crops
        if not labels and ocr_fn is not None:
            labels.extend(self._ocr_band(img, x_band, ocr_fn))
            labels.extend(self._ocr_band(img, y_band, ocr_fn))

        if not labels:
            result.warnings.append(
                "No text candidates found. Provide candidate_labels or ocr_fn."
            )
            return result

        # ---- Filter to X band, parse, project onto X axis ------------
        x_ticks: List[TickLabel] = []
        for text, box in labels:
            if not self._box_center_in(box, x_band):
                continue
            val = _try_parse_number(text)
            if val is None:
                continue
            x, y, w, h = box
            x_ticks.append(TickLabel(
                value=val, raw_text=text, pixel=float(x + w / 2), box=box,
            ))

        # ---- Filter to Y band, parse, project onto Y axis ------------
        y_ticks: List[TickLabel] = []
        for text, box in labels:
            if not self._box_center_in(box, y_band):
                continue
            val = _try_parse_number(text)
            if val is None:
                continue
            x, y, w, h = box
            y_ticks.append(TickLabel(
                value=val, raw_text=text, pixel=float(y + h / 2), box=box,
            ))

        # Deduplicate (same value at nearly same pixel can be detected twice)
        x_ticks = self._dedupe(x_ticks)
        y_ticks = self._dedupe(y_ticks)

        result.x_labels = x_ticks
        result.y_labels = y_ticks

        if len(x_ticks) < self.min_ticks:
            result.warnings.append(
                f"Only {len(x_ticks)} X-tick label(s) found in the X band. "
                f"Need at least {self.min_ticks}."
            )
        if len(y_ticks) < self.min_ticks:
            result.warnings.append(
                f"Only {len(y_ticks)} Y-tick label(s) found in the Y band. "
                f"Need at least {self.min_ticks}."
            )

        # ---- Fit each axis --------------------------------------------
        if len(x_ticks) >= self.min_ticks:
            result.x_fit = self._fit_axis(x_ticks)
        if len(y_ticks) >= self.min_ticks:
            result.y_fit = self._fit_axis(y_ticks)

        # ---- Build the axis_config dict your app expects --------------
        if result.x_fit and result.y_fit:
            result.axis_config = self._build_axis_config(
                result.x_fit, result.y_fit, plot_area,
            )
        else:
            result.warnings.append("Calibration incomplete; axis_config is None.")

        return result

    # ---- OCR helper (only used if you didn't pre-supply boxes) -----------

    def _ocr_band(self, img, band, ocr_fn) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """Run OCR on a band of the image and return labels in image coords."""
        h, w = img.shape[:2]
        x0 = max(0, int(band[0]))
        y0 = max(0, int(band[1]))
        x1 = min(w, int(band[2]))
        y1 = min(h, int(band[3]))
        if x1 <= x0 or y1 <= y0:
            return []
        crop = img[y0:y1, x0:x1]
        out = []
        for text, box in ocr_fn(crop):
            bx, by, bw, bh = box
            out.append((text, (bx + x0, by + y0, bw, bh)))
        return out

    # ---- Dedupe near-duplicate detections --------------------------------

    @staticmethod
    def _dedupe(ticks: List[TickLabel], tol_px: float = 4.0) -> List[TickLabel]:
        """Drop ticks with the same value within tol_px of each other."""
        ticks = sorted(ticks, key=lambda t: (t.value, t.pixel))
        kept: List[TickLabel] = []
        for t in ticks:
            if any(abs(t.pixel - k.pixel) < tol_px and t.value == k.value for k in kept):
                continue
            kept.append(t)
        return kept

    # ---- Linear vs log fitting -------------------------------------------

    def _fit_axis(self, ticks: List[TickLabel]) -> AxisFit:
        """Fit the best linear or log calibration through tick labels.

        Strategy:
          1. Try a linear fit (value vs pixel) via least squares.
          2. If all values are positive, also try log fit (log10(value) vs pixel).
          3. Keep whichever has lower RMS residual *in pixel space*
             (so the comparison is fair across scales).
          4. Drop outliers whose residual exceeds max_residual_drop_px and refit.
        """
        # Linear fit
        lin = self._fit_linear(ticks, log=False)
        # Log fit (only if all values are positive)
        log = None
        if all(t.value > 0 for t in ticks):
            log = self._fit_linear(ticks, log=True)

        # Pick the better one by pixel-space residual
        best = lin
        if log is not None and log.rms_residual_px < lin.rms_residual_px:
            best = log

        # Outlier rejection: drop the worst tick if it's beyond threshold, refit
        refined_ticks = list(ticks)
        dropped = 0
        while len(refined_ticks) > self.min_ticks:
            residuals = self._tick_residuals(refined_ticks, best)
            worst_i = int(np.argmax(np.abs(residuals)))
            if abs(residuals[worst_i]) <= self.max_residual_drop_px:
                break
            refined_ticks.pop(worst_i)
            dropped += 1
            # Refit
            best = self._fit_linear(refined_ticks, log=best.is_log)

        best.n_ticks_used = len(refined_ticks)
        best.n_ticks_dropped = dropped
        return best

    @staticmethod
    def _fit_linear(ticks: List[TickLabel], log: bool) -> AxisFit:
        """Least-squares fit value (or log10 value) vs pixel."""
        px = np.array([t.pixel for t in ticks], dtype=float)
        if log:
            vals = np.array([math.log10(t.value) for t in ticks], dtype=float)
        else:
            vals = np.array([t.value for t in ticks], dtype=float)

        # Fit vals = slope * px + intercept
        A = np.vstack([px, np.ones_like(px)]).T
        slope, intercept = np.linalg.lstsq(A, vals, rcond=None)[0]

        # Pixel-space residual: convert back through value_to_pixel
        # to get a fair comparison between linear and log fits.
        fit = AxisFit(
            is_log=log, slope=slope, intercept=intercept,
            rms_residual_px=0.0, n_ticks_used=len(ticks), n_ticks_dropped=0,
        )
        residuals = SmartAxisExtractor._tick_residuals(ticks, fit)
        fit.rms_residual_px = float(np.sqrt(np.mean(residuals ** 2)))
        return fit

    @staticmethod
    def _tick_residuals(ticks: List[TickLabel], fit: AxisFit) -> np.ndarray:
        """Residual of each tick in pixel space (label_px - predicted_px)."""
        out = np.zeros(len(ticks))
        for i, t in enumerate(ticks):
            predicted_px = fit.value_to_pixel(t.value)
            out[i] = t.pixel - predicted_px
        return out

    # ---- Build the axis_config dict your app expects ---------------------

    @staticmethod
    def _build_axis_config(x_fit: AxisFit, y_fit: AxisFit, plot_area) -> dict:
        """Produce the same dict shape `detect_axis_calibration` returns.

        We synthesize two X anchor points and two Y anchor points from the
        fitted line. We pick the plot edges as anchors so the calibration
        is well-conditioned and the on-image markers look natural.
        """
        px0, py0, px1, py1 = plot_area

        # X anchors: left and right edges of plot area
        x1_px, x2_px = float(px0), float(px1)
        x1_val = x_fit.pixel_to_value(x1_px)
        x2_val = x_fit.pixel_to_value(x2_px)
        x_calib_y = float(py1)  # bottom edge

        # Y anchors: bottom and top edges of plot area
        y1_py, y2_py = float(py1), float(py0)
        y1_val = y_fit.pixel_to_value(y1_py)
        y2_val = y_fit.pixel_to_value(y2_py)
        y_calib_x = float(px0)  # left edge

        return {
            "x1_px": x1_px, "x1_py": x_calib_y, "x1_val": _round_pretty(x1_val),
            "x2_px": x2_px, "x2_py": x_calib_y, "x2_val": _round_pretty(x2_val),
            "y1_px": y_calib_x, "y1_py": y1_py, "y1_val": _round_pretty(y1_val),
            "y2_px": y_calib_x, "y2_py": y2_py, "y2_val": _round_pretty(y2_val),
            "xIsLogScale": x_fit.is_log,
            "yIsLogScale": y_fit.is_log,
        }


def _round_pretty(v: float) -> float:
    """Round to a reasonable number of sig figs for display."""
    if v == 0:
        return 0.0
    mag = math.floor(math.log10(abs(v)))
    # Keep ~4 sig figs
    digits = max(0, 3 - mag)
    return round(v, digits)
