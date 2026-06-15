# -*- coding: utf-8 -*-
"""
vlm_screener.py — VLM-based figure pre-filter.

Wraps Claude (vision) with a tight JSON-only screening prompt. Given any image
extracted from a paper, returns a structured judgement on whether it's worth
sending through the chart-data extraction pipeline:

    {
      "extractable": true | false,
      "chart_type": "line_chart" | "scatter" | "bar" | "heatmap" | ...,
      "n_panels": int,
      "has_axes": bool,
      "axis_labels_readable": bool,
      "n_lines_estimate": int | null,
      "is_log_scale_x": bool | null,
      "is_log_scale_y": bool | null,
      "reason": str
    }

Used by batch_extract.py to skip schematics, photos, equations, multi-panel
composites, and figures with unreadable axes before the heavier LineFormer +
ChartDete pipeline runs.

API: requires ANTHROPIC_API_KEY env var. Uses claude-haiku-4-5 by default
(fast and cheap: a few tenths of a cent per image).
"""

import os
import io
import json
import base64

import cv2

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except Exception:
    anthropic = None
    ANTHROPIC_AVAILABLE = False


# Cheap, fast vision model. Override per-call via the model= argument.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

SCREEN_PROMPT = """\
You screen figures from scientific papers for an automated chart-data
extraction pipeline. Reply with ONE JSON object only. No prose, no markdown
fences, no commentary.

Schema (all keys required):
{
  "extractable": true | false,
  "chart_type": "line_chart" | "scatter" | "bar" | "heatmap" | "contour" | "schematic" | "photo" | "table" | "equation" | "multi_panel" | "mixed" | "other",
  "n_panels": <integer, at least 1>,
  "has_axes": true | false,
  "axis_labels_readable": true | false,
  "n_lines_estimate": <integer or null>,
  "is_log_scale_x": true | false | null,
  "is_log_scale_y": true | false | null,
  "reason": "<one short sentence>"
}

Set "extractable": true ONLY if ALL of these hold:
- It is a SINGLE-panel line chart or scatter plot with continuous data
- The axes are clearly visible with readable numeric tick labels
- At least one data series (line, curve, or dense scatter) is present
- It is NOT a schematic, photograph, equation, microscopy image, table,
  heatmap, contour map, bar chart with categorical X axis, or multi-panel
  composite (a)(b)(c)(d) figure

If the figure shows multiple sub-panels (e.g., (a), (b), (c)),
set extractable=false and chart_type="multi_panel".

If axes exist but tick labels are too small or blurry to read with confidence,
set extractable=false and axis_labels_readable=false.

Return JSON only.
"""


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` fences if the model wrapped them around the JSON."""
    t = text.strip()
    if t.startswith("```"):
        # Drop the opening fence line
        if "\n" in t:
            t = t.split("\n", 1)[1]
        # Drop the closing fence
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3].rstrip()
    return t.strip()


class VLMScreener:
    """Vision-model screener that judges if a figure is worth extracting."""

    def __init__(self, api_key=None, model=DEFAULT_MODEL, max_image_dim=1600):
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError(
                "The 'anthropic' package is required for VLM screening.\n"
                "Install it with:  pip install anthropic"
            )
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set.\n"
                "Set it with:  export ANTHROPIC_API_KEY='sk-ant-...'"
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_image_dim = max_image_dim

    def _encode_image(self, img_bgr):
        """Resize (if needed) and PNG-encode the image to base64."""
        h, w = img_bgr.shape[:2]
        m = max(h, w)
        if m > self.max_image_dim:
            scale = self.max_image_dim / float(m)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            raise RuntimeError("Failed to PNG-encode the screening image.")
        return base64.standard_b64encode(buf.tobytes()).decode("ascii")

    def screen(self, img_bgr):
        """
        Run the screening prompt on a single figure.

        Returns a dict with the schema documented at the top of this module.
        On any API/parse failure, returns a permissive default with
        extractable=True so a transient error does not silently drop figures.
        The failure mode is recorded in _screening_error.
        """
        result = {
            "extractable": True,
            "chart_type": "unknown",
            "n_panels": 1,
            "has_axes": False,
            "axis_labels_readable": False,
            "n_lines_estimate": None,
            "is_log_scale_x": None,
            "is_log_scale_y": None,
            "reason": "",
            "_model": self.model,
        }
        try:
            data_b64 = self._encode_image(img_bgr)
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": data_b64,
                            },
                        },
                        {"type": "text", "text": SCREEN_PROMPT},
                    ],
                }],
            )
            # Concatenate any text blocks in the response.
            text_parts = []
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    text_parts.append(block.text)
            raw = "".join(text_parts)
            cleaned = _strip_fences(raw)
            parsed = json.loads(cleaned)
            # Merge over the defaults so any missing keys keep sane values.
            for k in result.keys():
                if k in parsed:
                    result[k] = parsed[k]
            return result
        except Exception as e:
            result["_screening_error"] = f"{type(e).__name__}: {e}"
            result["reason"] = "screening failed; default permissive"
            return result


def quick_test(image_path):
    """Smoke test: screen a single image and pretty-print the result."""
    import sys
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        sys.exit(1)
    screener = VLMScreener()
    out = screener.screen(img)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vlm_screener.py <image.png>")
        sys.exit(1)
    quick_test(sys.argv[1])


# ============================================================================
# Page-render extension (added for --page-render mode).
# Monkey-patches screen_page() onto VLMScreener.
# ============================================================================

PAGE_SCREEN_PROMPT = """\
You are looking at a single rendered page from a scientific paper. Your job
is to locate EVERY extractable line chart or scatter plot on the page so it
can be cropped and sent to an automated data-extraction pipeline.

Reply with ONE JSON object only. No prose, no markdown fences.

Schema:
{
  "n_charts": <integer>,
  "charts": [
    {
      "bbox_norm": [x0, y0, x1, y1],
      "chart_type": "line_chart" | "scatter" | "bar" | "heatmap" | "schematic" | "photo" | "other",
      "extractable": true | false,
      "n_lines_estimate": <integer or null>,
      "axis_labels_readable": true | false,
      "figure_label": "<e.g. 'Fig 3a' or empty string>",
      "reason": "<short reason>"
    }
  ]
}

Bounding-box rules (READ CAREFULLY):
- bbox_norm coordinates are NORMALIZED to [0, 1] relative to page width and height.
- (x0, y0) is the TOP-LEFT corner; (x1, y1) is the BOTTOM-RIGHT corner.
- Origin (0, 0) is at the TOP-LEFT of the page; (1, 1) is at the BOTTOM-RIGHT.
- Include the FULL chart area: plot region + axis tick labels + axis titles + legend.
- EXCLUDE the figure caption text below ("Figure 3 | Discharge rate ...").
- Add a small margin (about 1-2% of the page) so axis labels are not clipped.

Inclusion rules:
- A multi-panel figure (e.g. Fig 1a / 1b / 1c) -> return EACH sub-panel that
  contains a line chart or scatter as a SEPARATE entry. Sub-panels that are
  microscopy / schematic / photograph -> skip (do not include).
- Charts on the same page but in different figures -> return each one separately.
- Set extractable=true ONLY for line charts or scatter plots with continuous
  numeric data, readable axes, and at least one data series.
- Set extractable=false for: bar charts with categorical X, heatmaps, contour
  maps, schematics, photographs, micrographs, equations, tables, or charts
  with unreadable axis labels.

If the page has no chart at all, return {"n_charts": 0, "charts": []}.

Return JSON only.
"""


def _screen_page_method(self, page_img_bgr, model=None):
    """Find ALL extractable chart regions on a rendered PDF page."""
    result = {"n_charts": 0, "charts": [], "_model": model or self.model}
    try:
        data_b64 = self._encode_image(page_img_bgr)
        resp = self.client.messages.create(
            model=model or self.model,
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": data_b64,
                    }},
                    {"type": "text", "text": PAGE_SCREEN_PROMPT},
                ],
            }],
        )
        text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        raw = "".join(text_parts)
        cleaned = _strip_fences(raw)
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            charts = parsed.get("charts", [])
            clean_charts = []
            if isinstance(charts, list):
                for c in charts:
                    if not isinstance(c, dict):
                        continue
                    bbox = c.get("bbox_norm", [0, 0, 1, 1])
                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        continue
                    try:
                        bbox = [float(v) for v in bbox]
                    except Exception:
                        continue
                    bbox = [max(0.0, min(1.0, v)) for v in bbox]
                    x0, y0, x1, y1 = bbox
                    if x1 <= x0 or y1 <= y0:
                        continue
                    clean_charts.append({
                        "bbox_norm": [x0, y0, x1, y1],
                        "chart_type": c.get("chart_type", "unknown"),
                        "extractable": bool(c.get("extractable", False)),
                        "n_lines_estimate": c.get("n_lines_estimate"),
                        "axis_labels_readable": bool(c.get("axis_labels_readable", False)),
                        "figure_label": str(c.get("figure_label", "")).strip(),
                        "reason": str(c.get("reason", "")).strip(),
                    })
            result["charts"] = clean_charts
            result["n_charts"] = len(clean_charts)
        return result
    except Exception as e:
        result["_screening_error"] = f"{type(e).__name__}: {e}"
        return result


# Attach as a method on the VLMScreener class.
VLMScreener.screen_page = _screen_page_method


# ============================================================================
# Figure-refinement agent (added for reliable cropping).
# Reviews the boxes proposed by screen_page by DRAWING them back onto the page
# (numbered) and asking a stronger model to correct them: tighten each box to
# exactly one chart, reject regions that aren't real data charts, split merged
# panels, and add any missed chart. Visual grounding (the model sees its own
# proposals on the image) is far more reliable than re-asking for coordinates.
# ============================================================================

REFINE_MODEL = "claude-sonnet-4-6"   # box precision benefits from the stronger model

REFINE_PAGE_PROMPT = """\
You are reviewing the output of an automatic chart detector on a rendered page
from a scientific paper. The page image has {n} candidate region(s) drawn on it
as numbered RED rectangles. These proposals are frequently WRONG: a box may clip
part of a chart, include neighbouring text or a second panel, outline something
that is not a data chart at all, or merge several distinct charts into one.

Your job: return the CORRECT set of bounding boxes so each one can be cropped and
sent to a line/scatter data-extraction pipeline. Reply with ONE JSON object only,
no prose, no markdown fences.

Schema:
{{
  "charts": [
    {{
      "bbox_norm": [x0, y0, x1, y1],
      "chart_type": "line_chart" | "scatter" | "bar" | "heatmap" | "schematic" | "photo" | "other",
      "extractable": true | false,
      "figure_label": "<e.g. 'Fig 3a' or empty string>",
      "action": "kept" | "tightened" | "split" | "added" | "rejected",
      "reason": "<short reason>"
    }}
  ]
}}

Bounding-box rules (READ CAREFULLY):
- Coordinates are NORMALIZED to [0, 1]. (x0, y0) = TOP-LEFT, (x1, y1) = BOTTOM-RIGHT,
  origin at the TOP-LEFT of the page.
- Each output box must enclose EXACTLY ONE distinct chart: its plot area + axis
  tick labels + axis titles + legend, with a small ~1% margin so labels are not
  clipped.
- EXCLUDE, by tightening the box edges: the figure caption text; ANY neighbouring
  sub-panel (especially adjacent photos / SEM / TEM / microscopy / schematic
  images — these are NOT part of the chart even if they share the figure); the
  running page header / footer / journal banner (e.g. "NATURE | Vol 458 | ...")
  and page numbers; and surrounding body text. Pull each edge in until it touches
  only this one chart's own plot, axes, labels and legend.
- If two panels sit side by side (e.g. an XRD plot 'a' next to an SEM image 'b'),
  the box for the chart must stop at the gutter between them — never include the
  neighbour.

Correction rules:
- If a numbered box is too tight or too loose, return the corrected tight box (action "tightened").
- If a numbered box is NOT a real, extractable data chart (text, schematic, photo,
  micrograph, table, bar/categorical, a partial fragment, or just whitespace),
  set extractable=false (action "rejected"). You may omit rejected regions entirely.
- If a numbered box contains MULTIPLE distinct charts/panels, output one tight box
  per chart (action "split").
- If a real extractable chart on the page has NO box around it, add it (action "added").
- extractable=true ONLY for line charts / scatter plots with continuous numeric data,
  readable axes, and at least one data series.

Return ONLY the charts you believe are genuinely extractable plus any you reject
(for transparency). If the page truly has no extractable chart, return
{{"charts": []}}. Return JSON only.
"""


def _refine_page_charts_method(self, page_img_bgr, candidate_charts, model=None):
    """
    Review + correct the chart bboxes proposed by screen_page for one page.

    Draws the candidates as numbered red boxes on a copy of the page and asks a
    stronger model to return corrected boxes. Returns a dict with the same
    "charts" shape as screen_page. On any failure, falls back to the original
    candidates (never worse than the input) and records the error.
    """
    use_model = model or REFINE_MODEL
    result = {"charts": list(candidate_charts or []), "_model": use_model, "_refined": False}
    if not candidate_charts:
        return result

    H, W = page_img_bgr.shape[:2]
    annotated = page_img_bgr.copy()
    for i, c in enumerate(candidate_charts, start=1):
        bb = c.get("bbox_norm")
        if not bb or len(bb) != 4:
            continue
        x0, y0 = int(bb[0] * W), int(bb[1] * H)
        x1, y1 = int(bb[2] * W), int(bb[3] * H)
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 255), 3)
        label_y = y0 + 34 if y0 < 34 else y0 - 8
        cv2.putText(annotated, str(i), (x0 + 4, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)

    try:
        data_b64 = self._encode_image(annotated)
        resp = self.client.messages.create(
            model=use_model,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": data_b64,
                    }},
                    {"type": "text",
                     "text": REFINE_PAGE_PROMPT.format(n=len(candidate_charts))},
                ],
            }],
        )
        raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        parsed = json.loads(_strip_fences(raw))
        if not isinstance(parsed, dict):
            return result
        clean = []
        for c in parsed.get("charts", []):
            if not isinstance(c, dict):
                continue
            bbox = c.get("bbox_norm")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                bbox = [max(0.0, min(1.0, float(v))) for v in bbox]
            except Exception:
                continue
            x0, y0, x1, y1 = bbox
            if x1 <= x0 or y1 <= y0:
                continue
            clean.append({
                "bbox_norm": [x0, y0, x1, y1],
                "chart_type": c.get("chart_type", "unknown"),
                "extractable": bool(c.get("extractable", False)),
                "n_lines_estimate": c.get("n_lines_estimate"),
                "axis_labels_readable": bool(c.get("axis_labels_readable", False)),
                "figure_label": str(c.get("figure_label", "")).strip(),
                "reason": str(c.get("reason", "")).strip(),
                "action": str(c.get("action", "")).strip(),
            })
        # Only accept the refined set if it parsed to a list (possibly empty =
        # "all proposals were false positives"). Keep the flag for callers/logs.
        result["charts"] = clean
        result["_refined"] = True
        return result
    except Exception as e:
        result["_refine_error"] = f"{type(e).__name__}: {e}"
        return result


VLMScreener.refine_page_charts = _refine_page_charts_method
