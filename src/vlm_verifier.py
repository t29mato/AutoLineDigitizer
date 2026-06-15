# -*- coding: utf-8 -*-
"""
vlm_verifier.py — focused gap-fill + stray-remove via Claude.
Sends original chart + numbered overlay (with px grid) to Claude.
Returns corrections: missing points to add, stray points to remove.
No merging, no swaps — keep it simple and safe.
"""
import os, io, json, base64
import cv2, numpy as np

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

DEFAULT_MODEL = "claude-opus-4-8"

_SYSTEM_PROMPT = (
    "You audit chart-line extractions. You compare the original chart against "
    "an overlay of detected points and report ONLY two kinds of fixes: "
    "(1) missing points to add where the real curve has no markers, "
    "(2) stray points to remove where markers sit OFF the curve. "
    "You output ONLY a single JSON object — no markdown fences, no extra text."
)


class VLMVerifier:
    def __init__(self, api_key=None, model=DEFAULT_MODEL, max_tokens=4096,
                 verify_ssl=True, remove_tolerance_px=15):
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError("Run: pip install anthropic")
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY env var or pass api_key=...")
        if verify_ssl:
            self.client = anthropic.Anthropic(api_key=key)
        else:
            import httpx
            self.client = anthropic.Anthropic(
                api_key=key, http_client=httpx.Client(verify=False))
        self.model = model
        self.max_tokens = max_tokens
        self.remove_tolerance_px = remove_tolerance_px

    def verify_and_correct(self, original_img, data_series, axis_config=None,
                           colors=None, grid_step=100):
        h, w = original_img.shape[:2]
        overlay = self._draw_grid_overlay(
            original_img, data_series, axis_config, colors=colors, step=grid_step)
        original_b64 = self._encode_png(original_img)
        overlay_b64 = self._encode_png(overlay)
        prompt = self._build_prompt(w, h, data_series, colors)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "IMAGE 1 — original chart:"},
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/png",
                                "data": original_b64}},
                    {"type": "text",
                     "text": "IMAGE 2 — current extraction overlay "
                             "(numbered lines + gray coordinate grid):"},
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/png",
                                "data": overlay_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        raw_text = "".join(b.text for b in message.content if b.type == "text")
        parsed = self._parse_response(raw_text)
        corrected = self._apply_corrections(data_series, parsed, w, h)
        return corrected, {"raw": raw_text, "parsed": parsed}

    def verify_axis_calibration(self, img, plot_area, axis_config=None, model=None):
        """
        Read the axis tick labels with the VLM and return corrected calibration.

        OCR-based calibration mis-reads scientific / exponential ticks
        (10^-4, ×10^3) and log scales. This sends the chart with the detected
        plot box outlined and asks the model to read the ticks properly and
        report, per axis, whether it is log scale and a list of ticks as
        {value, frac} where frac is the position ALONG the axis inside the plot
        box (x: 0=left, 1=right; y: 0=bottom, 1=top).

        Returns the parsed dict:
          {"x_axis": {"is_log": bool, "ticks": [{"value", "frac"}, ...]},
           "y_axis": {...}, "notes": "..."}
        Raises on API/parse failure (caller handles).
        """
        px0, py0, px1, py1 = [float(v) for v in plot_area]
        annotated = img.copy()
        cv2.rectangle(annotated, (int(px0), int(py0)), (int(px1), int(py1)),
                      (255, 0, 255), 2)
        # Light fractional guide lines (0, .25, .5, .75, 1) to anchor `frac`.
        for f in (0.25, 0.5, 0.75):
            gx = int(px0 + f * (px1 - px0))
            gy = int(py1 - f * (py1 - py0))
            cv2.line(annotated, (gx, int(py0)), (gx, int(py1)), (255, 0, 255), 1)
            cv2.line(annotated, (int(px0), gy), (int(px1), gy), (255, 0, 255), 1)

        b64 = self._encode_png(annotated)
        cur = ""
        if axis_config:
            cur = (f"\nThe current (possibly WRONG) calibration is: "
                   f"x in [{axis_config.get('x1_val')}, {axis_config.get('x2_val')}] "
                   f"(log={axis_config.get('xIsLogScale')}), "
                   f"y in [{axis_config.get('y1_val')}, {axis_config.get('y2_val')}] "
                   f"(log={axis_config.get('yIsLogScale')}). Correct it if wrong.")
        prompt = self._build_axis_prompt() + cur

        message = self.client.messages.create(
            model=model or self.model,
            max_tokens=1500,
            system=("You read chart AXIS calibrations precisely. You handle "
                    "scientific/exponential notation and log scales. You output "
                    "ONLY a single JSON object — no markdown fences, no prose."),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/png",
                                "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        raw_text = "".join(b.text for b in message.content if b.type == "text")
        return self._parse_response(raw_text)

    def label_lines_by_legend(self, img, data_series, colors=None, model=None):
        """
        Read the chart's legend and assign each detected line its series label.

        Sends the original chart (with its legend) + an overlay where each
        detected line is drawn as numbered colored markers tracing the real
        curves. The model matches each numbered series to the legend entry of
        the curve it follows. Returns a list of labels aligned to data_series
        (None where there is no clear legend match). Raises on API/parse failure.
        """
        n = len(data_series)
        if n == 0:
            return []
        if colors is None:
            colors = self._fallback_colors(n)
        overlay = self._draw_numbered_lines(img, data_series, colors)
        original_b64 = self._encode_png(img)
        overlay_b64 = self._encode_png(overlay)
        desc = []
        for idx in range(n):
            b, g, r = colors[idx % len(colors)]
            desc.append(f"  line {idx}: overlay markers in RGB({int(r)},{int(g)},{int(b)})")
        prompt = self._build_label_prompt(n, "\n".join(desc))

        message = self.client.messages.create(
            model=model or self.model,
            max_tokens=1200,
            system=("You match extracted chart curves to their legend labels by "
                    "reading the chart. You output ONLY a single JSON object — no "
                    "markdown fences, no prose."),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "IMAGE 1 — original chart (read its legend here):"},
                    {"type": "image", "source": {"type": "base64",
                        "media_type": "image/png", "data": original_b64}},
                    {"type": "text", "text": "IMAGE 2 — same chart, each detected line drawn "
                                             "as numbered colored markers over the real curves:"},
                    {"type": "image", "source": {"type": "base64",
                        "media_type": "image/png", "data": overlay_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        raw = "".join(b.text for b in message.content if b.type == "text")
        parsed = self._parse_response(raw)
        labels = [None] * n
        for item in (parsed.get("labels") or []):
            try:
                i = int(item.get("id"))
            except (TypeError, ValueError, AttributeError):
                continue
            if 0 <= i < n:
                lab = item.get("label")
                lab = str(lab).strip() if lab is not None else ""
                labels[i] = lab or None
        return labels

    def _draw_numbered_lines(self, img, data_series, colors):
        out = img.copy()
        for idx, series in enumerate(data_series):
            color = tuple(int(c) for c in colors[idx % len(colors)])
            pts = series.get("points", [])
            for px, py in pts:
                cv2.drawMarker(out, (int(px), int(py)), color,
                               cv2.MARKER_CROSS, markerSize=9, thickness=2)
            if pts:
                lx, ly = min(pts, key=lambda p: p[0])
                self._label(out, str(idx), (int(lx) - 6, int(ly) - 8), color)
        return out

    @staticmethod
    def _build_label_prompt(n, color_desc):
        return (
            f"This chart has {n} detected data series. In IMAGE 2 each is drawn as "
            f"colored markers labeled with a number (0..{n - 1}), tracing over the "
            f"real curves in IMAGE 1.\n\n"
            f"Overlay colors:\n{color_desc}\n\n"
            f"TASK: read the chart's LEGEND in IMAGE 1 and give each numbered series "
            f"the legend label (series name) of the REAL curve it traces. Match by the "
            f"underlying curve's OWN color, line style, and position — NOT by the "
            f"overlay marker color. Copy the legend text concisely and verbatim "
            f"(keep units/symbols). If there is no legend, or a series has no clear "
            f"match, set its label to null.\n\n"
            f"Output ONLY this JSON object:\n"
            f'{{"labels": [{{"id": 0, "label": "<legend text or null>"}}, ...]}}'
        )

    @staticmethod
    def _build_axis_prompt():
        return (
            "The magenta rectangle outlines the PLOT AREA. Thin magenta lines mark "
            "the 25%, 50%, 75% positions inside it to help you judge positions.\n\n"
            "Read the X and Y axis tick labels on this chart. Be careful with "
            "exponential / scientific notation and log scales:\n"
            "- A tick written as 10^-4 (or 1E-4, ×10⁻⁴) has VALUE 0.0001.\n"
            "- If the axis has a shared multiplier like '×10³' near the end, fold it "
            "into every tick value (a tick '2' under '×10³' has value 2000).\n"
            "- Decide log vs linear from the tick SPACING (equal spacing of powers of "
            "ten ⇒ log).\n\n"
            "For each axis return its tick labels as {value, frac}, where `value` is "
            "the true numeric value (multiplier folded in) and `frac` is the tick's "
            "position ALONG the axis INSIDE the plot rectangle:\n"
            "  X axis: frac = 0.0 at the LEFT edge, 1.0 at the RIGHT edge.\n"
            "  Y axis: frac = 0.0 at the BOTTOM edge, 1.0 at the TOP edge.\n"
            "Give at least the two outermost ticks per axis; more is better. Estimate "
            "`frac` against the magenta guide lines as accurately as you can.\n\n"
            "Output ONLY this JSON object:\n"
            "{\n"
            '  "x_axis": {"is_log": true|false, "ticks": [{"value": <num>, "frac": <0..1>}, ...]},\n'
            '  "y_axis": {"is_log": true|false, "ticks": [{"value": <num>, "frac": <0..1>}, ...]},\n'
            '  "notes": "<short note on what you corrected, e.g. log scale / 10^-4 ticks>"\n'
            "}"
        )

    @staticmethod
    def _encode_png(img_bgr):
        ok, buf = cv2.imencode(".png", img_bgr)
        return base64.b64encode(buf).decode("utf-8")

    def _draw_grid_overlay(self, img, data_series, axis_config, colors=None,
                           step=100):
        out = img.copy()
        h, w = out.shape[:2]
        grid_color = (210, 210, 210)
        label_color = (90, 90, 90)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for x in range(0, w, step):
            cv2.line(out, (x, 0), (x, h), grid_color, 1)
            cv2.putText(out, str(x), (x + 2, 14), font, 0.4, label_color, 1,
                        cv2.LINE_AA)
        for y in range(0, h, step):
            cv2.line(out, (0, y), (w, y), grid_color, 1)
            cv2.putText(out, str(y), (2, y - 2), font, 0.4, label_color, 1,
                        cv2.LINE_AA)
        if colors is None:
            colors = self._fallback_colors(len(data_series))
        for idx, series in enumerate(data_series):
            color = tuple(int(c) for c in colors[idx % len(colors)])
            pts = series.get("points", [])
            for px, py in pts:
                cv2.drawMarker(out, (int(px), int(py)), color,
                               cv2.MARKER_CROSS, markerSize=8, thickness=2)
            if pts:
                lx, ly = min(pts, key=lambda p: p[0])
                self._label(out, str(idx), (int(lx) - 6, int(ly) - 8), color)
        return out

    @staticmethod
    def _label(img, text, pos, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), base = cv2.getTextSize(text, font, 0.6, 2)
        x, y = pos
        cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + base + 2),
                      (255, 255, 255), -1)
        cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + base + 2),
                      (0, 0, 0), 1)
        cv2.putText(img, text, (x, y), font, 0.6, color, 2, cv2.LINE_AA)

    @staticmethod
    def _fallback_colors(n):
        rng = np.linspace(0, 179, max(n, 1), dtype=np.uint8)
        hsv = np.stack([rng, np.full_like(rng, 220), np.full_like(rng, 255)], 1)
        bgr = cv2.cvtColor(hsv.reshape(-1, 1, 3),
                           cv2.COLOR_HSV2BGR).reshape(-1, 3)
        return [tuple(int(c) for c in row) for row in bgr]

    def _build_prompt(self, width, height, data_series, colors):
        if colors is None:
            colors = self._fallback_colors(len(data_series))
        lines_summary = []
        for idx, series in enumerate(data_series):
            pts = series.get("points", [])
            b, g, r = colors[idx % len(colors)]
            if pts:
                sx, sy = pts[0]; ex, ey = pts[-1]
                span = (f"{len(pts)} pts, from ({int(sx)},{int(sy)}) "
                        f"to ({int(ex)},{int(ey)})")
            else:
                span = "0 pts"
            lines_summary.append(
                f"  line_id {idx}: marker RGB({int(r)},{int(g)},{int(b)}) — {span}")
        summary = "\n".join(lines_summary) if lines_summary else "  (none)"

        schema = (
            '{\n'
            '  "overall_assessment": "one short sentence",\n'
            '  "corrections": [\n'
            '    {\n'
            '      "line_id": <int>,\n'
            '      "add_points": [[x, y], ...],\n'
            '      "remove_points": [[x, y], ...],\n'
            '      "note": "short reason"\n'
            '    }\n'
            '  ]\n'
            '}'
        )

        return (
            f"I am auditing an automated extraction of curve data from this chart.\n\n"
            f"IMAGE 1 = original chart. IMAGE 2 = same chart with the current "
            f"extraction drawn on top: each detected line is colored markers "
            f"labeled with its line_id, over a gray pixel grid (every 100 px, "
            f"labeled along top/left edges).\n\n"
            f"COORDINATE SYSTEM: image is {width}x{height} px. Origin (0,0) is "
            f"TOP-LEFT. x increases right, y increases DOWNWARD. Read coords "
            f"against the gray grid labels.\n\n"
            f"Current extraction:\n{summary}\n\n"
            f"YOUR TASK — ONLY TWO THINGS:\n\n"
            f"1. ADD missing points: for each line, look at the real curve in "
            f"IMAGE 1 that this line is tracing (match by color/position). If "
            f"there is a PART OF THAT SAME CURVE that has no marker on it in "
            f"IMAGE 2 — at the start, middle (gaps), or end (dropped tails) — "
            f"add points along the missing part. List points in order, roughly "
            f"every 15-25 px apart, following the curve's shape.\n\n"
            f"2. REMOVE stray points: any marker in IMAGE 2 that sits clearly "
            f"OFF the real curve (in white background, on a different curve, "
            f"on an axis/label, etc.) — flag it for removal.\n\n"
            f"RULES — read carefully:\n"
            f"- DO NOT merge or split lines. Each line_id stays a separate line.\n"
            f"- DO NOT reassign points to different line_ids. Just add/remove on "
            f"the existing line.\n"
            f"- DO NOT add new lines (no \"new_lines\" — only existing line_ids).\n"
            f"- Coordinates are integer [x, y] in IMAGE 1's pixel space "
            f"(0..{width}, 0..{height}), origin top-left.\n"
            f"- Use [] when nothing to add/remove for a line.\n"
            f"- If a line is already correct, emit it with empty arrays.\n"
            f"- Be CONSERVATIVE on removals — only flag points that are "
            f"obviously, clearly stray. When in doubt, leave them.\n\n"
            f"Output ONLY this JSON object:\n{schema}"
        )

    @staticmethod
    def _parse_response(text):
        s = text.strip()
        if s.startswith("```"):
            s = s.split("```", 2)
            s = s[1] if len(s) >= 2 else text
            if s.lstrip().lower().startswith("json"):
                s = s.lstrip()[4:]
            s = s.strip()
        if not s.startswith("{"):
            start, end = s.find("{"), s.rfind("}")
            if start == -1 or end == -1:
                raise ValueError(f"No JSON object found in VLM response:\n{text[:500]}")
            s = s[start:end + 1]
        return json.loads(s)

    def _apply_corrections(self, data_series, parsed, w, h):
        """Apply add/remove only. No merging, no reassignment, no new lines."""
        def clamp(p):
            return [int(np.clip(p[0], 0, w - 1)),
                    int(np.clip(p[1], 0, h - 1))]

        out = [{"points": [list(p) for p in s.get("points", [])]}
               for s in data_series]

        for corr in parsed.get("corrections", []):
            lid = corr.get("line_id")
            if not isinstance(lid, int) or not (0 <= lid < len(out)):
                continue
            pts = out[lid]["points"]
            # REMOVE strays (nearest point within tolerance).
            for rp in corr.get("remove_points", []) or []:
                if len(rp) < 2 or not pts: continue
                rpc = clamp(rp)
                arr = np.asarray(pts, dtype=float)
                d = np.hypot(arr[:, 0] - rpc[0], arr[:, 1] - rpc[1])
                j = int(np.argmin(d))
                if d[j] <= self.remove_tolerance_px:
                    pts.pop(j)
            # ADD missing points.
            for ap in corr.get("add_points", []) or []:
                if len(ap) >= 2:
                    pts.append(clamp(ap))
            # Sort along x for clean rendering.
            pts.sort(key=lambda p: (p[0], p[1]))

        # IMPORTANT: do NOT drop empty lines (preserve original count).
        # Only protect: if a line is left empty by removals, keep it empty
        # but don't delete it — preserves line_id stability.
        return out
