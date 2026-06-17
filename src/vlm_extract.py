# -*- coding: utf-8 -*-
"""
vlm_extract.py — VLM-only chart digitization with spline densification.

Asks Claude to identify each data curve in a chart and return ~N sparse points
along each curve (read against a labeled pixel grid we draw on the image), then
fits a parametric cubic B-spline through those points and resamples densely.
Writes a JSON file with both sparse and dense points, plus an overlay PNG so you
can eyeball quality.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python vlm_extract.py path/to/chart.png
    python vlm_extract.py chart.png --points-per-curve 25 --dense 200
    python vlm_extract.py chart.png --no-proxy --no-ssl-verify   # NIMS quirks
    python vlm_extract.py chart.png --model claude-sonnet-4-6    # cheaper
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from scipy.interpolate import splprep, splev
except ImportError:
    sys.exit("scipy required: python -m pip install scipy")

try:
    import anthropic
except ImportError:
    sys.exit("anthropic required: python -m pip install anthropic")


DEFAULT_MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = (
    "You are a careful chart-digitization assistant. Given a line chart with a "
    "labeled pixel-coordinate grid overlaid, you identify each distinct data "
    "curve and emit sparse points along each curve as integer pixel "
    "coordinates. You output ONLY a single JSON object: no markdown fences, no "
    "prose before or after."
)


# --------------------------------------------------------------------- imaging

def draw_grid(img, step=100):
    """Light gray pixel-coordinate grid with labels — gives Claude reference
    lines to read coordinates against instead of guessing freehand."""
    out = img.copy()
    h, w = out.shape[:2]
    grid = (210, 210, 210)
    label = (90, 90, 90)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), grid, 1)
        cv2.putText(out, str(x), (x + 2, 14), font, 0.4, label, 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), grid, 1)
        cv2.putText(out, str(y), (2, y - 2), font, 0.4, label, 1, cv2.LINE_AA)
    return out


def encode_png(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return base64.b64encode(buf).decode("utf-8")


def distinct_colors(n):
    if n <= 0:
        return []
    hues = np.linspace(0, 179, n, dtype=np.uint8)
    hsv = np.stack([hues, np.full_like(hues, 220), np.full_like(hues, 255)], 1)
    bgr = cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    return [tuple(int(c) for c in row) for row in bgr]


def draw_overlay(img, lines_dense, lines_sparse):
    """Overlay both Claude's sparse anchors (crosses) and the spline curve
    (polyline) so you can see how well the fit follows the underlying curve."""
    out = img.copy()
    colors = distinct_colors(len(lines_dense))
    for idx, (dense, sparse) in enumerate(zip(lines_dense, lines_sparse)):
        color = colors[idx]
        if len(dense) >= 2:
            pts = np.array(dense, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], False, color, 2, cv2.LINE_AA)
        for x, y in sparse:
            cv2.drawMarker(out, (int(x), int(y)), color,
                           cv2.MARKER_CROSS, 10, 2)
        if sparse:
            lx, ly = min(sparse, key=lambda p: p[0])
            cv2.putText(out, str(idx), (int(lx) - 6, int(ly) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------- prompt

def build_prompt(width, height, points_per_curve):
    schema = (
        '{\n'
        '  "summary": "one sentence describing the chart",\n'
        '  "lines": [\n'
        '    {\n'
        '      "label": "short descriptor, e.g. red curve / topmost line",\n'
        '      "points": [[x, y], [x, y], ...]\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    return (
        f"This is a line chart, {width}px wide by {height}px tall, with a "
        f"light gray pixel-coordinate grid overlaid. Grid lines are spaced "
        f"every 100 px and labeled with their pixel coordinate along the top "
        f"(x) and left (y) edges. Origin (0,0) is the TOP-LEFT corner; x "
        f"increases rightward, y increases DOWNWARD.\n\n"
        f"Identify every distinct data curve in the plot area. For each curve, "
        f"return approximately {points_per_curve} points along it as integer "
        f"pixel coordinates [x, y], in ORDER along the curve from one endpoint "
        f"to the other. Read coordinates against the gray grid labels.\n\n"
        f"Rules:\n"
        f"- Points must lie ON the curve, not floating off it.\n"
        f"- Distribute points along the WHOLE visible curve, including any "
        f"steep / near-vertical TAIL at the end. Do not stop early — the tail "
        f"is the most important part to cover.\n"
        f"- Order points consistently along the curve (typically left-to-right "
        f"through the body, then top-to-bottom along the vertical tail).\n"
        f"- IGNORE axis labels, legend entries, titles, gridlines, annotations "
        f"and curve labels. Only the actual data curves.\n"
        f"- Output ONLY this JSON, no markdown fences:\n{schema}"
    )


# ---------------------------------------------------------------------- parser

def parse_response(text):
    s = text.strip()
    if s.startswith("```"):
        parts = s.split("```")
        s = parts[1] if len(parts) >= 2 else text
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
        s = s.strip()
    if not s.startswith("{"):
        i, j = s.find("{"), s.rfind("}")
        if i == -1 or j == -1:
            raise ValueError(f"No JSON found in response:\n{text[:500]}")
        s = s[i:j + 1]
    return json.loads(s)


# ----------------------------------------------------------------- spline fit

def densify_with_spline(points, n_dense=150, smoothing=None):
    """Parametric cubic B-spline through sparse points, sampled at n_dense
    chord-length-uniform parameters. Handles the steep vertical tail correctly
    because it's parametric (x(t), y(t)) rather than y as a function of x.
    Falls back to linear interpolation if there are too few points or the
    spline fit fails.
    """
    pts = np.asarray(points, dtype=float)

    # Strip near-duplicates — splprep chokes on them.
    if len(pts) >= 2:
        keep = [0]
        for i in range(1, len(pts)):
            if not np.allclose(pts[i], pts[keep[-1]], atol=0.5):
                keep.append(i)
        pts = pts[keep]

    if len(pts) < 2:
        return [[int(round(p[0])), int(round(p[1]))] for p in pts]

    # Too few for a cubic spline → linear interpolate.
    if len(pts) < 4:
        return _linear_interp(pts, n_dense)

    s = smoothing if smoothing is not None else len(pts)
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=s, k=3)
        u = np.linspace(0, 1, n_dense)
        x, y = splev(u, tck)
        return [[int(round(xi)), int(round(yi))] for xi, yi in zip(x, y)]
    except Exception as e:
        print(f"    spline failed ({e}); falling back to linear")
        return _linear_interp(pts, n_dense)


def _linear_interp(pts, n_dense):
    t = np.linspace(0, 1, n_dense)
    u = np.linspace(0, 1, len(pts))
    x = np.interp(t, u, pts[:, 0])
    y = np.interp(t, u, pts[:, 1])
    return [[int(round(xi)), int(round(yi))] for xi, yi in zip(x, y)]


# ------------------------------------------------------------------------ cli

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("image", help="Path to chart image (png/jpg/etc).")
    ap.add_argument("--points-per-curve", type=int, default=20,
                    help="Sparse points Claude returns per curve (default 20).")
    ap.add_argument("--dense", type=int, default=150,
                    help="Dense points per curve after spline (default 150).")
    ap.add_argument("--smoothing", type=float, default=None,
                    help="splprep s param; default = len(sparse). Set 0 for "
                         "interpolating spline (passes through every point).")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Claude model (default {DEFAULT_MODEL}). Try "
                         f"claude-sonnet-4-6 to cut cost ~10x.")
    ap.add_argument("--no-ssl-verify", action="store_true",
                    help="Disable SSL verification (TLS-intercepting proxies).")
    ap.add_argument("--no-proxy", action="store_true",
                    help="Strip HTTP(S)_PROXY env vars for this call.")
    ap.add_argument("--grid-step", type=int, default=100,
                    help="Pixel grid spacing sent to Claude (default 100).")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: alongside the image).")
    args = ap.parse_args()

    img_path = Path(args.image).expanduser()
    if not img_path.exists():
        sys.exit(f"image not found: {img_path}")
    img = cv2.imread(str(img_path))
    if img is None:
        sys.exit(f"could not read image: {img_path}")
    h, w = img.shape[:2]
    print(f"[1/4] loaded {img_path.name} ({w}x{h})")

    if args.no_proxy:
        for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.pop(v, None)
        print("       proxy env vars stripped for this run")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set in environment")

    if args.no_ssl_verify:
        import httpx
        client = anthropic.Anthropic(
            api_key=api_key, http_client=httpx.Client(verify=False)
        )
    else:
        client = anthropic.Anthropic(api_key=api_key)

    print(f"[2/4] asking {args.model} for ~{args.points_per_curve} pts/curve...")
    overlay_for_vlm = draw_grid(img, step=args.grid_step)
    msg = client.messages.create(
        model=args.model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/png",
                            "data": encode_png(overlay_for_vlm)}},
                {"type": "text",
                 "text": build_prompt(w, h, args.points_per_curve)},
            ],
        }],
    )
    raw = "".join(b.text for b in msg.content if b.type == "text")
    parsed = parse_response(raw)
    print(f"       Claude: {parsed.get('summary', '(no summary)')}")

    sparse_lines = []
    labels = []
    for line in parsed.get("lines", []):
        pts = [[int(np.clip(p[0], 0, w - 1)), int(np.clip(p[1], 0, h - 1))]
               for p in (line.get("points") or []) if len(p) >= 2]
        if pts:
            sparse_lines.append(pts)
            labels.append(line.get("label", ""))

    if not sparse_lines:
        sys.exit("Claude returned no usable lines. Raw response:\n" + raw[:1000])

    print(f"[3/4] densifying {len(sparse_lines)} curves -> {args.dense} pts each (cubic B-spline)...")
    dense_lines = [
        densify_with_spline(line, n_dense=args.dense, smoothing=args.smoothing)
        for line in sparse_lines
    ]

    out_dir = Path(args.out).expanduser() if args.out else img_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    json_path = out_dir / f"{stem}_vlm_extract.json"
    png_path = out_dir / f"{stem}_vlm_overlay.png"

    payload = {
        "image": str(img_path),
        "width": w, "height": h,
        "model": args.model,
        "points_per_curve_requested": args.points_per_curve,
        "dense_per_curve": args.dense,
        "summary": parsed.get("summary", ""),
        "lines": [
            {"label": labels[i],
             "sparse": sparse_lines[i],
             "dense": dense_lines[i]}
            for i in range(len(sparse_lines))
        ],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    overlay_img = draw_overlay(img, dense_lines, sparse_lines)
    cv2.imwrite(str(png_path), overlay_img)

    print(f"[4/4] wrote:")
    print(f"       {json_path}")
    print(f"       {png_path}")
    for i, (s, d, lbl) in enumerate(zip(sparse_lines, dense_lines, labels)):
        tag = f" — {lbl}" if lbl else ""
        print(f"       line {i}: {len(s)} sparse -> {len(d)} dense{tag}")


if __name__ == "__main__":
    main()