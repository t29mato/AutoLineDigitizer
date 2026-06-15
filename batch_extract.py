# -*- coding: utf-8 -*-
"""
batch_extract.py — Cross-paper batch chart extraction.

Takes a folder of PDFs, finds figures in each, runs the AutoLineDigitizer
pipeline (LineFormer + ChartDete axis detection + color refinement), and
writes axis-calibrated data plus full provenance.

Output layout:
    <output>/
        aggregate.csv            # long-format: every (paper, figure, line, point)
        summary.json             # per-paper extraction stats
        <paper_id>/
            fig_p003_i1.png          # cropped figure
            fig_p003_i1_overlay.png  # extraction drawn on top
            fig_p003_i1.csv          # wide-format calibrated data
            fig_p003_i1.json         # axes, caption, confidence, model

Usage:
    python batch_extract.py --input papers/ --output extractions/
    python batch_extract.py --input papers/ --output out/ --model general_v2 --require-axis
    python batch_extract.py --input papers/ --output out/ --limit-pdfs 5 --limit-charts 10 --verbose

Requires PyMuPDF:  pip install pymupdf
"""

import argparse
import os
import sys
import io
import re
import json
import csv
from datetime import datetime, timezone
from pathlib import Path

# Make local imports resolve (same dir as desktop_app.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF is required. Install it with:\n    pip install pymupdf")
    sys.exit(1)

# Reuse the full pipeline from the desktop app. Importing this module does NOT
# launch the GUI (that's guarded by __main__), it only brings in the model
# wiring, sys.path setup, and the LineFormerApp class.
try:
    from desktop_app import (
        LineFormerApp,
        LINEFORMER_MODELS,
        get_models_dir,
        check_models_exist,
        download_model,
        MODEL_FILES,
        CHARTDETE_AVAILABLE,
    )
except Exception as e:
    print(f"ERROR: could not import pipeline from desktop_app.py: {e}")
    print("Run this script from the AutoLineDigitizer folder (next to desktop_app.py).")
    sys.exit(1)

# Optional VLM screener (--vlm-filter). Imported defensively so the script
# still runs end-to-end if anthropic isn't installed or the module is missing.
try:
    from vlm_screener import VLMScreener, ANTHROPIC_AVAILABLE as _VLM_SDK_AVAILABLE
    VLM_SCREENER_AVAILABLE = True
except Exception as _vlm_err:
    VLMScreener = None
    _VLM_SDK_AVAILABLE = False
    VLM_SCREENER_AVAILABLE = False
    _VLM_IMPORT_ERROR = str(_vlm_err)


# DOI regex (Crossref's recommended pattern).
DOI_PATTERN = re.compile(r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b', re.IGNORECASE)
# Figure caption line opener, e.g. "Fig. 3", "Figure 12b".
FIGURE_PATTERN = re.compile(r'^(fig(?:ure)?\.?\s*\d+[a-z]?)', re.IGNORECASE)


def safe_paper_id(pdf_path):
    """Sanitize a PDF filename into a filesystem-safe folder name."""
    stem = Path(pdf_path).stem
    return re.sub(r'[^A-Za-z0-9_-]+', '_', stem)[:80] or "paper"


def try_extract_doi(pdf_path):
    """Look for a DOI in the first few pages of a PDF. Returns '' if none."""
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""
    doi = ""
    try:
        # Check embedded metadata first.
        meta = doc.metadata or {}
        for v in meta.values():
            if v and isinstance(v, str):
                m = DOI_PATTERN.search(v)
                if m:
                    doi = m.group(0)
                    break
        # Then scan first 3 pages of text.
        if not doi:
            for page in doc[:3]:
                m = DOI_PATTERN.search(page.get_text())
                if m:
                    doi = m.group(0).rstrip('.')
                    break
    except Exception:
        pass
    finally:
        doc.close()
    return doi


def extract_figures(pdf_path, min_size=200, min_aspect=0.3, max_aspect=4.0):
    """
    Yield (image_bgr, meta) for each plausible raster figure in a PDF.

    Filters by minimum pixel size and aspect ratio to skip logos, icons,
    rules, and tiny inline glyphs. Tries to attach the nearest "Fig N..."
    caption found below the image on the same page.

    Note: only raster (embedded bitmap) figures are extracted. Pure-vector
    figures (some matplotlib-to-PDF exports) won't appear here; those need a
    page-render-and-crop step, which is out of scope for this MVP.
    """
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow is required. Install it with: pip install pillow")
        sys.exit(1)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"    [open fail] {e}")
        return

    try:
        for page_idx, page in enumerate(doc, start=1):
            try:
                image_list = page.get_images(full=True)
            except Exception:
                image_list = []
            if not image_list:
                continue

            try:
                text_blocks = page.get_text("blocks")
            except Exception:
                text_blocks = []

            for img_idx, img_info in enumerate(image_list, start=1):
                xref = img_info[0]
                try:
                    base = doc.extract_image(xref)
                except Exception:
                    continue
                img_bytes = base.get("image")
                ext = base.get("ext", "png")
                if not img_bytes:
                    continue
                try:
                    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception:
                    continue

                w, h = pil.size
                if w < min_size or h < min_size:
                    continue
                ar = w / max(1, h)
                if ar < min_aspect or ar > max_aspect:
                    continue

                # Locate this image's bbox on the page (for caption matching).
                bbox = None
                try:
                    rects = page.get_image_rects(xref)
                    if rects:
                        bbox = rects[0]
                except Exception:
                    bbox = None

                caption = ""
                if bbox is not None and text_blocks:
                    candidates = []
                    for tb in text_blocks:
                        if len(tb) < 5:
                            continue
                        x0, y0, x1, y1, text = tb[0], tb[1], tb[2], tb[3], tb[4]
                        # text block starting just below the image
                        if y0 >= bbox.y1 - 8:
                            t = (text or "").strip().replace("\n", " ")
                            if FIGURE_PATTERN.match(t):
                                candidates.append((y0 - bbox.y1, t))
                    if candidates:
                        candidates.sort(key=lambda c: abs(c[0]))
                        caption = candidates[0][1][:500]

                arr_rgb = np.array(pil)
                arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
                yield arr_bgr, {
                    "page": page_idx,
                    "img_idx_on_page": img_idx,
                    "width": int(w),
                    "height": int(h),
                    "caption": caption,
                    "format": ext,
                }
    finally:
        doc.close()


def extract_figures_via_page_render(pdf_path, screener, dpi=200, min_size=200,
                                     page_vlm_model=None, verbose=False):
    """
    Render each PDF page and use the VLM to find chart bounding boxes,
    yielding cropped chart images. This is the path for papers whose figures
    are vector PDF graphics (no embedded bitmap), where extract_figures()
    misses everything.

    For each page:
      1. Render at the requested DPI
      2. Send the page image to screener.screen_page() to get a list of
         normalised chart bboxes
      3. For each extractable chart, crop the page image and yield it with
         provenance metadata (page index, bbox, VLM verdict, etc.)

    Yields (image_bgr, meta) where meta also contains "vlm_chart" (the page-
    level VLM judgement on this region) so the caller can attach it to the
    final per-figure metadata.
    """
    if screener is None:
        return  # Page-render mode requires a VLM screener.

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"    [open fail] {e}")
        return

    zoom = dpi / 72.0  # PDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    try:
        for page_idx, page in enumerate(doc, start=1):
            try:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
            except Exception as e:
                if verbose:
                    print(f"      [render fail page {page_idx}] {e}")
                continue

            # Convert the pixmap to a BGR numpy array.
            try:
                img_arr = np.frombuffer(pix.samples, dtype=np.uint8)
                img_arr = img_arr.reshape(pix.height, pix.width, pix.n)
            except Exception as e:
                if verbose:
                    print(f"      [decode fail page {page_idx}] {e}")
                continue

            if pix.n == 4:
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            elif pix.n == 1:
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
            else:
                if verbose:
                    print(f"      [unsupported pixmap channels page {page_idx}: n={pix.n}]")
                continue
            page_img_bgr = np.ascontiguousarray(img_arr)
            H, W = page_img_bgr.shape[:2]

            # Ask the VLM which chart regions are on this page.
            print(f"      page {page_idx} ({W}x{H} px) -> VLM page screen ... ",
                  end="", flush=True)
            try:
                verdict = screener.screen_page(page_img_bgr, model=page_vlm_model)
            except Exception as e:
                print(f"VLM error: {e}")
                continue

            charts = verdict.get("charts", [])
            n_extractable = sum(1 for c in charts if c.get("extractable", False))
            print(f"found {len(charts)} chart region(s), {n_extractable} extractable")

            if verbose and "_screening_error" in verdict:
                print(f"        (page screen error: {verdict['_screening_error']})")

            chart_seq = 0
            for chart in charts:
                if not chart.get("extractable", False):
                    if verbose:
                        ct = chart.get("chart_type", "?")
                        lbl = chart.get("figure_label", "")
                        reason = chart.get("reason", "")[:80]
                        print(f"        skip region [{lbl} {ct}]: {reason}")
                    continue
                bbox = chart.get("bbox_norm")
                if not bbox or len(bbox) != 4:
                    continue
                x0 = max(0, int(round(bbox[0] * W)))
                y0 = max(0, int(round(bbox[1] * H)))
                x1 = min(W, int(round(bbox[2] * W)))
                y1 = min(H, int(round(bbox[3] * H)))
                # FIX: expand bbox to guarantee axis labels are included
                # (Sonnet often crops too tight to plot area, clipping tick labels)
                margin_x = int(0.02 * W)   # 2% — for y-axis labels on left
                margin_y = int(0.015 * H)  # 1.5% — for x-axis labels on bottom
                x0 = max(0, x0 - margin_x)
                y0 = max(0, y0 - margin_y // 2)
                x1 = min(W, x1 + margin_x // 2)
                y1 = min(H, y1 + margin_y)
                if x1 - x0 < min_size or y1 - y0 < min_size:
                    if verbose:
                        lbl = chart.get("figure_label", "")
                        print(f"        skip region [{lbl}]: too small ({x1-x0}x{y1-y0})")
                    continue
                crop = page_img_bgr[y0:y1, x0:x1].copy()
                chart_seq += 1
                yield crop, {
                    "page": page_idx,
                    "img_idx_on_page": chart_seq,
                    "width": int(x1 - x0),
                    "height": int(y1 - y0),
                    "caption": chart.get("figure_label", ""),
                    "format": "page-render-crop",
                    "render_dpi": dpi,
                    "bbox_norm": [float(v) for v in bbox],
                    "bbox_px": [x0, y0, x1, y1],
                    "page_size_px": [W, H],
                    "vlm_chart": chart,
                }
    finally:
        doc.close()


def process_figure(app, img_bgr, fig_meta, out_dir, fig_basename, args, screener=None):
    """Run the extraction pipeline on a single figure image. Returns a result dict."""
    result = {
        "page": fig_meta["page"],
        "img_idx_on_page": fig_meta["img_idx_on_page"],
        "caption": fig_meta["caption"],
        "width": fig_meta["width"],
        "height": fig_meta["height"],
        "n_lines": 0,
        "n_points": 0,
        "axis_detected": False,
        "x_axis_name": "",
        "y_axis_name": "",
        "error": "",
        "files": {},
        "vlm_screen": None,
    }

    try:
        # 0) Optional VLM pre-filter. Decides if this image is even a chart.
        if screener is not None:
            verdict = screener.screen(img_bgr)
            result["vlm_screen"] = verdict
            if not verdict.get("extractable", True):
                ct = verdict.get("chart_type", "unknown")
                reason = verdict.get("reason", "")
                result["error"] = f"vlm: not extractable ({ct}) — {reason}"
                return result

        # Fresh state for this figure.
        app.current_image = img_bgr
        app.current_image_path = None
        app.cached_plot_area = None
        app.axis_config = None
        app.ocr_results = None
        app.raw_lines = None
        app.data_series = None

        # 1) Axis detection (also acts as a "is this a chart?" filter).
        if app.auto_axis and app.chartdete_module is not None:
            try:
                app.axis_config, app.ocr_results = app.detect_axis_calibration(img_bgr)
            except Exception as e:
                if args.verbose:
                    print(f"      [axis fail] {e}")

        if app.axis_config is None and args.require_axis:
            result["error"] = "no axis detected"
            return result

        result["axis_detected"] = app.axis_config is not None
        if app.axis_config is not None:
            x_name, y_name = app.get_axis_titles()
            result["x_axis_name"] = x_name
            result["y_axis_name"] = y_name

        # FIX: ChartDete's detect_axis_calibration sets cached_plot_area
        # as a side effect; that crop can clip curves at plot edges (e.g. low-V
        # tails). axis_config is preserved separately, so we reset here.
        app.cached_plot_area = None

        # 2) Line extraction (full image; axis info kept in app.axis_config).
        app.data_series = app.extract_lines(img_bgr)
        if not app.data_series:
            result["error"] = "no lines detected"
            return result

        result["n_lines"] = len(app.data_series)
        result["n_points"] = sum(len(s["points"]) for s in app.data_series)

        # 3) Save cropped figure + overlay.
        img_path = out_dir / f"{fig_basename}.png"
        cv2.imwrite(str(img_path), img_bgr)
        result["files"]["image"] = img_path.name

        overlay = app.draw_points_on_image(img_bgr, app.data_series, app.axis_config)
        overlay_path = out_dir / f"{fig_basename}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)
        result["files"]["overlay"] = overlay_path.name

        # 4) Save wide-format calibrated CSV.
        x_name = result["x_axis_name"] or "X"
        y_name = result["y_axis_name"] or "Y"
        csv_str = app.to_wide_csv(x_name=x_name, y_name=y_name)
        csv_path = out_dir / f"{fig_basename}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(csv_str)
        result["files"]["csv"] = csv_path.name

        # 5) Save per-figure metadata.
        meta = {k: v for k, v in result.items() if k != "files"}
        meta["axis_config"] = app.axis_config
        meta["model_key"] = app.current_model_key
        meta["extracted_at"] = datetime.now(timezone.utc).isoformat()
        meta_path = out_dir / f"{fig_basename}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, default=str, indent=2)
        result["files"]["meta"] = meta_path.name

        # 6) Per-point rows for the aggregate CSV.
        rows = []
        for line_idx, series in enumerate(app.data_series):
            for pt_idx, pt in enumerate(series["points"]):
                xv, yv = app.pixel_to_data(pt[0], pt[1])
                rows.append((line_idx, pt_idx, xv, yv, int(pt[0]), int(pt[1])))
        result["_aggregate_rows"] = rows

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch chart-data extraction from a folder of PDFs."
    )
    parser.add_argument("--input", required=True, help="Folder containing PDF files")
    parser.add_argument("--output", required=True, help="Output folder for results")
    parser.add_argument("--model", default="general_v2",
                        choices=list(LINEFORMER_MODELS.keys()),
                        help="LineFormer model variant (default: general_v2)")
    parser.add_argument("--limit-pdfs", type=int, default=None,
                        help="Process at most N PDFs")
    parser.add_argument("--limit-charts", type=int, default=None,
                        help="Process at most N figures per PDF")
    parser.add_argument("--require-axis", action="store_true",
                        help="Skip figures with no detected axis (chart filter)")
    parser.add_argument("--vlm-filter", action="store_true",
                        help="Use Claude to pre-screen figures and skip non-extractable ones "
                             "(schematics, photos, multi-panel composites, unreadable axes). "
                             "Requires ANTHROPIC_API_KEY.")
    parser.add_argument("--vlm-model", default="claude-haiku-4-5-20251001",
                        help="Anthropic model id for screening (default: claude-haiku-4-5-20251001)")
    parser.add_argument("--page-render", action="store_true",
                        help="Render each PDF page and use VLM to find chart bounding boxes. "
                             "Required for papers with vector PDF graphics (e.g. Nature, Cell). "
                             "Implies --vlm-filter.")
    parser.add_argument("--page-dpi", type=int, default=200,
                        help="DPI for page rendering in --page-render mode (default 200)")
    parser.add_argument("--page-vlm-model", default=None,
                        help="Override Anthropic model for page-level bbox detection. "
                             "Defaults to --vlm-model. Sonnet may give more accurate bboxes.")
    parser.add_argument("--no-color-refinement", action="store_true",
                        help="Disable color-based line refinement; use raw LineFormer output only "
                             "(ablation switch).")
    parser.add_argument("--min-size", type=int, default=200,
                        help="Minimum figure width/height in pixels (default 200)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    # --page-render mode requires the VLM screener; enable --vlm-filter automatically.
    if args.page_render and not args.vlm_filter:
        print("Note: --page-render requires VLM; enabling --vlm-filter automatically.")
        args.vlm_filter = True

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.is_dir():
        print(f"ERROR: input folder not found: {input_dir}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob("*.pdf")) + sorted(input_dir.glob("*.PDF"))
    pdfs = sorted(set(pdfs))
    if not pdfs:
        print(f"No PDFs found in {input_dir}")
        sys.exit(1)
    if args.limit_pdfs:
        pdfs = pdfs[:args.limit_pdfs]

    print(f"Found {len(pdfs)} PDF(s) to process.\n")

    # ---- Load models ----
    print("[1/4] Loading models...")
    missing = check_models_exist()
    if missing:
        print(f"  Missing base models: {missing} — downloading from GitHub...")
        models_dir = get_models_dir()
        for fname in missing:
            print(f"    downloading {fname} ...")
            try:
                download_model(fname, models_dir)
            except Exception as e:
                print(f"    ERROR downloading {fname}: {e}")
                sys.exit(1)

    app = LineFormerApp()
    if args.no_color_refinement:
        app.use_color_refinement = False
        print("  color refinement DISABLED (--no-color-refinement)")
    print(f"  loading LineFormer ({args.model}) ...")
    try:
        app.load_lineformer_model(args.model)
    except Exception as e:
        print(f"  ERROR loading LineFormer: {e}")
        sys.exit(1)

    if CHARTDETE_AVAILABLE:
        print("  loading ChartDete ...")
        try:
            app.load_chartdete_model()
        except Exception as e:
            print(f"  WARNING: ChartDete failed ({e}); axis detection disabled.")
            app.auto_axis = False
    else:
        print("  ChartDete not available; axis detection disabled.")
        app.auto_axis = False

    # Optional VLM screener.
    screener = None
    if args.vlm_filter:
        if not VLM_SCREENER_AVAILABLE:
            print(f"  WARNING: --vlm-filter requested but vlm_screener could not be loaded:")
            print(f"           {_VLM_IMPORT_ERROR if 'VLM_IMPORT_ERROR' in dir() else 'unknown'}")
            print(f"           Continuing WITHOUT VLM filter.")
        elif not _VLM_SDK_AVAILABLE:
            print(f"  WARNING: --vlm-filter requested but anthropic SDK not installed.")
            print(f"           Install with:  pip install anthropic")
            print(f"           Continuing WITHOUT VLM filter.")
        else:
            try:
                print(f"  initializing VLM screener ({args.vlm_model}) ...")
                screener = VLMScreener(model=args.vlm_model)
                print("  VLM screener ready.")
            except Exception as e:
                print(f"  WARNING: VLM screener init failed ({e}); continuing without filter.")
                screener = None

    print("  models ready.\n")

    # ---- Process ----
    aggregate_rows = []
    summary = {
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "require_axis": args.require_axis,
        "n_pdfs": len(pdfs),
        "pdfs": [],
    }
    n_total_figures = 0
    n_extracted = 0
    n_failed = 0

    print("[2/4] Processing PDFs...\n")
    for pdf_idx, pdf_path in enumerate(pdfs, start=1):
        paper_id = safe_paper_id(pdf_path)
        paper_dir = output_dir / paper_id
        paper_dir.mkdir(exist_ok=True)
        doi = try_extract_doi(str(pdf_path))

        print(f"  [{pdf_idx}/{len(pdfs)}] {pdf_path.name}" + (f"   DOI: {doi}" if doi else ""))

        pdf_summary = {
            "paper_id": paper_id,
            "source_pdf": pdf_path.name,
            "doi": doi,
            "n_figures_seen": 0,
            "n_figures_extracted": 0,
            "n_figures_failed": 0,
            "figures": [],
        }

        figures_seen = 0

        # Pick extraction strategy: page-render (VLM bbox detection, vector-aware)
        # or raster (embedded bitmaps via PyMuPDF). Page-render is the path for
        # papers whose figures are stored as vector graphics, not raster.
        if args.page_render and screener is not None:
            fig_iter = extract_figures_via_page_render(
                str(pdf_path), screener,
                dpi=args.page_dpi,
                min_size=args.min_size,
                page_vlm_model=args.page_vlm_model,
                verbose=args.verbose,
            )
            pre_screened = True
        else:
            if args.page_render and screener is None:
                print("      (page-render needs VLM screener; falling back to raster.)")
            fig_iter = extract_figures(str(pdf_path), min_size=args.min_size)
            pre_screened = False

        for fig_idx, (img_bgr, fig_meta) in enumerate(fig_iter, start=1):
            if args.limit_charts and figures_seen >= args.limit_charts:
                break
            figures_seen += 1
            n_total_figures += 1

            fig_basename = f"fig_p{fig_meta['page']:03d}_i{fig_meta['img_idx_on_page']}"
            print(f"      fig {fig_idx} (p{fig_meta['page']}, {fig_meta['width']}x{fig_meta['height']}) ... ",
                  end="", flush=True)

            # In page-render mode the chart was already screened at page level,
            # so don't re-screen each crop.
            per_fig_screener = None if pre_screened else screener
            result = process_figure(app, img_bgr, fig_meta, paper_dir, fig_basename, args,
                                     screener=per_fig_screener)

            # Lift the page-level VLM verdict into result.vlm_screen so it
            # flows into per-figure metadata and the aggregate stats.
            if pre_screened and fig_meta.get("vlm_chart"):
                vc = fig_meta["vlm_chart"]
                result["vlm_screen"] = {
                    "extractable": vc.get("extractable", True),
                    "chart_type": vc.get("chart_type", "unknown"),
                    "n_panels": 1,
                    "axis_labels_readable": vc.get("axis_labels_readable", False),
                    "n_lines_estimate": vc.get("n_lines_estimate"),
                    "figure_label": vc.get("figure_label", ""),
                    "reason": vc.get("reason", ""),
                    "bbox_norm": fig_meta.get("bbox_norm"),
                    "_source": "page-render",
                }

            if result["error"]:
                n_failed += 1
                pdf_summary["n_figures_failed"] += 1
                print(f"skip ({result['error']})")
            else:
                n_extracted += 1
                pdf_summary["n_figures_extracted"] += 1
                axis_flag = "axis OK" if result["axis_detected"] else "no axis"
                vlm_info = ""
                if result.get("vlm_screen"):
                    ct = result["vlm_screen"].get("chart_type", "?")
                    vlm_info = f", vlm:{ct}"
                print(f"OK  {result['n_lines']} lines, {result['n_points']} pts, {axis_flag}{vlm_info}")
                for (line_idx, pt_idx, xv, yv, px, py) in result.get("_aggregate_rows", []):
                    aggregate_rows.append({
                        "paper_id": paper_id,
                        "source_pdf": pdf_path.name,
                        "doi": doi,
                        "page": fig_meta["page"],
                        "figure_basename": fig_basename,
                        "caption": fig_meta["caption"],
                        "x_axis": result["x_axis_name"],
                        "y_axis": result["y_axis_name"],
                        "line_idx": line_idx,
                        "point_idx": pt_idx,
                        "x_value": xv,
                        "y_value": yv,
                        "x_pixel": px,
                        "y_pixel": py,
                    })

            result.pop("_aggregate_rows", None)
            pdf_summary["figures"].append({"basename": fig_basename, **result})

        pdf_summary["n_figures_seen"] = figures_seen
        summary["pdfs"].append(pdf_summary)
        if figures_seen == 0:
            print("      (no figures matched filters)")
        print()

    # ---- Aggregate CSV ----
    print(f"[3/4] Writing aggregate CSV ({len(aggregate_rows)} data points)...")
    agg_path = output_dir / "aggregate.csv"
    fieldnames = ["paper_id", "source_pdf", "doi", "page", "figure_basename", "caption",
                  "x_axis", "y_axis", "line_idx", "point_idx", "x_value", "y_value",
                  "x_pixel", "y_pixel"]
    with open(agg_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    print(f"  -> {agg_path}")

    # ---- Summary JSON ----
    summary["run_finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["n_figures_total"] = n_total_figures
    summary["n_figures_extracted"] = n_extracted
    summary["n_figures_failed"] = n_failed
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, default=str, indent=2)
    print(f"  -> {summary_path}")

    # ---- Final stats ----
    print("\n[4/4] Done.")
    print(f"  PDFs processed:   {len(pdfs)}")
    print(f"  Figures seen:     {n_total_figures}")
    print(f"  Extracted:        {n_extracted}")
    print(f"  Skipped/failed:   {n_failed}")
    if n_total_figures:
        print(f"  Success rate:     {100.0 * n_extracted / n_total_figures:.1f}%")
    print(f"  Output folder:    {output_dir}")


if __name__ == "__main__":
    main()