# -*- coding: utf-8 -*-
"""
pdf_figures.py — Pull chart figures out of PDFs.

Single source of truth for PDF figure extraction, shared by both the batch
CLI (batch_extract.py) and the desktop GUI (desktop_app.py). Keeping these
functions here (rather than in batch_extract.py, which imports desktop_app)
avoids a circular import when the GUI wants to reuse them.

Two strategies:
  - extract_figures():                 embedded raster bitmaps (no API key)
  - extract_figures_via_page_render():  render pages + VLM bbox detection
                                        (needed for vector-only figures)

Plus render_pdf_page() for re-cropping a figure from its source page.

Requires PyMuPDF:  pip install pymupdf
"""

import io
import os
import re
import sys

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


# Figure caption line opener, e.g. "Fig. 3", "Figure 12b".
FIGURE_PATTERN = re.compile(r'^(fig(?:ure)?\.?\s*\d+[a-z]?)', re.IGNORECASE)


def _require_fitz():
    if fitz is None:
        print("ERROR: PyMuPDF is required. Install it with:\n    pip install pymupdf")
        raise ImportError("PyMuPDF (fitz) not installed")


def snap_bbox_to_content(img, x0, y0, x1, y1, W, H,
                         expand_frac=0.015, pad_px=6, white_thresh=244):
    """
    Tighten a *rough* VLM bounding box to the chart's real extent.

    The VLM only locates a chart approximately — boxes clip axis tick labels or
    carry extra margin. This pads the box a little (so labels/legends are never
    clipped), then trims the near-blank borders so the crop snaps to the actual
    inked content. Returns a corrected (x0, y0, x1, y1); falls back to the input
    on any degeneracy. No model calls.
    """
    bw, bh = x1 - x0, y1 - y0
    if bw <= 1 or bh <= 1:
        return x0, y0, x1, y1
    ex = int(round(bw * expand_frac))
    ey = int(round(bh * expand_frac))
    ax0 = max(0, x0 - ex); ay0 = max(0, y0 - ey)
    ax1 = min(W, x1 + ex); ay1 = min(H, y1 + ey)
    region = img[ay0:ay1, ax0:ax1]
    if region.size == 0:
        return x0, y0, x1, y1
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    ink = gray < white_thresh
    rh, rw = ink.shape
    # A row/col counts as "content" only if it has more than a trace of ink —
    # this ignores faint scanner speckle so we snap to the real chart.
    col_has = ink.sum(axis=0) > max(2, int(0.006 * rh))
    row_has = ink.sum(axis=1) > max(2, int(0.006 * rw))
    if not col_has.any() or not row_has.any():
        return x0, y0, x1, y1
    cmin = int(np.argmax(col_has)); cmax = rw - 1 - int(np.argmax(col_has[::-1]))
    rmin = int(np.argmax(row_has)); rmax = rh - 1 - int(np.argmax(row_has[::-1]))
    nx0 = ax0 + max(0, cmin - pad_px); ny0 = ay0 + max(0, rmin - pad_px)
    nx1 = ax0 + min(rw, cmax + 1 + pad_px); ny1 = ay0 + min(rh, rmax + 1 + pad_px)
    if nx1 - nx0 < 20 or ny1 - ny0 < 20:
        return x0, y0, x1, y1
    return nx0, ny0, nx1, ny1


def extract_figures(pdf_path, min_size=200, min_aspect=0.3, max_aspect=4.0):
    """
    Yield (image_bgr, meta) for each plausible raster figure in a PDF.

    Filters by minimum pixel size and aspect ratio to skip logos, icons,
    rules, and tiny inline glyphs. Tries to attach the nearest "Fig N..."
    caption found below the image on the same page.

    meta includes "bbox_pdf" (the figure's rectangle on the page in PDF
    points, [x0, y0, x1, y1]) when locatable, so the GUI can pre-seed a
    re-crop box. Note: only raster (embedded bitmap) figures are extracted.
    Pure-vector figures won't appear here; use page-render mode for those.
    """
    _require_fitz()
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

                # Locate this image's bbox on the page (caption + recrop).
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
                    "bbox_pdf": [float(bbox.x0), float(bbox.y0),
                                 float(bbox.x1), float(bbox.y1)] if bbox is not None else None,
                }
    finally:
        doc.close()


def extract_figures_via_page_render(pdf_path, screener, dpi=200, min_size=200,
                                     page_vlm_model=None, verbose=False,
                                     refine=False, refine_model=None):
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
    _require_fitz()
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

            # Optional refinement agent: review + correct the proposed boxes
            # (tighten / reject non-charts / split merged panels / add missed).
            refined_flag = False
            if refine and charts and hasattr(screener, "refine_page_charts"):
                print(f"      page {page_idx} -> refine boxes ... ", end="", flush=True)
                try:
                    ref = screener.refine_page_charts(page_img_bgr, charts,
                                                      model=refine_model)
                    if ref.get("_refined"):
                        charts = ref.get("charts", charts)
                        refined_flag = True
                        n_extractable = sum(1 for c in charts if c.get("extractable", False))
                        print(f"-> {len(charts)} region(s), {n_extractable} extractable")
                    else:
                        err = ref.get("_refine_error", "no change")
                        print(f"(kept original; {err})")
                except Exception as e:
                    print(f"refine error: {e}")

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
                # Snap the rough VLM box to the chart's real extent (pad so axis
                # labels aren't clipped, then trim blank borders).
                x0, y0, x1, y1 = snap_bbox_to_content(page_img_bgr, x0, y0, x1, y1, W, H)
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
                    "bbox_norm": [x0 / W, y0 / H, x1 / W, y1 / H],
                    "bbox_px": [x0, y0, x1, y1],
                    "page_size_px": [W, H],
                    "vlm_chart": chart,
                    "refined": refined_flag,
                    "refine_action": chart.get("action", ""),
                }
    finally:
        doc.close()


def render_pdf_page(pdf_path, page_idx, dpi=200):
    """
    Render a single PDF page (1-based page_idx) to a BGR numpy image.

    Used by the GUI's re-crop tool: the user adjusts a rectangle on the full
    rendered page, so this works for both raster and vector figures. Returns
    (image_bgr, zoom) where zoom = dpi/72 maps PDF points -> rendered pixels.
    Returns (None, 1.0) on failure.
    """
    _require_fitz()
    zoom = dpi / 72.0
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"    [open fail] {e}")
        return None, zoom
    try:
        if page_idx < 1 or page_idx > doc.page_count:
            return None, zoom
        page = doc[page_idx - 1]
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_arr = np.frombuffer(pix.samples, dtype=np.uint8)
        img_arr = img_arr.reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        elif pix.n == 1:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
        else:
            return None, zoom
        return np.ascontiguousarray(img_arr), zoom
    except Exception as e:
        print(f"    [render fail page {page_idx}] {e}")
        return None, zoom
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# DocLayout-YOLO figure locator (fast, offline, no API)
# ---------------------------------------------------------------------------
_DLY_MODEL = None
DLY_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
DLY_WEIGHTS = "doclayout_yolo_docstructbench_imgsz1024.pt"


def doclayout_available():
    try:
        import doclayout_yolo  # noqa: F401
        import huggingface_hub  # noqa: F401
        return True
    except Exception:
        return False


def _load_doclayout():
    """Lazy-load + cache the DocLayout-YOLO model (downloads weights once)."""
    global _DLY_MODEL
    if _DLY_MODEL is None:
        from huggingface_hub import hf_hub_download
        from doclayout_yolo import YOLOv10
        weights = hf_hub_download(repo_id=DLY_REPO, filename=DLY_WEIGHTS)
        _DLY_MODEL = YOLOv10(weights)
    return _DLY_MODEL


def extract_figures_via_doclayout(pdf_path, dpi=200, min_size=200, conf=0.20,
                                  classes=("figure",), pad_px=6, imgsz=1024,
                                  verbose=False):
    """
    Locate figures on each PDF page with DocLayout-YOLO and yield (image_bgr, meta).

    DocLayout-YOLO is a document-layout detector: it cleanly separates `figure`
    from `figure_caption` and the running header/footer (`abandon`), so figure
    crops exclude captions and page banners — no API, no per-page Claude call.
    Note it returns WHOLE figures (a multi-panel figure is one box) and labels
    everything `figure` (it doesn't say chart vs. photo); the user picks which to
    digitize. Pass classes=("figure","table") to also pull tables.
    """
    _require_fitz()
    model = _load_doclayout()
    names = model.names
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"    [open fail] {e}")
        return
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        for page_idx, page in enumerate(doc, start=1):
            try:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                elif pix.n == 1:
                    img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                else:
                    continue
                img = np.ascontiguousarray(img)
            except Exception as e:
                if verbose:
                    print(f"      [render fail page {page_idx}] {e}")
                continue
            H, W = img.shape[:2]
            print(f"      page {page_idx} ({W}x{H}) -> DocLayout-YOLO ... ",
                  end="", flush=True)
            try:
                res = model.predict(img, imgsz=imgsz, conf=conf, device="cpu",
                                    verbose=False)[0]
            except Exception as e:
                print(f"error: {e}")
                continue
            seq = 0
            for b in res.boxes:
                cls = names[int(b.cls)]
                if cls not in classes:
                    continue
                x0, y0, x1, y1 = [int(round(v)) for v in b.xyxy[0].tolist()]
                x0 = max(0, x0 - pad_px); y0 = max(0, y0 - pad_px)
                x1 = min(W, x1 + pad_px); y1 = min(H, y1 + pad_px)
                if x1 - x0 < min_size or y1 - y0 < min_size:
                    continue
                seq += 1
                yield img[y0:y1, x0:x1].copy(), {
                    "page": page_idx,
                    "img_idx_on_page": seq,
                    "width": int(x1 - x0),
                    "height": int(y1 - y0),
                    "caption": "",
                    "format": "doclayout-crop",
                    "render_dpi": dpi,
                    "bbox_norm": [x0 / W, y0 / H, x1 / W, y1 / H],
                    "bbox_px": [x0, y0, x1, y1],
                    "page_size_px": [W, H],
                    "detector": "doclayout",
                    "det_class": cls,
                    "conf": float(b.conf),
                }
            print(f"{seq} figure(s)")
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# MinerU PP-DocLayoutV2 chart locator (RT-DETR; separates charts from photos
# and splits composite panels instance-by-instance). Best figure locator.
# ---------------------------------------------------------------------------
_MINERU_DET = None
_MINERU_WEIGHTS = os.path.join("pp_doclayoutv2_weights", "models", "Layout",
                               "PP-DocLayoutV2")


def _mineru_weights_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), _MINERU_WEIGHTS)


def mineru_available():
    try:
        import mineru_layout  # noqa: F401
        import transformers   # noqa: F401
    except Exception:
        return False
    wdir = _mineru_weights_dir()
    return os.path.isdir(wdir) and os.path.exists(os.path.join(wdir, "model.safetensors"))


def _load_mineru(conf=0.45):
    """Lazy-load + cache the PP-DocLayoutV2 ChartDetector (mps/cuda/cpu)."""
    global _MINERU_DET
    if _MINERU_DET is None:
        import torch
        from mineru_layout import ChartDetector
        if torch.backends.mps.is_available():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
        print(f"[mineru] loading PP-DocLayoutV2 on {dev} ...")
        _MINERU_DET = ChartDetector(_mineru_weights_dir(), device=dev, conf=conf)
    return _MINERU_DET


def extract_figures_via_mineru(pdf_path, dpi=200, min_size=200, conf=0.45,
                               gutter_split=False, verbose=False):
    """
    Locate charts on each PDF page with MinerU's PP-DocLayoutV2 and yield
    (image_bgr, meta).

    PP-DocLayoutV2 (RT-DETR) has a dedicated `chart` class distinct from
    `image` (photos / SEM / schematics), and is instance-based, so the panels
    of a composite figure usually come back as SEPARATE chart boxes — unlike
    DocLayout-YOLO's single `figure` box around the whole composite. Captions,
    headers and titles are separate classes, so chart crops exclude them.
    Optional `gutter_split` whitespace-splits any composite the detector merged.
    """
    _require_fitz()
    det = _load_mineru(conf=conf)
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"    [open fail] {e}")
        return
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        for page_idx, page in enumerate(doc, start=1):
            try:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                elif pix.n == 1:
                    img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                else:
                    continue
                img = np.ascontiguousarray(img)
            except Exception as e:
                if verbose:
                    print(f"      [render fail page {page_idx}] {e}")
                continue
            H, W = img.shape[:2]
            print(f"      page {page_idx} ({W}x{H}) -> PP-DocLayoutV2 ... ",
                  end="", flush=True)
            try:
                charts = det.detect_charts(img, include_images=False,
                                           gutter_split=gutter_split)
            except Exception as e:
                print(f"error: {e}")
                continue
            seq = 0
            for c in charts:
                x0, y0, x1, y1 = [int(round(v)) for v in c["bbox"]]
                x0 = max(0, x0); y0 = max(0, y0)
                x1 = min(W, x1); y1 = min(H, y1)
                if x1 - x0 < min_size or y1 - y0 < min_size:
                    continue
                seq += 1
                yield img[y0:y1, x0:x1].copy(), {
                    "page": page_idx,
                    "img_idx_on_page": seq,
                    "width": int(x1 - x0),
                    "height": int(y1 - y0),
                    "caption": "",
                    "format": "mineru-crop",
                    "render_dpi": dpi,
                    "bbox_norm": [x0 / W, y0 / H, x1 / W, y1 / H],
                    "bbox_px": [x0, y0, x1, y1],
                    "page_size_px": [W, H],
                    "detector": "mineru",
                    "det_class": c.get("label", "chart"),
                    "conf": float(c.get("score", 0.0)),
                }
            print(f"{seq} chart(s)")
    finally:
        doc.close()
