#!/usr/bin/env python3
"""apply_pagerender_patch.py — Adds page-render + VLM bbox detection to
vlm_screener.py + batch_extract.py in place.

Safe to re-run: detects if already patched and skips.
"""
import sys
import os
import shutil


VLM_PATH = "vlm_screener.py"
BATCH_PATH = "batch_extract.py"


def patch_vlm_screener(s: str) -> str:
    """Two patches: PAGE_SCREEN_PROMPT constant + screen_page() method."""
    if "PAGE_SCREEN_PROMPT" in s:
        print("  vlm_screener.py: already has PAGE_SCREEN_PROMPT, skipping.")
        return s

    # --- Patch A: add PAGE_SCREEN_PROMPT after SCREEN_PROMPT ---
    anchor_a = '''If the figure shows multiple sub-panels (e.g., (a), (b), (c)),
set extractable=false and chart_type="multi_panel".

If axes exist but tick labels are too small or blurry to read with confidence,
set extractable=false and axis_labels_readable=false.

Return JSON only.
"""'''
    if anchor_a not in s:
        raise SystemExit("vlm_screener.py: anchor_a not found (SCREEN_PROMPT closing)")

    page_prompt = '''If the figure shows multiple sub-panels (e.g., (a), (b), (c)),
set extractable=false and chart_type="multi_panel".

If axes exist but tick labels are too small or blurry to read with confidence,
set extractable=false and axis_labels_readable=false.

Return JSON only.
"""


# Prompt for full-page bounding-box detection (--page-render mode).
PAGE_SCREEN_PROMPT = """\\
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
"""'''
    s = s.replace(anchor_a, page_prompt)

    # --- Patch B: add screen_page() method after screen() method ---
    anchor_b = '''        except Exception as e:
            result["_screening_error"] = f"{type(e).__name__}: {e}"
            result["reason"] = "screening failed; default permissive"
            return result'''
    if anchor_b not in s:
        raise SystemExit("vlm_screener.py: anchor_b not found (screen() ending)")

    screen_page_method = '''        except Exception as e:
            result["_screening_error"] = f"{type(e).__name__}: {e}"
            result["reason"] = "screening failed; default permissive"
            return result

    def screen_page(self, page_img_bgr, model=None):
        """
        Find ALL extractable chart regions on a rendered PDF page.

        Returns {"n_charts": int, "charts": [...]}. Each chart entry has:
            bbox_norm   : [x0,y0,x1,y1] normalized to [0,1]
            chart_type  : line_chart / scatter / ...
            extractable : bool
            axis_labels_readable, n_lines_estimate, figure_label, reason

        Caller multiplies bbox_norm by page dims to crop.
        Defensive: returns {"n_charts": 0, "charts": []} on any failure with
        _screening_error set.
        """
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
            return result'''
    s = s.replace(anchor_b, screen_page_method)
    return s


def patch_batch_extract(s: str) -> str:
    """Adds page-render extraction + CLI flags + main() branching."""
    if "extract_figures_via_page_render" in s:
        print("  batch_extract.py: already has extract_figures_via_page_render, skipping.")
        return s

    # --- Patch C: insert extract_figures_via_page_render after extract_figures ---
    anchor_c = '''                yield arr_bgr, {
                    "page": page_idx,
                    "img_idx_on_page": img_idx,
                    "width": int(w),
                    "height": int(h),
                    "caption": caption,
                    "format": ext,
                }
    finally:
        doc.close()'''
    if anchor_c not in s:
        raise SystemExit("batch_extract.py: anchor_c not found (extract_figures end)")

    page_render_func = '''                yield arr_bgr, {
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
    are vector PDF graphics (no embedded bitmap).

    Yields (image_bgr, meta) where meta also contains "vlm_chart" (the page-
    level VLM judgement on this region).
    """
    if screener is None:
        return

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
            except Exception as e:
                if verbose:
                    print(f"      [render fail page {page_idx}] {e}")
                continue

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
                    print(f"      [unsupported channels page {page_idx}: n={pix.n}]")
                continue
            page_img_bgr = np.ascontiguousarray(img_arr)
            H, W = page_img_bgr.shape[:2]

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
        doc.close()'''
    s = s.replace(anchor_c, page_render_func)

    # --- Patch D: add CLI flags after --vlm-model ---
    anchor_d = '''    parser.add_argument("--vlm-model", default="claude-haiku-4-5-20251001",
                        help="Anthropic model id for screening (default: claude-haiku-4-5-20251001)")
    parser.add_argument("--min-size", type=int, default=200,'''
    if anchor_d not in s:
        raise SystemExit("batch_extract.py: anchor_d not found (CLI flags)")

    new_flags = '''    parser.add_argument("--vlm-model", default="claude-haiku-4-5-20251001",
                        help="Anthropic model id for screening (default: claude-haiku-4-5-20251001)")
    parser.add_argument("--page-render", action="store_true",
                        help="Render each PDF page and use VLM to find chart bounding boxes. "
                             "Required for papers with vector PDF graphics (Nature, Cell, etc.). "
                             "Implies --vlm-filter.")
    parser.add_argument("--page-dpi", type=int, default=200,
                        help="DPI for page rendering in --page-render mode (default 200)")
    parser.add_argument("--page-vlm-model", default=None,
                        help="Override Anthropic model for page-level bbox detection. "
                             "Defaults to --vlm-model. Sonnet may give more accurate bboxes.")
    parser.add_argument("--min-size", type=int, default=200,'''
    s = s.replace(anchor_d, new_flags)

    # --- Patch E: auto-enable vlm_filter after parse_args ---
    anchor_e = '''    args = parser.parse_args()

    input_dir = Path(args.input)'''
    if anchor_e not in s:
        raise SystemExit("batch_extract.py: anchor_e not found (after parse_args)")

    new_post_parse = '''    args = parser.parse_args()

    # --page-render requires the VLM screener; enable --vlm-filter automatically.
    if args.page_render and not args.vlm_filter:
        print("Note: --page-render requires VLM; enabling --vlm-filter automatically.")
        args.vlm_filter = True

    input_dir = Path(args.input)'''
    s = s.replace(anchor_e, new_post_parse)

    # --- Patch F: switch extraction path in main() loop ---
    anchor_f = '''        figures_seen = 0
        for fig_idx, (img_bgr, fig_meta) in enumerate(
                extract_figures(str(pdf_path), min_size=args.min_size), start=1):
            if args.limit_charts and figures_seen >= args.limit_charts:
                break
            figures_seen += 1
            n_total_figures += 1

            fig_basename = f"fig_p{fig_meta['page']:03d}_i{fig_meta['img_idx_on_page']}"
            print(f"      fig {fig_idx} (p{fig_meta['page']}, {fig_meta['width']}x{fig_meta['height']}) ... ",
                  end="", flush=True)

            result = process_figure(app, img_bgr, fig_meta, paper_dir, fig_basename, args,
                                     screener=screener)'''
    if anchor_f not in s:
        raise SystemExit("batch_extract.py: anchor_f not found (extraction loop)")

    new_loop = '''        figures_seen = 0

        # Pick extraction strategy: page-render (VLM bbox, vector-aware)
        # or raster (embedded bitmaps via PyMuPDF).
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

            # In page-render mode the chart was already screened at page level.
            per_fig_screener = None if pre_screened else screener
            result = process_figure(app, img_bgr, fig_meta, paper_dir, fig_basename, args,
                                     screener=per_fig_screener)

            # Lift the page-level VLM verdict into result.vlm_screen.
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
                }'''
    s = s.replace(anchor_f, new_loop)

    return s


def main():
    if not os.path.isfile(VLM_PATH):
        sys.exit(f"ERROR: {VLM_PATH} not found in current directory.")
    if not os.path.isfile(BATCH_PATH):
        sys.exit(f"ERROR: {BATCH_PATH} not found in current directory.")

    with open(VLM_PATH) as f:
        s1 = f.read()
    with open(BATCH_PATH) as f:
        s2 = f.read()

    print("Patching vlm_screener.py ...")
    s1_new = patch_vlm_screener(s1)
    print("Patching batch_extract.py ...")
    s2_new = patch_batch_extract(s2)

    # Backups
    if s1_new != s1:
        shutil.copy(VLM_PATH, VLM_PATH + ".backup_pre_pagerender")
    if s2_new != s2:
        shutil.copy(BATCH_PATH, BATCH_PATH + ".backup_pre_pagerender")

    with open(VLM_PATH, "w") as f:
        f.write(s1_new)
    with open(BATCH_PATH, "w") as f:
        f.write(s2_new)

    print("\n✅ All patches applied successfully.")
    print(f"   Backups: {VLM_PATH}.backup_pre_pagerender, {BATCH_PATH}.backup_pre_pagerender")


if __name__ == "__main__":
    main()
