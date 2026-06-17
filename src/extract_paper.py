# -*- coding: utf-8 -*-
"""
extract_paper.py — End-to-end PDF -> KMDS + chart extractions.

Pipeline:
  1. pdf_processor.py        -> metadata + figure crops + captions
  2. Claude API + prompt     -> KMDS JSON (EN + JA)
  3. LineFormer              -> per-figure raw line traces (pixel coords)
  4. Output everything to <paper_name>_kmds/ ready for editing

Usage:
  python extract_paper.py paper.pdf
  python extract_paper.py paper.pdf -o my_output/ --skip-kmds
  python extract_paper.py paper.pdf --lf-model general_v2

After running, open desktop_app.py on any figure to refine:
  python desktop_app.py
  # then File > Open > <paper_name>_kmds/figures/figure_001.png
"""

import os
import sys
import re
import json
import time
import base64
import asyncio
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import cv2

# Local modules (must be in same dir)
from pdf_processor import process_pdf, PDFPackage

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False


# ===================================================================
# Part 1: KMDS extraction via Claude API
# ===================================================================

def _encode_pdf_b64(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def extract_kmds_via_claude(pdf_path: str,
                              prompt_path: str,
                              output_dir: str,
                              model: str = "claude-sonnet-4-6",
                              base_name: Optional[str] = None) -> Dict[str, Any]:
    """Run Claude API with the extraction prompt on the PDF.

    Returns dict with paths to en/ja JSON files (and any errors).
    """
    if not ANTHROPIC_AVAILABLE:
        return {"_error": "anthropic SDK not installed. pip install anthropic"}
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"_error": "ANTHROPIC_API_KEY not set"}
    if not os.path.exists(prompt_path):
        return {"_error": f"Prompt file not found: {prompt_path}"}

    prompt_text = open(prompt_path, "r", encoding="utf-8").read()
    if base_name is None:
        base_name = Path(pdf_path).stem

    print(f"⤷ Sending PDF + extraction_prompt.md to {model}...")
    print(f"   (this is the slow step — Claude reads the whole paper)")

    client = anthropic.Anthropic()
    pdf_b64 = _encode_pdf_b64(pdf_path)

    instructional = (
        prompt_text
        + "\n\n---\n\n"
        + "IMPORTANT for this API run:\n"
        + "- Do NOT actually write out file attachments (your environment cannot).\n"
        + "- Instead, output the English KMDS JSON inside a markdown code block "
        + "labeled ```json_en, and the Japanese KMDS JSON inside ```json_ja.\n"
        + "- Then on a new line write '## Schema extension candidates' "
        + "and your candidates (or 'none').\n"
        + "- The base filename for both JSONs is: " + base_name + "\n"
    )

    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=16000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "document", "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_b64,
                }},
                {"type": "text", "text": instructional},
            ],
        }],
    )
    elapsed = time.time() - t0
    raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    print(f"   Claude finished in {elapsed:.1f}s "
          f"(input {resp.usage.input_tokens:,} + output {resp.usage.output_tokens:,} tokens)")

    # Parse out the two JSON blocks
    en_match = re.search(r"```(?:json_en|json)?\s*\n(\{[\s\S]+?\})\s*```", raw, re.MULTILINE)
    ja_match = re.search(r"```json_ja\s*\n(\{[\s\S]+?\})\s*```", raw, re.MULTILINE)

    result: Dict[str, Any] = {"raw_response_path": None, "elapsed_sec": elapsed,
                                "input_tokens": resp.usage.input_tokens,
                                "output_tokens": resp.usage.output_tokens}
    raw_path = os.path.join(output_dir, "_claude_raw_response.md")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw)
    result["raw_response_path"] = raw_path

    if en_match:
        try:
            en_data = json.loads(en_match.group(1))
            en_path = os.path.join(output_dir, f"{base_name}.json")
            with open(en_path, "w", encoding="utf-8") as f:
                json.dump(en_data, f, indent=2, ensure_ascii=False)
            result["en_path"] = en_path
            print(f"   ✅ Saved EN: {en_path}")
        except json.JSONDecodeError as e:
            result["en_parse_error"] = str(e)
            print(f"   ⚠ EN JSON parse failed: {e}")
    else:
        result["_warn"] = "No EN JSON block found in response"
        print("   ⚠ No EN JSON block found. See _claude_raw_response.md")

    if ja_match:
        try:
            ja_data = json.loads(ja_match.group(1))
            ja_path = os.path.join(output_dir, f"{base_name}_ja.json")
            with open(ja_path, "w", encoding="utf-8") as f:
                json.dump(ja_data, f, indent=2, ensure_ascii=False)
            result["ja_path"] = ja_path
            print(f"   ✅ Saved JA: {ja_path}")
        except json.JSONDecodeError as e:
            result["ja_parse_error"] = str(e)
            print(f"   ⚠ JA JSON parse failed: {e}")
    else:
        print("   ⚠ No JA JSON block found")

    return result


# ===================================================================
# Part 2: AutoLineDigitizer chart extraction
# ===================================================================

def extract_charts(figure_entries: List,
                    output_dir: str,
                    lf_model: str = "general_v2",
                    use_color_refinement: bool = False) -> List[Dict[str, Any]]:
    """Run LineFormer on every figure crop, saving raw traces only.

    Args:
        figure_entries: list of FigureEntry objects from pdf_processor
        output_dir: where to save per-figure outputs
        lf_model: LineFormer model key (e.g. 'general_v2', 'battery_finetuned')
        use_color_refinement: pass to LineFormerApp (default False — your ablation showed better without)

    Returns:
        list of per-figure result dicts (with path to lines.json)
    """
    # Lazy import so we don't pull Flet for users running --skip-charts
    print(f"⤷ Loading LineFormer model: {lf_model} (this can take a few seconds)...")
    try:
        from desktop_app import LineFormerApp
    except ImportError as e:
        print(f"   ⚠ desktop_app.py import failed: {e}")
        return []

    app = LineFormerApp()
    app.use_color_refinement = use_color_refinement
    try:
        app.load_lineformer_model(lf_model)
    except Exception as e:
        print(f"   ⚠ Model load failed: {e}")
        return []
    print(f"   Model loaded.")

    charts_subdir = os.path.join(output_dir, "chart_extracts")
    os.makedirs(charts_subdir, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for fig in figure_entries:
        img_path = fig.image_path
        fig_idx = fig.index
        print(f"  Figure {fig_idx} ({os.path.basename(img_path)})...", end=" ")

        img = cv2.imread(img_path)
        if img is None:
            print("✗ unreadable")
            continue

        try:
            data_series = app.extract_lines(img)
        except Exception as e:
            print(f"✗ extract failed: {e}")
            continue

        raw_lines = app.raw_lines or []
        n_curves = len(raw_lines)

        # Save raw traces
        lines_path = os.path.join(charts_subdir, f"figure_{fig_idx:03d}_lines.json")
        with open(lines_path, "w") as f:
            json.dump({
                "figure_index": fig_idx,
                "source_image": img_path,
                "n_curves": n_curves,
                "lf_model": lf_model,
                "use_color_refinement": use_color_refinement,
                "raw_lines": [
                    [[int(p[0]), int(p[1])] for p in curve]
                    for curve in raw_lines
                ],
            }, f, indent=2)

        print(f"✓ {n_curves} curves")

        results.append({
            "figure_index": fig_idx,
            "image_path": img_path,
            "lines_json": lines_path,
            "n_curves": n_curves,
            "edited_by_human": False,
        })

    return results


# ===================================================================
# Part 3: Top-level orchestrator
# ===================================================================

def main():
    p = argparse.ArgumentParser(
        description="PDF -> KMDS JSON + AutoLineDigitizer chart data, end-to-end."
    )
    p.add_argument("pdf", help="Path to paper PDF")
    p.add_argument("-o", "--output", default=None,
                   help="Output dir (default: <pdf_name>_kmds/)")
    p.add_argument("--prompt", default="extraction_prompt.md",
                   help="Path to KMDS extraction prompt (default: ./extraction_prompt.md)")
    p.add_argument("--kmds-model", default="claude-sonnet-4-6",
                   help="Claude model for KMDS extraction")
    p.add_argument("--kmds-parallel", action="store_true",
                   help="Use parallel KMDS extraction (kmds_parallel.py: 5 focused "
                        "section calls + 1 translation) instead of the monolithic call")
    p.add_argument("--bbox-model", default="claude-sonnet-4-6",
                   help="Claude model for figure bbox detection in pdf_processor")
    p.add_argument("--lf-model", default="general_v2",
                   help="LineFormer model key (default: general_v2)")
    p.add_argument("--skip-kmds", action="store_true",
                   help="Skip KMDS extraction (only run pdf_processor + chart extraction)")
    p.add_argument("--skip-charts", action="store_true",
                   help="Skip AutoLineDigitizer (only run pdf_processor + KMDS)")
    p.add_argument("--raster", action="store_true",
                   help="Use raster mode in pdf_processor (instead of page-render)")
    args = p.parse_args()

    if not os.path.exists(args.pdf):
        print(f"❌ PDF not found: {args.pdf}")
        sys.exit(1)

    base_name = Path(args.pdf).stem
    output_dir = args.output or f"{base_name}_kmds"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"PAPER: {args.pdf}")
    print(f"OUTPUT: {output_dir}")
    print("=" * 70)

    # === STAGE 1: pdf_processor ===
    print("\n[STAGE 1] PDF processing (metadata + figures + captions)")
    print("-" * 70)
    pkg = process_pdf(
        args.pdf, output_dir,
        page_render=not args.raster,
        vlm_model=args.bbox_model,
        use_crossref=True,
        use_screener=True,
    )

    # === STAGE 2: KMDS extraction ===
    kmds_result = None
    if not args.skip_kmds:
        print("\n[STAGE 2] KMDS extraction (Claude + extraction_prompt.md)")
        print("-" * 70)
        if args.kmds_parallel:
            import kmds_parallel
            kmds_result = asyncio.run(kmds_parallel.extract_kmds_parallel(
                args.pdf, output_dir, base_name=base_name, prompt_path=args.prompt,
                model=args.kmds_model,
            ))
        else:
            kmds_result = extract_kmds_via_claude(
                args.pdf, args.prompt, output_dir,
                model=args.kmds_model, base_name=base_name,
            )
    else:
        print("\n[STAGE 2] Skipped (--skip-kmds)")

    # === STAGE 3: Chart extraction ===
    chart_results = None
    if not args.skip_charts:
        print("\n[STAGE 3] AutoLineDigitizer chart extraction")
        print("-" * 70)
        chart_results = extract_charts(
            pkg.figures, output_dir,
            lf_model=args.lf_model,
        )
    else:
        print("\n[STAGE 3] Skipped (--skip-charts)")

    # === Write combined summary ===
    summary = {
        "pdf": os.path.abspath(args.pdf),
        "output_dir": os.path.abspath(output_dir),
        "metadata": pkg.metadata,
        "n_figures": len(pkg.figures),
        "kmds": kmds_result,
        "charts": chart_results,
    }
    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    # === Final printout ===
    print("\n" + "=" * 70)
    print(f"✅  Done.")
    print("=" * 70)
    print(f"Output dir:  {output_dir}/")
    print(f"  • paper_manifest.json     -- figure metadata")
    if kmds_result and kmds_result.get("en_path"):
        print(f"  • {base_name}.json        -- KMDS English")
    if kmds_result and kmds_result.get("ja_path"):
        print(f"  • {base_name}_ja.json     -- KMDS Japanese")
    print(f"  • figures/                -- cropped chart images")
    if chart_results:
        print(f"  • chart_extracts/         -- LineFormer traces (figure_NNN_lines.json)")
    print(f"  • pipeline_summary.json   -- everything joined together")
    print()
    print("To refine any figure in the desktop app:")
    print(f"  python desktop_app.py")
    print(f"  # then File > Open > {output_dir}/figures/figure_NNN.png")
    print()


if __name__ == "__main__":
    main()
