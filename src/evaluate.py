"""
AutoLineDigitizer Evaluation Script (v2)
========================================
Compares AutoLineDigitizer's prediction against ground-truth (x, y) curves
from the Starrydata dataset.

This version adds the OFFICIAL CHART-Info Challenge metrics used by the
LineFormer paper (arxiv 2305.01837, Section 7):

  - Pairwise similarity = 1 - mean(|pred_y - gt_y| / y_range), clipped to [0, 1]
  - Task 6a (Visual Element Detection): K = N_gt, ignores false positives
  - Task 6b (Data Extraction): K = max(N_gt, N_pred), penalizes extras
  - Bipartite assignment via Hungarian algorithm

These scores are directly comparable to LineFormer's published numbers
(e.g. UB-PMC task 6b ~88.25), so this evaluation is defensible against
the standard benchmark.

We also keep our existing diagnostic metrics (axis errors, identity accuracy,
unit conversion handling) because they tell us WHERE the errors are, which
the single-score metric can't.

USAGE:
    # Put these files in the same folder as this script:
    #   - manifest.json
    #   - chart_NN_pred.zip   (AutoLineDigitizer's StarryDigitizer export)
    #
    # Then run:
    python evaluate.py

OUTPUTS:
    - chart_NN_eval.png         side-by-side comparison
    - chart_NN_eval_report.txt  text report
    - eval_summary.csv          scorecard with all metrics

TO ADD MORE CHARTS:
    Drop chart_NN_pred.zip alongside the manifest, then add the test_id to
    TEST_IDS below and re-run.
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------
TEST_IDS = ["chart_99"]              # add "chart_01", "chart_02", etc. as ready
MATCH_THRESHOLD_FRAC = 0.20          # MAE < 20% of GT Y range => a "match"
WORK_DIR = Path(__file__).parent     # script's own folder

# Unit conversions: (gt_unit_lower, target_unit_lower) -> multiplier_for_gt
# To convert GT into the target unit, multiply GT value by this factor.
UNIT_CONVERSIONS = {
    ("a*s/kg",  "mah/g"): 1.0 / 3600.0,    # 1 mAh/g = 3600 A·s/kg
    ("a·s/kg",  "mah/g"): 1.0 / 3600.0,
    ("a*s/g",   "mah/g"): 1.0 / 3.6,
    ("c/g",     "mah/g"): 1.0 / 3.6,       # 1 mAh = 3.6 C
}


def maybe_convert_units(gt_unit, pred_axis_range, gt_values):
    """If GT and predicted axis ranges differ wildly in magnitude, try a
    known unit conversion. Returns (converted_values, new_unit, note)."""
    gt = np.asarray(gt_values, dtype=float)
    if len(gt) == 0:
        return gt, gt_unit, "no GT data"
    gt_span = gt.max() - gt.min()
    pred_span = abs(pred_axis_range[1] - pred_axis_range[0])
    if gt_span == 0 or pred_span == 0:
        return gt, gt_unit, "skipped (zero span)"
    ratio = gt_span / pred_span
    if ratio > 100 or ratio < 0.01:
        gt_u = (gt_unit or "").lower().strip()
        for (src, tgt), factor in UNIT_CONVERSIONS.items():
            if src == gt_u:
                converted = gt * factor
                new_span = converted.max() - converted.min()
                if 0.1 < new_span / pred_span < 10:
                    return converted, tgt, f"converted GT from {gt_unit} to {tgt} (x{factor:g})"
        return gt, gt_unit, f"WARN: GT/pred span ratio = {ratio:.0f} but no matching conversion"
    return gt, gt_unit, "none"


def load_ground_truth(manifest_path, test_id):
    with open(manifest_path) as f:
        all_entries = json.load(f)
    matching = [e for e in all_entries if e["test_id"] == test_id]
    if not matching:
        raise ValueError(f"No entry for {test_id} in {manifest_path}")
    return matching[0]


def load_prediction(zip_path):
    """Return dict with predicted axis range and list of predicted curves
    (each curve = dict with 'name', 'x', 'y' in data space)."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("project.json") as f:
            proj = json.load(f)

    ax = proj["axisSets"][0]
    x1_px = ax["x1"]["coord"]["xPx"];  x1_val = ax["x1"]["value"]
    x2_px = ax["x2"]["coord"]["xPx"];  x2_val = ax["x2"]["value"]
    y1_py = ax["y1"]["coord"]["yPx"];  y1_val = ax["y1"]["value"]
    y2_py = ax["y2"]["coord"]["yPx"];  y2_val = ax["y2"]["value"]
    x_log = ax.get("xIsLogScale", False)
    y_log = ax.get("yIsLogScale", False)

    def px_to_data(xPx, yPx):
        x_raw = x1_val + (xPx - x1_px) * (x2_val - x1_val) / (x2_px - x1_px)
        y_raw = y1_val + (yPx - y1_py) * (y2_val - y1_val) / (y2_py - y1_py)
        return (10 ** x_raw if x_log else x_raw,
                10 ** y_raw if y_log else y_raw)

    pred_curves = []
    for ds in proj.get("datasets", []):
        pts = ds.get("points", [])
        if not pts:
            continue
        pts_sorted = sorted(pts, key=lambda p: p["xPx"])
        xs, ys = [], []
        for p in pts_sorted:
            x, y = px_to_data(p["xPx"], p["yPx"])
            xs.append(x); ys.append(y)
        pred_curves.append({"name": ds.get("name", "?"),
                            "x": np.array(xs), "y": np.array(ys)})

    return {
        "x_range": (x1_val, x2_val),
        "y_range": (y1_val, y2_val),
        "x_log": x_log,
        "y_log": y_log,
        "curves": pred_curves,
    }


# ---------------------------------------------------------------------------
# Similarity & MAE helpers
# ---------------------------------------------------------------------------

def curve_mae(pred_x, pred_y, gt_x, gt_y):
    """MAE in y at overlap region. Interpolates prediction onto GT x.
    Returns raw MAE (same units as y). Returns np.inf if no overlap."""
    x_lo = max(pred_x.min(), gt_x.min())
    x_hi = min(pred_x.max(), gt_x.max())
    if x_hi <= x_lo:
        return np.inf
    mask = (gt_x >= x_lo) & (gt_x <= x_hi)
    if mask.sum() < 2:
        return np.inf
    order = np.argsort(pred_x)
    px = pred_x[order]; py = pred_y[order]
    _, uniq = np.unique(px, return_index=True)
    px = px[uniq]; py = py[uniq]
    pred_at_gt = np.interp(gt_x[mask], px, py)
    return float(np.mean(np.abs(pred_at_gt - gt_y[mask])))


def pairwise_similarity(pred_x, pred_y, gt_x, gt_y, y_range):
    """Official CHART-Info Task 6a/6b pairwise similarity score.

    From the LineFormer paper Section 7.1:
        similarity = 1 - mean(|pred_y - gt_y| / y_range), clipped to [0, 1]

    The prediction is linearly interpolated onto the GT x values, as the
    paper specifies. Returns 0 if there is no x-overlap (the prediction
    can't explain that GT line at all).
    """
    if y_range <= 0:
        return 0.0
    x_lo = max(pred_x.min(), gt_x.min())
    x_hi = min(pred_x.max(), gt_x.max())
    if x_hi <= x_lo:
        return 0.0
    mask = (gt_x >= x_lo) & (gt_x <= x_hi)
    if mask.sum() < 1:
        return 0.0
    order = np.argsort(pred_x)
    px = pred_x[order]; py = pred_y[order]
    _, uniq = np.unique(px, return_index=True)
    px = px[uniq]; py = py[uniq]
    pred_at_gt = np.interp(gt_x[mask], px, py)
    norm_err = np.abs(pred_at_gt - gt_y[mask]) / y_range
    similarity = 1.0 - float(np.mean(norm_err))
    # Coverage penalty: if the prediction only covers part of the GT x-range,
    # scale the similarity by the fraction covered. This is implicit in the
    # paper (the GT points outside x-overlap contribute 0).
    coverage = mask.sum() / max(1, len(gt_x))
    similarity = similarity * coverage
    return max(0.0, min(1.0, similarity))


def worst_gap(pred_x, pred_y, gt_x, gt_y):
    """Largest |pred_y - gt_y| at overlap region (raw y units)."""
    x_lo = max(pred_x.min(), gt_x.min())
    x_hi = min(pred_x.max(), gt_x.max())
    if x_hi <= x_lo:
        return np.nan
    mask = (gt_x >= x_lo) & (gt_x <= x_hi)
    if mask.sum() < 2:
        return np.nan
    order = np.argsort(pred_x)
    px = pred_x[order]; py = pred_y[order]
    _, uniq = np.unique(px, return_index=True)
    px = px[uniq]; py = py[uniq]
    pred_at_gt = np.interp(gt_x[mask], px, py)
    return float(np.max(np.abs(pred_at_gt - gt_y[mask])))


# ---------------------------------------------------------------------------
# Per-chart evaluation
# ---------------------------------------------------------------------------

def evaluate_one(test_id, work_dir):
    print(f"\n[{test_id}] Starting evaluation")
    manifest_path = work_dir / "manifest.json"
    pred_zip = work_dir / f"{test_id}_pred.zip"
    if not manifest_path.exists():
        print(f"[{test_id}] ERROR: manifest.json not found at {manifest_path}")
        return None
    if not pred_zip.exists():
        print(f"[{test_id}] SKIP: {pred_zip.name} not found")
        return None

    entry = load_ground_truth(manifest_path, test_id)
    pred = load_prediction(pred_zip)

    # ---- Build GT arrays + convert units if needed ----
    gt_curves_raw = entry["ground_truth_curves"]
    unit_x = gt_curves_raw[0]["unit_x"]
    unit_y = gt_curves_raw[0]["unit_y"]

    all_gt_x_raw = np.concatenate([np.array(c["x"], dtype=float) for c in gt_curves_raw])
    all_gt_y_raw = np.concatenate([np.array(c["y"], dtype=float) for c in gt_curves_raw])
    conv_x, unit_x_used, note_x = maybe_convert_units(unit_x, pred["x_range"], all_gt_x_raw)
    conv_y, unit_y_used, note_y = maybe_convert_units(unit_y, pred["y_range"], all_gt_y_raw)
    x_factor = conv_x.max() / all_gt_x_raw.max() if all_gt_x_raw.max() != 0 else 1.0
    y_factor = conv_y.max() / all_gt_y_raw.max() if all_gt_y_raw.max() != 0 else 1.0

    gt_curves = []
    for c in gt_curves_raw:
        gx = np.array(c["x"], dtype=float) * x_factor
        gy = np.array(c["y"], dtype=float) * y_factor
        gt_curves.append({
            "x": gx, "y": gy,
            "composition": c.get("composition") or f"sample_{c.get('sample_id')}",
        })

    print(f"[{test_id}] Unit note (x): {note_x}")
    print(f"[{test_id}] Unit note (y): {note_y}")

    # ---- Axis range metrics ----
    all_x = np.concatenate([c["x"] for c in gt_curves])
    all_y = np.concatenate([c["y"] for c in gt_curves])
    gt_xr = (float(all_x.min()), float(all_x.max()))
    gt_yr = (float(all_y.min()), float(all_y.max()))
    pred_xr = pred["x_range"]
    pred_yr = pred["y_range"]
    gt_xs = gt_xr[1] - gt_xr[0];  gt_ys = gt_yr[1] - gt_yr[0]
    pred_xs = pred_xr[1] - pred_xr[0];  pred_ys = pred_yr[1] - pred_yr[0]

    x_range_err = abs(pred_xs - gt_xs) / gt_xs * 100 if gt_xs else float("nan")
    y_range_err = abs(pred_ys - gt_ys) / gt_ys * 100 if gt_ys else float("nan")
    x_min_err = abs(pred_xr[0] - gt_xr[0]) / gt_xs * 100 if gt_xs else float("nan")
    x_max_err = abs(pred_xr[1] - gt_xr[1]) / gt_xs * 100 if gt_xs else float("nan")
    y_min_err = abs(pred_yr[0] - gt_yr[0]) / gt_ys * 100 if gt_ys else float("nan")
    y_max_err = abs(pred_yr[1] - gt_yr[1]) / gt_ys * 100 if gt_ys else float("nan")

    # ============================================================
    # OFFICIAL CHART-Info metrics (LineFormer paper Section 7)
    # ============================================================
    # Build similarity matrix S[i, j] for GT curve i vs predicted curve j.
    # y_range used for normalization is the GT y-range (so a 0.03 V error on
    # a 0.54 V chart counts as 5.6%, comparable to other charts).
    n_gt = len(gt_curves)
    n_pred = len(pred["curves"])
    similarity_mat = np.zeros((n_gt, n_pred))
    cost_mae_mat = np.full((n_gt, n_pred), 1e9)
    for i, g in enumerate(gt_curves):
        for j, pr in enumerate(pred["curves"]):
            similarity_mat[i, j] = pairwise_similarity(
                pr["x"], pr["y"], g["x"], g["y"], gt_ys
            )
            m = curve_mae(pr["x"], pr["y"], g["x"], g["y"])
            cost_mae_mat[i, j] = m if np.isfinite(m) else 1e9

    # Hungarian assignment maximizing similarity = minimizing (1 - similarity)
    if n_pred == 0 or n_gt == 0:
        row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)
        sum_matched_similarity = 0.0
    else:
        # Pad similarity to square if shapes differ (scipy handles non-square already)
        row_ind, col_ind = linear_sum_assignment(-similarity_mat)
        sum_matched_similarity = float(similarity_mat[row_ind, col_ind].sum())

    # Task 6a — Visual Element Detection (recall only)
    # score = (1/N_gt) * Σ S_ij for matched (i,j)
    task_6a = sum_matched_similarity / n_gt if n_gt > 0 else float("nan")

    # Task 6b — Data Extraction (penalizes false positives via K = max(N_gt, N_pred))
    K = max(n_gt, n_pred) if max(n_gt, n_pred) > 0 else 1
    task_6b = sum_matched_similarity / K

    # ============================================================
    # Our diagnostic metrics (kept for actionable insight)
    # ============================================================
    # MAE-based matching with threshold (for "identity accuracy")
    if n_pred == 0 or n_gt == 0:
        mae_row, mae_col = np.array([], dtype=int), np.array([], dtype=int)
    else:
        mae_row, mae_col = linear_sum_assignment(cost_mae_mat)

    threshold = MATCH_THRESHOLD_FRAC * gt_ys
    matches = []
    for i, j in zip(mae_row, mae_col):
        mae = cost_mae_mat[i, j]
        is_match = mae < threshold
        matches.append({"gt_idx": int(i), "pred_idx": int(j),
                        "mae": float(mae), "ok": bool(is_match)})
    n_matched = sum(m["ok"] for m in matches)
    identity_acc = n_matched / n_gt * 100 if n_gt else float("nan")
    matched_maes = [m["mae"] for m in matches if m["ok"]]
    avg_mae = float(np.mean(matched_maes)) if matched_maes else float("nan")
    # Normalized MAE (as % of GT y range) — comparable across charts
    avg_mae_norm = (avg_mae / gt_ys * 100) if (matched_maes and gt_ys > 0) else float("nan")

    # Worst single-point gap on matched pairs (raw y units)
    worst = []
    for m in matches:
        if not m["ok"]:
            continue
        g = gt_curves[m["gt_idx"]]; pr = pred["curves"][m["pred_idx"]]
        worst.append(worst_gap(pr["x"], pr["y"], g["x"], g["y"]))
    worst_gap_val = float(np.nanmax(worst)) if worst else float("nan")
    worst_gap_norm = (worst_gap_val / gt_ys * 100) if (worst and gt_ys > 0) else float("nan")

    # ---- Write text report ----
    report_path = work_dir / f"{test_id}_eval_report.txt"
    lines = []
    lines.append("=" * 64)
    lines.append(f"EVALUATION REPORT — {test_id}")
    lines.append("=" * 64)
    lines.append(f"Paper:  {(entry.get('paper_title') or '')[:70]}")
    lines.append(f"DOI:    {entry.get('paper_doi')}")
    lines.append(f"Figure: {entry.get('figure_name')}  ({entry.get('num_curves_in_figure')} curves in GT)")
    lines.append(f"Properties: {gt_curves_raw[0]['prop_x']} ({unit_x_used}) vs {gt_curves_raw[0]['prop_y']} ({unit_y_used})")
    lines.append(f"Unit conversion (x): {note_x}")
    lines.append(f"Unit conversion (y): {note_y}")
    lines.append("")
    lines.append("CHART-INFO OFFICIAL SCORES (comparable to LineFormer paper)")
    lines.append(f"  Task 6a (Visual Element Detection, recall only): {task_6a:.4f}")
    lines.append(f"  Task 6b (Data Extraction, penalizes extras):     {task_6b:.4f}")
    lines.append(f"  (LineFormer paper reports 0.8825 on UB-PMC task 6b for reference)")
    lines.append("")
    lines.append("AXIS CALIBRATION (our extension — not in LineFormer eval)")
    lines.append(f"  GT X range:    [{gt_xr[0]:.3g}, {gt_xr[1]:.3g}] {unit_x_used}  (size: {gt_xs:.3g})")
    lines.append(f"  Pred X range:  [{pred_xr[0]:.3g}, {pred_xr[1]:.3g}] {unit_x_used}  (size: {pred_xs:.3g})")
    lines.append(f"  X range size error: {x_range_err:.1f}%")
    lines.append(f"  X-min error: {x_min_err:.1f}%, X-max error: {x_max_err:.1f}%")
    lines.append("")
    lines.append(f"  GT Y range:    [{gt_yr[0]:.3g}, {gt_yr[1]:.3g}] {unit_y_used}  (size: {gt_ys:.3g})")
    lines.append(f"  Pred Y range:  [{pred_yr[0]:.3g}, {pred_yr[1]:.3g}] {unit_y_used}  (size: {pred_ys:.3g})")
    lines.append(f"  Y range size error: {y_range_err:.1f}%")
    lines.append(f"  Y-min error: {y_min_err:.1f}%, Y-max error: {y_max_err:.1f}%")
    lines.append("")
    lines.append("CURVE EXTRACTION (diagnostic)")
    lines.append(f"  Lines predicted in image: {n_pred}")
    lines.append(f"  Curves in ground truth:    {n_gt}")
    lines.append(f"  Line identity accuracy:    {n_matched}/{n_gt} = {identity_acc:.0f}%")
    for m in matches:
        tag = "✓" if m["ok"] else "✗"
        g = gt_curves[m["gt_idx"]]; pr = pred["curves"][m["pred_idx"]]
        lines.append(f"    {tag} {g['composition']} -> {pr['name']} (MAE = {m['mae']:.3g} {unit_y_used})")
    lines.append(f"  Mean absolute Y error (matched, raw):       {avg_mae:.3g} {unit_y_used}")
    lines.append(f"  Mean absolute Y error (matched, normalized): {avg_mae_norm:.2f}% of GT Y range")
    lines.append(f"  Worst single-point gap (matched, raw):       {worst_gap_val:.3g} {unit_y_used}")
    lines.append(f"  Worst single-point gap (matched, normalized): {worst_gap_norm:.2f}% of GT Y range")
    lines.append("=" * 64)
    report_path.write_text("\n".join(lines))
    print(f"[{test_id}] Wrote {report_path.name}")
    print("\n".join(lines))

    # ---- Comparison plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    gt_colors = ["tab:blue", "tab:green", "tab:red", "tab:orange",
                 "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]

    ax_l = axes[0]
    for i, g in enumerate(gt_curves):
        ax_l.plot(g["x"], g["y"], "o-", color=gt_colors[i % len(gt_colors)],
                  label=g["composition"], markersize=4, linewidth=1.5)
    ax_l.set_xlabel(f"{gt_curves_raw[0]['prop_x']} ({unit_x_used})")
    ax_l.set_ylabel(f"{gt_curves_raw[0]['prop_y']} ({unit_y_used})")
    ax_l.set_title("Ground Truth\n(from Starrydata)")
    ax_l.legend(loc="best", fontsize=8)
    ax_l.grid(alpha=0.3)

    ax_r = axes[1]
    matched_pred_to_gt = {m["pred_idx"]: m["gt_idx"] for m in matches if m["ok"]}
    for j, pr in enumerate(pred["curves"]):
        if j in matched_pred_to_gt:
            gi = matched_pred_to_gt[j]
            color = gt_colors[gi % len(gt_colors)]
            label = f"{pr['name']} → GT {gt_curves[gi]['composition']} ✓"
            lw = 2; alpha = 0.95
        else:
            color = "lightgray"
            label = f"{pr['name']} (no match)"
            lw = 1; alpha = 0.6
        ax_r.plot(pr["x"], pr["y"], "-", color=color, label=label, linewidth=lw, alpha=alpha)
    ax_r.set_xlabel(f"{gt_curves_raw[0]['prop_x']} ({unit_x_used})")
    ax_r.set_ylabel(f"{gt_curves_raw[0]['prop_y']} ({unit_y_used})")
    ax_r.set_title("AutoLineDigitizer Prediction")
    ax_r.legend(loc="best", fontsize=8)
    ax_r.grid(alpha=0.3)

    y_pad = 0.2 * gt_ys
    for a in axes:
        a.set_ylim(gt_yr[0] - y_pad, gt_yr[1] + y_pad)

    plt.subplots_adjust(bottom=0.22)
    metrics_text = (
        f"OFFICIAL — Task 6a: {task_6a:.3f}   |   Task 6b: {task_6b:.3f}       "
        f"OURS — Identity: {identity_acc:.0f}% ({n_matched}/{n_gt})   |   "
        f"MAE matched: {avg_mae_norm:.1f}% of Y range"
    )
    fig.text(0.5, 0.06, metrics_text, ha="center", fontsize=11,
             bbox=dict(facecolor="lightyellow", edgecolor="gray",
                       boxstyle="round,pad=0.5"))
    fig.suptitle(f"AutoLineDigitizer Evaluation — {test_id} ({entry.get('paper_doi')})",
                 fontsize=13, y=0.99)

    plot_path = work_dir / f"{test_id}_eval.png"
    plt.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[{test_id}] Wrote {plot_path.name}")

    return {
        "test_id": test_id,
        "num_curves_gt": n_gt,
        "num_curves_pred": n_pred,
        "task_6a": round(task_6a, 4) if not np.isnan(task_6a) else None,
        "task_6b": round(task_6b, 4) if not np.isnan(task_6b) else None,
        "identity_acc_pct": round(identity_acc, 1),
        "mae_matched_norm_pct": round(avg_mae_norm, 2) if not np.isnan(avg_mae_norm) else None,
        "mae_matched_raw": round(avg_mae, 4) if not np.isnan(avg_mae) else None,
        "worst_gap_norm_pct": round(worst_gap_norm, 2) if not np.isnan(worst_gap_norm) else None,
        "x_range_err_pct": round(x_range_err, 2),
        "y_range_err_pct": round(y_range_err, 2),
        "paper_doi": entry.get("paper_doi"),
    }


def main():
    results = []
    for tid in TEST_IDS:
        r = evaluate_one(tid, WORK_DIR)
        if r is not None:
            results.append(r)

    if results:
        import csv
        summary_path = WORK_DIR / "eval_summary.csv"
        cols = ["test_id", "num_curves_gt", "num_curves_pred",
                "task_6a", "task_6b",
                "identity_acc_pct", "mae_matched_norm_pct", "mae_matched_raw",
                "worst_gap_norm_pct",
                "x_range_err_pct", "y_range_err_pct", "paper_doi"]
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"\nWrote summary: {summary_path}")
        print("\nSUMMARY (across all charts)")
        print("-" * 105)
        print(f"{'test_id':<12}{'GT':<4}{'Pred':<6}{'Task6a':<9}{'Task6b':<9}"
              f"{'Identity%':<11}{'MAE_norm%':<11}{'X_err%':<8}{'Y_err%':<8}")
        print("-" * 105)
        for r in results:
            print(f"{r['test_id']:<12}{r['num_curves_gt']:<4}{r['num_curves_pred']:<6}"
                  f"{str(r['task_6a']):<9}{str(r['task_6b']):<9}"
                  f"{str(r['identity_acc_pct']):<11}{str(r['mae_matched_norm_pct']):<11}"
                  f"{str(r['x_range_err_pct']):<8}{str(r['y_range_err_pct']):<8}")


if __name__ == "__main__":
    main()