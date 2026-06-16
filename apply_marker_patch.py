"""
apply_marker_patch.py
Adds 'Detect Markers' button to desktop_app.py.

What it does:
  1. Backs up desktop_app.py to desktop_app.py.before_marker_patch
  2. Adds `from line_marker_detector import find_markers_on_traced_line`
  3. Defines detect_markers_btn next to verify_btn
  4. Defines on_detect_markers_click handler next to on_verify_click
  5. Adds detect_markers_btn into the settings_panel (next to verify_btn)
  6. Enables the button after extraction succeeds
  7. Adds to nonlocal declarations in process_image

Run:
    python apply_marker_patch.py

To revert:
    cp desktop_app.py.before_marker_patch desktop_app.py
"""
import os
import shutil
import sys

TARGET = "desktop_app.py"
BACKUP = "desktop_app.py.before_marker_patch"

if not os.path.exists(TARGET):
    print(f"❌ {TARGET} not found in {os.getcwd()}")
    sys.exit(1)

# Make backup once
if not os.path.exists(BACKUP):
    shutil.copy(TARGET, BACKUP)
    print(f"✅ Backup saved: {BACKUP}")
else:
    print(f"(Backup already exists: {BACKUP})")

content = open(TARGET, encoding="utf-8").read()
edits_applied = 0
edits_already = 0


def replace_unique(content, old, new, label):
    """Replace `old` with `new`. Errors if `old` not found or appears >1 time."""
    global edits_applied, edits_already
    if new in content:
        print(f"  ⏭  [{label}] already applied")
        edits_already += 1
        return content
    count = content.count(old)
    if count == 0:
        print(f"  ❌ [{label}] anchor not found")
        return content
    if count > 1:
        print(f"  ❌ [{label}] anchor appears {count} times (not unique)")
        return content
    print(f"  ✅ [{label}] applied")
    edits_applied += 1
    return content.replace(old, new, 1)


# ============ Edit 1: Import at top ============
old1 = """from smart_axis_extractor import SmartAxisExtractor

try:
    from vlm_verifier import VLMVerifier, ANTHROPIC_AVAILABLE
    VLM_VERIFIER_AVAILABLE = True
except Exception as _vlm_err:
    VLM_VERIFIER_AVAILABLE = False
    ANTHROPIC_AVAILABLE = False
    print(f"VLM verifier not available: {_vlm_err}")"""

new1 = """from smart_axis_extractor import SmartAxisExtractor

try:
    from vlm_verifier import VLMVerifier, ANTHROPIC_AVAILABLE
    VLM_VERIFIER_AVAILABLE = True
except Exception as _vlm_err:
    VLM_VERIFIER_AVAILABLE = False
    ANTHROPIC_AVAILABLE = False
    print(f"VLM verifier not available: {_vlm_err}")

try:
    from line_marker_detector import find_markers_on_traced_line
    MARKER_DETECTOR_AVAILABLE = True
except Exception as _md_err:
    MARKER_DETECTOR_AVAILABLE = False
    print(f"line_marker_detector not available: {_md_err}")"""

content = replace_unique(content, old1, new1, "import")


# ============ Edit 2: Add detect_markers_btn = None initialization ============
old2 = """    export_sd_btn = None
    export_wpd_btn = None
    verify_btn = None"""

new2 = """    export_sd_btn = None
    export_wpd_btn = None
    verify_btn = None
    detect_markers_btn = None"""

content = replace_unique(content, old2, new2, "init_none")


# ============ Edit 3: Add to nonlocal in process_image ============
old3 = "        nonlocal export_sd_btn, export_wpd_btn, verify_btn"
new3 = "        nonlocal export_sd_btn, export_wpd_btn, verify_btn, detect_markers_btn"

content = replace_unique(content, old3, new3, "nonlocal")


# ============ Edit 4: Enable button after extraction ============
old4 = """            export_sd_btn.disabled = False
            export_wpd_btn.disabled = False
            verify_btn.disabled = not (VLM_VERIFIER_AVAILABLE and ANTHROPIC_AVAILABLE)"""

new4 = """            export_sd_btn.disabled = False
            export_wpd_btn.disabled = False
            verify_btn.disabled = not (VLM_VERIFIER_AVAILABLE and ANTHROPIC_AVAILABLE)
            detect_markers_btn.disabled = not MARKER_DETECTOR_AVAILABLE"""

content = replace_unique(content, old4, new4, "enable_btn")


# ============ Edit 5: Define detect_markers_btn next to verify_btn ============
old5 = """    verify_btn = ft.OutlinedButton(
        "Verify with AI", icon=ft.icons.AUTO_FIX_HIGH, disabled=True,
        tooltip="Send current extraction to Claude to fix missing/stray points. Needs ANTHROPIC_API_KEY.",
    )"""

new5 = """    verify_btn = ft.OutlinedButton(
        "Verify with AI", icon=ft.icons.AUTO_FIX_HIGH, disabled=True,
        tooltip="Send current extraction to Claude to fix missing/stray points. Needs ANTHROPIC_API_KEY.",
    )

    detect_markers_btn = ft.OutlinedButton(
        "Detect Markers", icon=ft.icons.RADIO_BUTTON_CHECKED, disabled=True,
        tooltip="Replace each line's points with actual marker positions, detected via color-thickness peaks along the LineFormer trace. Best for sparse marker charts (e.g. thermoelectric data).",
    )"""

content = replace_unique(content, old5, new5, "btn_define")


# ============ Edit 6: Add on_detect_markers_click handler after verify_btn.on_click ============
old6 = "    verify_btn.on_click = on_verify_click"

new6 = """    verify_btn.on_click = on_verify_click

    def on_detect_markers_click(_):
        if app.current_image is None or not app.data_series:
            return
        if not MARKER_DETECTOR_AVAILABLE:
            process_status_text.value = "line_marker_detector module not installed."
            page.update()
            return
        app.selected_line_idx = None
        app.edit_mode = None
        app.add_anchors = []
        edit_panel.visible = False
        _hide_edit_subcontrols()
        process_status_text.value = "Detecting markers along each line..."
        process_progress_ring.visible = True
        page.update()

        def run_detect():
            try:
                new_series_unsorted = []
                total_markers = 0
                no_marker_lines = 0
                traces = app.raw_lines or []

                for curve in traces:
                    if len(curve) < 3:
                        new_series_unsorted.append({"points": []})
                        continue
                    markers, _color_lab, _thick = find_markers_on_traced_line(
                        app.current_image, curve,
                        color_tol_lab=12.0,
                        min_marker_thickness=3.5,
                        min_peak_distance_px=10,
                    )
                    if markers:
                        pts = [[int(round(x)), int(round(y))] for (x, y) in markers]
                        new_series_unsorted.append({"points": pts})
                        total_markers += len(pts)
                    else:
                        # Fallback: keep downsampled points for lines without clear markers
                        pts = app.downsample_points(curve)
                        new_series_unsorted.append({"points": pts})
                        no_marker_lines += 1

                app.data_series = app.sort_data_series(new_series_unsorted)
                app.result_image = app.draw_points_on_image(
                    app.current_image, app.data_series, app.axis_config
                )
                result_image.src_base64 = image_to_base64(
                    cv2.cvtColor(app.result_image, cv2.COLOR_BGR2RGB)
                )
                result_image.visible = True

                total_pts = sum(len(s["points"]) for s in app.data_series)
                line_pts = [len(s["points"]) for s in app.data_series]
                msg = (f"{len(app.data_series)} lines, {total_pts} marker points\\n"
                       f"Points per line: {', '.join(map(str, line_pts))}")
                if no_marker_lines:
                    msg += f"\\n({no_marker_lines} lines had no clear markers \u2014 using downsampled points instead)"
                info_text.value = msg
                populate_detected_lines()
                update_data_table()
                process_status_text.value = (
                    f"Detected {total_markers} markers across "
                    f"{len(app.data_series) - no_marker_lines} lines."
                )
            except Exception as ex:
                import traceback
                traceback.print_exc()
                process_status_text.value = f"Marker detection failed: {ex}"
            finally:
                process_progress_ring.visible = False
                page.update()

        page.run_thread(run_detect)

    detect_markers_btn.on_click = on_detect_markers_click"""

content = replace_unique(content, old6, new6, "handler")


# ============ Edit 7: Place button in settings_panel ============
old7 = """            edit_panel,
            verify_btn,
            ft.Divider(),
            ft.Text("Adjust in Digitizer", size=14, weight=ft.FontWeight.BOLD),"""

new7 = """            edit_panel,
            verify_btn,
            detect_markers_btn,
            ft.Divider(),
            ft.Text("Adjust in Digitizer", size=14, weight=ft.FontWeight.BOLD),"""

content = replace_unique(content, old7, new7, "settings_panel_placement")


# ============ Write back ============
print()
print(f"Edits applied:        {edits_applied}")
print(f"Edits already applied: {edits_already}")
print(f"Total expected: 7")

if edits_applied + edits_already == 7:
    open(TARGET, "w", encoding="utf-8").write(content)
    print(f"\n✅ All edits applied. Patched file: {TARGET}")
    print(f"   To revert: cp {BACKUP} {TARGET}")
else:
    print(f"\n⚠️  Some edits failed. File NOT modified. Check error messages above.")
    sys.exit(1)
