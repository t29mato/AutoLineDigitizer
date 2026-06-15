# mineru_layout — vendored PP-DocLayoutV2 chart detector

MinerU's layout detector (RT-DETR), vendored standalone for AutoLineDigitizer.
Key feature vs DocLayout-YOLO: a dedicated `chart` class separate from `image`
(photos / SEM / schematics), and instance-level detection that frequently
separates the panels of composite figures.

Source: https://github.com/opendatalab/MinerU (MinerU Open Source License,
Apache-2.0 based; see MINERU_LICENSE.md). Files `pp_doclayoutv2.py` and
`bbox_utils.py` are copied with one import patched and the demo main replaced.

## Install
Deps (likely already in `alddev`): torch, torchvision, transformers, pillow,
numpy, tqdm, opencv-python, huggingface_hub.

## Weights (one time)
    python download_weights.py ./pp_doclayoutv2_weights
Then weights dir = ./pp_doclayoutv2_weights/models/Layout/PP-DocLayoutV2

## Smoke test
    python pp_doclayoutv2.py page.png --model <weights_dir> --device mps --output vis.png

## Use
    from mineru_layout import ChartDetector
    det = ChartDetector("<weights_dir>", device="mps")
    charts = det.detect_charts(page_bgr, gutter_split=True)

`detect_charts` returns pixel + normalized bboxes for `chart` regions with a
1% label-safety margin. `gutter_split=True` post-splits any merged composite
along whitespace gutters (recursive, handles 2x2 grids).

## Integration note (batch_extract.py)
Replace the DocLayout-YOLO figure proposal with `detect_charts`, keep your
VLM screen() as the per-crop classifier, and keep the whitespace snap. The
detector outputs are candidates — your cached_plot_area / recrop flow stays
unchanged downstream.
