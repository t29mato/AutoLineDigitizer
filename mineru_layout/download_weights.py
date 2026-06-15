# -*- coding: utf-8 -*-
"""Download only the PP-DocLayoutV2 weights from PDF-Extract-Kit-1.0.

Usage:
    python download_weights.py ./pp_doclayoutv2_weights

Behind the NIMS proxy, export first:
    export HTTPS_PROXY=http://proxyout.nims.go.jp:8888
    export HF_HUB_ETAG_TIMEOUT=30
If TLS interception breaks huggingface.co, download on another network and
copy the folder over — the model loads from a plain local directory.
"""
import sys
from huggingface_hub import snapshot_download

REPO = "opendatalab/PDF-Extract-Kit-1.0"
SUBDIR = "models/Layout/PP-DocLayoutV2"

def main(target_dir: str):
    path = snapshot_download(
        repo_id=REPO,
        allow_patterns=[f"{SUBDIR}/*"],
        local_dir=target_dir,
    )
    print(f"Done. Pass this directory to ChartDetector:\n  {path}/{SUBDIR}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "./pp_doclayoutv2_weights")
