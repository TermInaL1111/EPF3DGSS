#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=1
deps=/data/userwj/projects/epf-deps
export PYTHONPATH="$PWD:$PYTHONPATH:$deps/OctreeSplat:$deps/octree-cpu-deps"
run=/data/userwj/epf-runs/source_view_supervision
flight="$EPF_MVSEC/indoor_flying"
for arm in target source; do
    out="$run/${arm}_12000"
    python -u dvlt_real_transfer.py "$flight" "$flight/calibration/camchain-imucam-indoor_flying.yaml" \
        "$EPF_ASSETS" "$out/latest.pt" "$deps/fast-feature-fields" "$EPF_ASSETS/f3.pth" \
        "$out/mvsec_preview.json" --known-source-intrinsics --calibrated-source-control \
        --preview-dir "$out/mvsec_preview"
done
MPLBACKEND=Agg python scripts/plot_source_view_supervision.py "$run" "$run/source_view_comparison.png"
