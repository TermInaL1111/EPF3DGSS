#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=0
deps=/data/userwj/projects/epf-deps
export PYTHONPATH="$PYTHONPATH:$deps/OctreeSplat:$deps/octree-cpu-deps"
out=/data/userwj/epf-runs/multiscene_gaussians
flight="$EPF_MVSEC/indoor_flying"
for head in fixed octree; do
    python -u dvlt_real_transfer.py "$flight" "$flight/calibration/camchain-imucam-indoor_flying.yaml" \
        "$EPF_ASSETS" "$out/$head/latest.pt" "$deps/fast-feature-fields" "$EPF_ASSETS/f3.pth" \
        "$out/${head}_input_paths.json" --lpips --calibrated-source-control --input-path-control
done
