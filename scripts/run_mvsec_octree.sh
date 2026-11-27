#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=0
deps=/data/userwj/projects/epf-deps
export PYTHONPATH="$PYTHONPATH:$deps/OctreeSplat:$deps/octree-cpu-deps"
run=/data/userwj/epf-runs
out="$run/mvsec_octree_no_f3"
flight="$EPF_MVSEC/indoor_flying"
calib="$flight/calibration/camchain-imucam-indoor_flying.yaml"
python -u train_dsec_gaussians.py "$flight" "$EPF_ASSETS" "$deps/fast-feature-fields" \
    "$run/dsec_full_no_f3/geometry/latest.pt" "$out/gs" --mvsec-calibration "$calib" \
    --train indoor_flying4 --val indoor_flying1 indoor_flying2 indoor_flying3 \
    --geometry predicted --head octree --steps 5000 --val-limit 24 --lpips
python -u dvlt_real_transfer.py "$flight" "$calib" "$EPF_ASSETS" "$out/gs/latest.pt" \
    "$deps/fast-feature-fields" "$EPF_ASSETS/f3.pth" "$out/shared_128x96.json" --lpips
python scripts/summarize_render.py "$out/shared_128x96.json" "$out/mvsec_summary.json"
