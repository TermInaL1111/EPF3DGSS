#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=0
run=/data/userwj/epf-runs
python -u train_event_depth_boundary.py /data/userwj/datasets/EventScape "$EPF_ASSETS" \
    "$run/eventscape_direct_20k/geometry/latest.pt" "$run/eventscape_boundary_5k" \
    --steps 5000 --val-windows 96
for dataset in eventscape dsec mvsec; do
    case "$dataset" in
        eventscape) root=/data/userwj/datasets/EventScape; frames=1;;
        dsec) root="$EPF_DSEC"; frames=24;;
        mvsec) root="$EPF_MVSEC/indoor_flying"; frames=24;;
    esac
    python -u evaluate_event_depth_boundary.py "$dataset" "$root" "$EPF_ASSETS" \
        "$run/eventscape_direct_20k/geometry/latest.pt" "$run/eventscape_boundary_5k/latest.pt" \
        "$run/eventscape_boundary_5k/${dataset}_test.json" --frames "$frames"
done
deps=/data/userwj/projects/epf-deps
export PYTHONPATH="$PYTHONPATH:$deps/OctreeSplat:$deps/octree-cpu-deps"
flight="$EPF_MVSEC/indoor_flying"
calib="$flight/calibration/camchain-imucam-indoor_flying.yaml"
out="$run/eventscape_boundary_5k"
python -u dvlt_real_transfer.py "$flight" "$calib" "$EPF_ASSETS" "$run/mvsec_octree_no_f3/gs/latest.pt" \
    "$deps/fast-feature-fields" "$EPF_ASSETS/f3.pth" "$out/mvsec_boundary_render.json" \
    --boundary "$out/latest.pt" --lpips
python scripts/summarize_render.py "$out/mvsec_boundary_render.json" "$out/mvsec_boundary_render_summary.json" --boundary
