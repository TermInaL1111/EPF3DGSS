#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
teacher_weight=0
case "$1" in
    target) export CUDA_VISIBLE_DEVICES=0; weight=0 ;;
    source) export CUDA_VISIBLE_DEVICES=1; weight=.5 ;;
    rgb_teacher) export CUDA_VISIBLE_DEVICES=1; weight=.5; teacher_weight=.1 ;;
    *) exit 2 ;;
esac
steps="${2:-12000}"
deps=/data/userwj/projects/epf-deps
export PYTHONPATH="$PWD:$PYTHONPATH:$deps/OctreeSplat:$deps/octree-cpu-deps"
run=/data/userwj/epf-runs
initial="$run/octree_pixel_mixed_12k/latest.pt"
out="$run/source_view_supervision/${1}_${steps}"
flight="$EPF_MVSEC/indoor_flying"
python -u train_dsec_gaussians.py "$EPF_DSEC" "$EPF_ASSETS" "$deps/fast-feature-fields" \
    "$run/eventscape_realm_20k/geometry/latest.pt" "$out" \
    --eventscape-root /data/userwj/datasets/EventScape \
    --eventscape-cameras "$run/eventscape_rgbd_cameras" \
    --dsec-temporal-cameras "$run/dsec_temporal_cameras" \
    --head octree_pixel --geometry predicted --known-source-intrinsics --target-offset 4 \
    --init-head "$initial" --source-view-weight "$weight" --rgb-teacher-weight "$teacher_weight" \
    --steps "$steps" --lr 1e-4 \
    --lpips --seed 11 --val-limit 144 --save-every 4000
python -u dvlt_real_transfer.py "$flight" "$flight/calibration/camchain-imucam-indoor_flying.yaml" \
    "$EPF_ASSETS" "$out/latest.pt" "$deps/fast-feature-fields" "$EPF_ASSETS/f3.pth" \
    "$out/mvsec.json" --known-source-intrinsics --calibrated-source-control --lpips
