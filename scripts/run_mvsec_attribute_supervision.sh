#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
case "$1" in
    photometric) export CUDA_VISIBLE_DEVICES=0; weight=0;;
    perceptual) export CUDA_VISIBLE_DEVICES=1; weight=.5;;
    *) exit 2;;
esac
deps=/data/userwj/projects/epf-deps
export PYTHONPATH="$PYTHONPATH:$deps/OctreeSplat:$deps/octree-cpu-deps"
run=/data/userwj/epf-runs
out="$run/multiscene_attribute_supervision/$1"
flight="$EPF_MVSEC/indoor_flying"
python -u train_dsec_gaussians.py "$EPF_DSEC" "$EPF_ASSETS" "$deps/fast-feature-fields" \
    "$run/eventscape_realm_20k/geometry/latest.pt" "$out" \
    --eventscape-root /data/userwj/datasets/EventScape \
    --eventscape-cameras "$run/eventscape_rgbd_cameras" \
    --dsec-temporal-cameras "$run/dsec_temporal_cameras" \
    --geometry predicted --head octree --steps 4034 --val-limit 144 --lpips \
    --init-head "$run/multiscene_gaussians/octree/latest.pt" --seed 11 --lr 1e-4 --lpips-weight "$weight"
python -u dvlt_real_transfer.py "$flight" "$flight/calibration/camchain-imucam-indoor_flying.yaml" \
    "$EPF_ASSETS" "$out/latest.pt" "$deps/fast-feature-fields" "$EPF_ASSETS/f3.pth" \
    "$out/mvsec.json" --lpips --calibrated-source-control
