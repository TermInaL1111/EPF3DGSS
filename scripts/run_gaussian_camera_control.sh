#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=1
deps=/data/userwj/projects/epf-deps
export PYTHONPATH="$PYTHONPATH:$deps/OctreeSplat:$deps/octree-cpu-deps"
run=/data/userwj/epf-runs

python -u train_dsec_gaussians.py "$EPF_DSEC" "$EPF_ASSETS" "$deps/fast-feature-fields" \
    "$run/eventscape_realm_20k/geometry/latest.pt" "$run/multiscene_gaussian_camera_control" \
    --eventscape-root /data/userwj/datasets/EventScape \
    --eventscape-cameras "$run/eventscape_rgbd_cameras" \
    --dsec-temporal-cameras "$run/dsec_temporal_cameras" \
    --geometry predicted --head octree --steps 0 --val-limit 1 --lpips --source-camera-control \
    --init-head "$run/multiscene_gaussians/octree/latest.pt"
