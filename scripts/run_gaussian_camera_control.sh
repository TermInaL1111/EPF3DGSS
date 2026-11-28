#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=1
deps=/data/userwj/projects/epf-deps
export PYTHONPATH="$PYTHONPATH:$deps/OctreeSplat:$deps/octree-cpu-deps"
run=/data/userwj/epf-runs
out="$run/multiscene_gaussian_camera_control"
flags=()
if [[ "${1:-}" == --components ]]; then
    out="${out}_components"
    flags=(--source-camera-components)
elif [[ "${1:-}" == --depth ]]; then
    out="${out}_depth"
    flags=(--source-depth-control)
elif [[ $# -ne 0 ]]; then
    exit 2
fi

python -u train_dsec_gaussians.py "$EPF_DSEC" "$EPF_ASSETS" "$deps/fast-feature-fields" \
    "$run/eventscape_realm_20k/geometry/latest.pt" "$out" \
    --eventscape-root /data/userwj/datasets/EventScape \
    --eventscape-cameras "$run/eventscape_rgbd_cameras" \
    --dsec-temporal-cameras "$run/dsec_temporal_cameras" \
    --geometry predicted --head octree --steps 0 --val-limit 1 --lpips --source-camera-control \
    --init-head "$run/multiscene_gaussians/octree/latest.pt" "${flags[@]}"
