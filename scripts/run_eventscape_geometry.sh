#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=0
run=/data/userwj/epf-runs
root=/data/userwj/datasets/EventScape
python -u evaluate_eventscape_geometry.py "$root/Town05_test.zip" "$EPF_ASSETS" \
    "$run/eventscape_rgbd_cameras/geometry_test.json" \
    "$run/eventscape_direct_20k/geometry/latest.pt" "$run/eventscape_realm_20k/geometry/latest.pt" \
    "$run/eventscape_geometry_with_saag.json" \
    --saag "$run/eventscape_saag_20k/geometry/latest.pt"
