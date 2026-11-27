#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=0
run=/data/userwj/epf-runs
python -u train_event_depth_boundary.py /data/userwj/datasets/EventScape "$EPF_ASSETS" \
    "$run/eventscape_direct_20k/geometry/latest.pt" "$run/eventscape_boundary_5k" \
    --steps 5000 --val-windows 96
