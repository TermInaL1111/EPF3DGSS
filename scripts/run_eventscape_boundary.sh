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
