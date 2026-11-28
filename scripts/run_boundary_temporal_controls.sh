#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3
run=/data/userwj/epf-runs
out="$run/eventscape_boundary_5k"
for dataset in eventscape dsec mvsec; do
    case "$dataset" in
        eventscape) root=/data/userwj/datasets/EventScape; frames=1;;
        dsec) root="$EPF_DSEC"; frames=24;;
        mvsec) root="$EPF_MVSEC/indoor_flying"; frames=24;;
    esac
    report="$out/${dataset}_temporal_controls.json"
    if [[ -s "$report" ]]; then continue; fi
    python -u evaluate_event_depth_boundary.py "$dataset" "$root" "$EPF_ASSETS" \
        "$run/eventscape_direct_20k/geometry/latest.pt" "$out/latest.pt" \
        "$report" --frames "$frames" --device cpu --temporal-controls
done
