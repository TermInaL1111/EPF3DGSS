#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=1
run=/data/userwj/epf-runs
f3=/data/userwj/projects/epf-deps/fast-feature-fields

case "${1:-}" in
    smoke) output="$run/dsec_temporal_geometry_smoke.json"; flags=(--limit 1);;
    depth) output="$run/dsec_temporal_geometry_depth.json"; flags=();;
    lidar)
        output="$run/dsec_temporal_geometry_lidar_scores.json"
        flags=(--lidar-reference "$run/dsec_temporal_geometry_lidar.json")
        ;;
    *) exit 2;;
esac
python -u evaluate_dsec_temporal_geometry.py "$EPF_DSEC" "$EPF_ASSETS" \
    "$run/dsec_temporal_geometry_clips.json" "$f3" "$output" \
    "$run/eventscape_direct_20k/geometry/latest.pt" "$run/eventscape_saag_20k/geometry/latest.pt" \
    "$run/eventscape_realm_20k/geometry/latest.pt" "$run/dsec_full_no_f3/geometry/latest.pt" \
    "$run/dsec_full_f3/geometry/latest.pt" "${flags[@]}"
