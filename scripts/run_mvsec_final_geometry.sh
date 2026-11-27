#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
run=/data/userwj/epf-runs
flight="$EPF_MVSEC/indoor_flying"
stereo="$flight/calibration/camchain-imucam-indoor_flying.yaml"
f3=/data/userwj/projects/epf-deps/fast-feature-fields
for arm in f3 no_f3; do
    if [[ "$arm" == f3 ]]; then
        export CUDA_VISIBLE_DEVICES=0
    else
        export CUDA_VISIBLE_DEVICES=1
    fi
    directory="$run/dsec_full_$arm"
    python -u evaluate_mvsec_geometry.py "$flight" "$stereo" "$EPF_ASSETS" "$f3" \
        "$directory/geometry/latest.pt" "$directory/mvsec_geometry.json"
done
python scripts/compare_dsec_long.py "$run/dsec_full_f3" "$run/dsec_full_no_f3" \
    "$run/dsec_full_comparison.json"
