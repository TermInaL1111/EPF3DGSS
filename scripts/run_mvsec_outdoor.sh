#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=1
run=/data/userwj/epf-runs
out="$run/mvsec_outdoor_day1_to_night1"
base="$run/eventscape_realm_20k/geometry/latest.pt"
geometry="$out/geometry/latest.pt"
day="$EPF_MVSEC/outdoor_day"
night="$EPF_MVSEC/outdoor_night"
f3=/data/userwj/projects/epf-deps/fast-feature-fields
flags=()
if [[ -s "$geometry" ]]; then flags=(--resume); fi
python -u train_dsec.py "$day" "$EPF_ASSETS" "$out/geometry" \
    --mvsec-calibration "$day/outdoor_day_calib.zip" --init-geometry "$base" \
    --step-gating saag --epochs 1 --max-views 18 --val-limit 96 --snapshot-every 1000 "${flags[@]}"
for arm in teacher base adapted; do
    checkpoint="$base"
    flags=()
    case "$arm" in
        teacher) flags=(--teacher);;
        adapted) checkpoint="$geometry";;
    esac
    python -u evaluate_mvsec_geometry.py "$night" "$night/outdoor_night_calib.zip" "$EPF_ASSETS" \
        "$f3" "$checkpoint" "$out/$arm.json" --flights outdoor_night1 --full-fov --max-depth 80 \
        --cloud-metrics "${flags[@]}"
done
python scripts/summarize_mvsec_outdoor.py "$out"
