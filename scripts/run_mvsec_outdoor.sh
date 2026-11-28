#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=1
run=/data/userwj/epf-runs
adaptation="$run/mvsec_outdoor_day1_to_night1"
base="$run/eventscape_realm_20k/geometry/latest.pt"
geometry="$adaptation/geometry/latest.pt"
day="$EPF_MVSEC/outdoor_day"
night="$EPF_MVSEC/outdoor_night"
f3=/data/userwj/projects/epf-deps/fast-feature-fields
if [[ $# -eq 0 ]]; then
    flight=outdoor_night1
    flags=()
    if [[ -s "$geometry" ]]; then flags=(--resume); fi
    python -u train_dsec.py "$day" "$EPF_ASSETS" "$adaptation/geometry" \
        --mvsec-calibration "$day/outdoor_day_calib.zip" --init-geometry "$base" \
        --step-gating saag --epochs 1 --max-views 18 --val-limit 96 --snapshot-every 1000 "${flags[@]}"
elif [[ $# -eq 2 && "$1" == --evaluate-night && "$2" =~ ^outdoor_night[123]$ ]]; then
    flight="$2"
    if [[ ! -s "$geometry" ]]; then
        echo "Missing day1 adaptation checkpoint: $geometry" >&2
        exit 1
    fi
else
    echo "Usage: $0 [--evaluate-night outdoor_night{1,2,3}]" >&2
    exit 2
fi
out="$run/mvsec_outdoor_day1_to_${flight#outdoor_}"
mkdir -p "$out"
for arm in teacher base adapted; do
    checkpoint="$base"
    flags=()
    case "$arm" in
        teacher) flags=(--teacher);;
        adapted) checkpoint="$geometry";;
    esac
    python -u evaluate_mvsec_geometry.py "$night" "$night/outdoor_night_calib.zip" "$EPF_ASSETS" \
        "$f3" "$checkpoint" "$out/$arm.json" --flights "$flight" --full-fov --max-depth 80 \
        --cloud-metrics "${flags[@]}"
done
python scripts/summarize_mvsec_outdoor.py "$out"
