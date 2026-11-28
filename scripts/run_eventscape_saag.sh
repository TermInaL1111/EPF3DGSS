#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=1
root=/data/userwj/datasets/EventScape
run=/data/userwj/epf-runs
out="$run/eventscape_saag_20k"
flags=()
if [[ -s "$out/geometry/latest.pt" ]]; then flags=(--resume); fi
python -u train_dsec.py "$root" "$EPF_ASSETS" "$out/geometry" --eventscape \
    --max-steps 20000 --max-views 18 --val-limit 96 --snapshot-every 1000 --step-gating saag "${flags[@]}"
python -u evaluate_eventscape_depth.py "$root/Town05_test.zip" "$EPF_ASSETS" \
    "$out/geometry/latest.pt" "$out/town05_depth.json"
python scripts/summarize_eventscape_depth.py "$run" --arm saag --arm realm \
    --out "$run/eventscape_distillation_ablation.json"
