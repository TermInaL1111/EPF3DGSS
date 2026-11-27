#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=
out=/data/userwj/epf-runs/dsec_temporal_cameras
mkdir -p "$out"
for split in test val train; do
    mapfile -t names < <(python -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))[sys.argv[2]]))' scripts/dsec_full_split.json "$split")
    (( ${#names[@]} > 0 ))
    count=24
    if [[ "$split" == train ]]; then count=96; fi
    python -u dsec_temporal_gaussian_data.py "$EPF_DSEC" "$out/$split.json" --windows "$count" --sequences "${names[@]}"
done
