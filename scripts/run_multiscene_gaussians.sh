#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=0
deps=/data/userwj/projects/epf-deps
export PYTHONPATH="$PYTHONPATH:$deps/OctreeSplat:$deps/octree-cpu-deps"
run=/data/userwj/epf-runs
out="$run/multiscene_gaussians"
flight="$EPF_MVSEC/indoor_flying"
calib="$flight/calibration/camchain-imucam-indoor_flying.yaml"
f3="$deps/fast-feature-fields"
mkdir -p "$out"

transfer() {
    local label=$1 weights=$2
    python -u dvlt_real_transfer.py "$flight" "$calib" "$EPF_ASSETS" "$weights" \
        "$f3" "$EPF_ASSETS/f3.pth" "$out/${label}_mvsec.json" --lpips
    python scripts/summarize_render.py "$out/${label}_mvsec.json" "$out/${label}_mvsec_summary.json"
    python scripts/summarize_evggs.py "$run/evggs/mvsec_shared.json" "$out/${label}_vs_evggs.json" \
        --epf "$out/${label}_mvsec.json"
}

for domain in dsec_temporal eventscape_realm; do
    transfer "$domain" "$run/${domain}_gaussians/octree/latest.pt"
done
for head in fixed octree; do
    python -u train_dsec_gaussians.py "$EPF_DSEC" "$EPF_ASSETS" "$f3" \
        "$run/eventscape_realm_20k/geometry/latest.pt" "$out/$head" \
        --eventscape-root /data/userwj/datasets/EventScape \
        --eventscape-cameras "$run/eventscape_rgbd_cameras" \
        --dsec-temporal-cameras "$run/dsec_temporal_cameras" \
        --geometry predicted --head "$head" --steps 12000 --val-limit 144 --lpips
    transfer "mixed_$head" "$out/$head/latest.pt"
done
