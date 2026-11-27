#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=0
deps=/data/userwj/projects/epf-deps
export PYTHONPATH="$PYTHONPATH:$deps/OctreeSplat:$deps/octree-cpu-deps"
run=/data/userwj/epf-runs
cameras="$run/dsec_temporal_cameras"
geometry="$run/dsec_full_no_f3/geometry/latest.pt"
out="$run/dsec_temporal_gaussians"
for head in fixed octree; do
    python -u train_dsec_gaussians.py "$EPF_DSEC" "$EPF_ASSETS" "$deps/fast-feature-fields" \
        "$geometry" "$out/$head" --dsec-temporal-cameras "$cameras" \
        --geometry predicted --head "$head" --steps 5000 --val-limit 144 --lpips
done
python scripts/summarize_eventscape_render.py "$cameras/test.json" "$out" --dsec
