#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=1
deps=/data/userwj/projects/epf-deps
export PYTHONPATH="$PYTHONPATH:$deps/OctreeSplat:$deps/octree-cpu-deps"
run=/data/userwj/epf-runs
out="$run/mvsec_shared_previews"
flight="$EPF_MVSEC/indoor_flying"
calibration="$flight/calibration/camchain-imucam-indoor_flying.yaml"
mkdir -p "$out"

python -u evaluate_evggs.py mvsec "$deps/EvGGS" "$EPF_ASSETS/evggs/CKPT/pretrain_evggs.pth" \
    "$flight" "$out/evggs.json" --calibration "$calibration" --preview-dir "$out/evggs"
python -u dvlt_real_transfer.py "$flight" "$calibration" "$EPF_ASSETS" \
    "$run/multiscene_gaussians/octree/latest.pt" "$deps/fast-feature-fields" \
    "$EPF_ASSETS/f3.pth" "$out/octree.json" --calibrated-source-control --preview-dir "$out/octree"
