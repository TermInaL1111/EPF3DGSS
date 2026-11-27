#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=1
deps=/data/userwj/projects/epf-deps
export PYTHONPATH="$PYTHONPATH:$deps/OctreeSplat:$deps/octree-cpu-deps"
root=/data/userwj/datasets/EventScape
run=/data/userwj/epf-runs
cameras="$run/eventscape_rgbd_cameras"
geometry="$run/eventscape_realm_20k/geometry/latest.pt"
out="$run/eventscape_realm_gaussians"
python - "$run/eventscape_realm_20k/town05_depth.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
if report['step'] != 20000 or report['windows'] != 1368:
    raise ValueError('EventScape distillation or matched depth evaluation is incomplete')
PY
for head in fixed octree; do
    python -u train_dsec_gaussians.py "$root" "$EPF_ASSETS" "$deps/fast-feature-fields" \
        "$geometry" "$out/$head" --eventscape-cameras "$cameras" \
        --geometry predicted --head "$head" --steps 5000 --val-limit 96 --lpips
done
python scripts/summarize_eventscape_render.py "$cameras/test.json" "$out"
