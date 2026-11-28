#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=0
deps=/data/userwj/projects/epf-deps
export PYTHONPATH="$PYTHONPATH:$deps/OctreeSplat:$deps/octree-cpu-deps"
run=/data/userwj/epf-runs
boundary="$run/eventscape_boundary_5k/latest.pt"
flight="$EPF_MVSEC/indoor_flying"
calib="$flight/calibration/camchain-imucam-indoor_flying.yaml"
prefix=mvsec_boundary_octree
scale=()
if [[ "${1:-}" == --preserve-boundary-scale ]]; then
    prefix=mvsec_boundary_scale_octree
    scale=(--preserve-boundary-scale)
fi
for arm in geometry event; do
    out="$run/${prefix}_$arm"
    flags=()
    if [[ "$arm" == event ]]; then flags=(--event-leaves); fi
    python -u train_dsec_gaussians.py "$flight" "$EPF_ASSETS" "$deps/fast-feature-fields" \
        "$run/dsec_full_no_f3/geometry/latest.pt" "$out/gs" --mvsec-calibration "$calib" \
        --train indoor_flying4 --val indoor_flying1 indoor_flying2 indoor_flying3 \
        --geometry predicted --head octree --steps 5000 --val-limit 24 --lpips \
        --boundary "$boundary" "${flags[@]}" "${scale[@]}"
    python -u dvlt_real_transfer.py "$flight" "$calib" "$EPF_ASSETS" "$out/gs/latest.pt" \
        "$deps/fast-feature-fields" "$EPF_ASSETS/f3.pth" "$out/shared_128x96.json" --lpips
done
python - "$run" "$prefix" <<'PY'
import json
import sys
from pathlib import Path
from statistics import fmean

root = Path(sys.argv[1])
prefix = sys.argv[2]
reports = [json.loads((root/f'{prefix}_{arm}/shared_128x96.json').read_text())
           for arm in ('geometry','event')]
keys = [[(row['flight'],row['start']) for row in report] for report in reports]
if len(set(keys[0])) != 72 or keys[0] != keys[1]:
    raise ValueError('Expected the same 72 MVSEC windows in both arms')
metrics = dict(psnr=1, ssim=1, lpips_vgg=-1, lpips_alex=-1)
summaries = {}
for condition in ('matched','zero'):
    a, b = ([row[condition] for row in report] for report in reports)
    counts = [row['gaussian_count'] for row in a]
    if min(counts) < 1 or counts != [row['gaussian_count'] for row in b]:
        raise ValueError(f'Unequal Gaussian counts for {condition}')
    scores = {}
    for metric, sign in metrics.items():
        geometry, events = (fmean(row[metric] for row in items) for items in (a, b))
        scores[metric] = dict(geometry=geometry,event=events,delta=events-geometry,
                              improvement_pct=100*sign*(events-geometry)/abs(geometry) if geometry else None)
    summaries[condition] = dict(mean_gaussian_count=fmean(counts),metrics=scores)
report = dict(windows=len(keys[0]), train_steps=5000, summary=summaries)
out = root/f'{prefix}_comparison.json'
tmp = out.with_suffix('.tmp')
tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
tmp.replace(out)
PY
