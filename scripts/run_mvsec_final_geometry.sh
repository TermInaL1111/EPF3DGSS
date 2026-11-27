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
    report="$directory/mvsec_geometry.json"
    if [[ -s "$report" ]] && python - "$report" "$directory/geometry/config.json" <<'PY'
import json
import sys
from pathlib import Path

report, config = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
sys.exit(not (report['step'] == config['schedule_steps'] and report['clip_views'] == 18
              and report['rows']))
PY
    then
        continue
    fi
    python -u evaluate_mvsec_geometry.py "$flight" "$stereo" "$EPF_ASSETS" "$f3" \
        "$directory/geometry/latest.pt" "$report.pending"
    mv "$report.pending" "$report"
done
python scripts/compare_dsec_long.py "$run/dsec_full_f3" "$run/dsec_full_no_f3" \
    "$run/dsec_full_comparison.json"
