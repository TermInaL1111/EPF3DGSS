#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
f3=/data/userwj/projects/epf-deps/fast-feature-fields
case "${1:-}" in
  f3) export CUDA_VISIBLE_DEVICES=0; features=(--f3 "$f3");;
  no_f3) export CUDA_VISIBLE_DEVICES=1; features=();;
  *) echo 'Usage: run_dsec_long.sh f3|no_f3' >&2; exit 2;;
esac
run="/data/userwj/epf-runs/dsec_full_$1"
mkdir -p "$run"
split=scripts/dsec_full_split.json
mapfile -t train < <(python -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["train"]))' "$split")
mapfile -t val < <(python -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["val"]))' "$split")
mapfile -t depth < <(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("\n".join(d["val"]+d["test"]))' "$split")
(( ${#train[@]} == 34 && ${#val[@]} == 6 && ${#depth[@]} == 7 ))
python -u train_dsec.py "$EPF_DSEC" "$EPF_ASSETS" "$run/geometry" --train "${train[@]}" --val "${val[@]}" --realm --max-steps 20000 --cover-train --realm-progress-epochs 30 --snapshot-every 1000 --max-views 18 --val-limit 96 "${features[@]}"
python -u evaluate_dsec_geometry.py "$EPF_DSEC" "$EPF_ASSETS" "$f3" "$run/geometry/latest.pt" "$run/depth.json" --sequences "${depth[@]}" --count 24
python -u train_dsec_gaussians.py "$EPF_DSEC" "$EPF_ASSETS" "$f3" "$run/geometry/latest.pt" "$run/gaussians" --train "${train[@]}" --val "${val[@]}" --geometry predicted --steps 5000 --val-limit 96 --lpips
python -u dvlt_real_transfer.py "$EPF_MVSEC/indoor_flying" "$EPF_MVSEC/indoor_flying/calibration/camchain-imucam-indoor_flying.yaml" "$EPF_ASSETS" "$run/gaussians/latest.pt" "$f3" "$EPF_ASSETS/f3.pth" "$run/mvsec.json" --lpips
python scripts/summarize_render.py "$run/gaussians/val_005000.json" "$run/dsec_render_summary.json"
python scripts/summarize_render.py "$run/mvsec.json" "$run/mvsec_summary.json"
