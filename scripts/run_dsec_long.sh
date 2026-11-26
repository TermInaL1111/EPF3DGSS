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
run="/data/userwj/epf-runs/dsec_long_$1"
mkdir -p "$run"
train=(--train zurich_city_02_a interlaken_00_c zurich_city_00_a)
val=(--val zurich_city_01_a)
python -u train_dsec.py "$EPF_DSEC" "$EPF_ASSETS" "$run/geometry" "${train[@]}" "${val[@]}" --realm --max-steps 20000 --max-views 18 --val-limit 48 "${features[@]}"
python -u evaluate_dsec_geometry.py "$EPF_DSEC" "$EPF_ASSETS" "$f3" "$run/geometry/latest.pt" "$run/depth.json" --sequences zurich_city_01_a thun_00_a --count 48
python -u train_dsec_gaussians.py "$EPF_DSEC" "$EPF_ASSETS" "$f3" "$run/geometry/latest.pt" "$run/gaussians" "${train[@]}" "${val[@]}" --geometry predicted --steps 5000 --val-limit 48 --lpips
python -u dvlt_real_transfer.py "$EPF_MVSEC/indoor_flying" "$EPF_MVSEC/indoor_flying/calibration/camchain-imucam-indoor_flying.yaml" "$EPF_ASSETS" "$run/gaussians/latest.pt" "$f3" "$EPF_ASSETS/f3.pth" "$run/mvsec.json" --lpips
python scripts/summarize_render.py "$run/gaussians/val_005000.json" "$run/dsec_render_summary.json"
python scripts/summarize_render.py "$run/mvsec.json" "$run/mvsec_summary.json"
