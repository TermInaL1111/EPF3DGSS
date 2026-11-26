#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
run=/data/userwj/epf-runs
root=$(dirname "$EPF_DSEC")
mapfile -t sequences < <(python -c 'import json; d=json.load(open("scripts/dsec_full_split.json")); print("\n".join(d["train"]+d["val"]+d["test"]))')
(( ${#sequences[@]} == 41 ))
# The extractor skips already complete files and publishes each file atomically.
nice -n 10 python -u scripts/extract_dsec_sequences.py "$root" "${sequences[@]}"
python - <<'PY'
import json, os
from pathlib import Path
from dsec_data import DSECTemporalPairs
split = json.loads(Path('scripts/dsec_full_split.json').read_text())
for group in ('train','val','test'):
    counts = {name:len(DSECTemporalPairs(os.environ['EPF_DSEC'],name).pairs)
              for name in split[group]}
    if not all(counts.values()):
        raise ValueError(f'Empty sequence in {group}: {counts}')
    print(group, sum(counts.values()), counts, flush=True)
PY
bash scripts/run_dsec_long.sh f3 > "$run/dsec_full_f3.log" 2>&1 &
f3_pid=$!
bash scripts/run_dsec_long.sh no_f3 > "$run/dsec_full_no_f3.log" 2>&1 &
no_f3_pid=$!
status=0
wait "$f3_pid" || status=1
wait "$no_f3_pid" || status=1
(( status == 0 ))
python scripts/compare_dsec_long.py "$run/dsec_full_f3" "$run/dsec_full_no_f3" "$run/dsec_full_comparison.json"
