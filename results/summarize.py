"""Build the manuscript table from the recorded input-sensitivity audits."""
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).parent
print("| Training objective | Teacher weight | Full image | Foreground | Foreground, permuted | Foreground, zero |")
print("|---|---:|---:|---:|---:|---:|")
for prefix, name in [("run","Full frame"), ("fg","Full frame + foreground")]:
    records = [json.loads((root/f"{prefix}{seed}"/"audit.json").read_text()) for seed in (10,11,12)]
    for weight in ("0.0","0.1"):
        keys = ("matched","matched_foreground_psnr","other_scene_same_time_foreground_psnr","zeros_foreground_psnr")
        scores = [mean(record["arms"][weight][key] for record in records) for key in keys]
        print("|",name,"|",weight,"|"," | ".join(f"{s:.3f}" for s in scores),"|")
