"""Average paired seed, window and flight scores with equal weight for every flight."""
from pathlib import Path
import json
from statistics import mean

root = Path(__file__).parent
print("| Real-flight teacher | Matched PSNR | Zero PSNR | Wrong-time PSNR | Rotation error (°) |")
print("|---|---:|---:|---:|---:|")
for name in ("none","aligned","wrong_training_time"):
    rows = []
    for flight in (1,2,3):
        report = json.loads((root/f"fold{flight}.json").read_text())
        assert len(report["train"])==128 and len(report["validation"])==32
        assert not {k[0] for k in report["train"]}&{k[0] for k in report["validation"]}
        for seed in (10,11,12):
            stats = report["arms"][f"{name}_{seed}"]
            rows.append([mean(row[index] for row in stats[key]) for key,index in
                         (("matched",0),("zero",0),("wrong_time",0),("matched",1))])
    print("| "+name+" | "+" | ".join(f"{mean(col):.3f}" for col in zip(*rows))+" |")
