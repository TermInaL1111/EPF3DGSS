"""Report paired cross-view fusion effects for the 128×96 scene-camera runs."""
import json
from pathlib import Path
from statistics import mean
from teacher_ablation import scene_macro

root = Path(__file__).parent
print("| Cross-view appearance | PSNR | Foreground PSNR | Foreground gain over zero | Aligned depth AbsRel |")
print("|---|---:|---:|---:|---:|")
for method in ("none","geometry","event"):
    rows = []
    for seed in (10,11,12):
        path = root/f"scene_dense_{method}_{seed}"
        scores = json.loads((path/"audit.json").read_text())["arms"]["0.1"]
        depth = json.loads((path/"student_depth.json").read_text())
        fg = scores["matched_foreground_psnr"]
        rows.append([scores["matched"],fg,fg-scores["zeros_foreground_psnr"],
                     scene_macro(depth,"matched_aligned_abs_rel")])
    print("| "+method+" | "+" | ".join(f"{mean(x):.4f}" for x in zip(*rows))+" |")
