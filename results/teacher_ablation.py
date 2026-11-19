"""Reproduce teacher ablation rows. Macro-average frames within scenes, then seeds."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).parent


def scene_macro(rows,key):
    scenes = defaultdict(list)
    for row in rows:
        scenes[row["scene"]].append(row[key])
    return mean(mean(items) for items in scenes.values())


def collect(prefix):
    values = []
    for seed in (10,11,12):
        folder = ROOT / f"{prefix}{seed}"
        audit = json.loads((folder/"audit.json").read_text())["arms"]["0.1"]
        depth = json.loads((folder/"student_depth.json").read_text())
        values.append([audit["matched"],audit["matched_foreground_psnr"],
                       audit["matched_foreground_psnr"]-audit["zeros_foreground_psnr"],
                       scene_macro(depth,"matched_aligned_abs_rel")])
    return [mean(col) for col in zip(*values)]


if __name__ == "__main__":
    print("| Teacher target | Full PSNR | Foreground PSNR | Foreground gain over zero | Depth AbsRel ↓ |")
    print("|---|---:|---:|---:|---:|")
    for prefix,name in [("f3rgb","Low-resolution image tokens"),("f3_encoder_","Image tokens"),
                        ("f3_loop_","Multiview tokens"),("f3_shuffled_","Multiview, wrong scene"),
                        ("f3_depth_uniform_","Multiview + relative depth"),
                        ("f3_depth_event_","Multiview + event-weighted relative depth")]:
        print("| "+name+" | "+" | ".join(f"{v:.4f}" for v in collect(prefix))+" |")
