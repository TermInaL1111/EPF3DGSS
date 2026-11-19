"""Plot mean within-foreground depth rank correlation for each held-out object."""
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

paper = Path(__file__).resolve().parent
results = paper.parent / "results" / "event_dvlt"
baseline = json.loads((results / "geometry.json").read_text())
geometry = json.loads((results / "geometry_supervised.json").read_text())
assert [(r["scene"], r["start"]) for r in baseline] == [(r["scene"], r["start"]) for r in geometry]
scenes = sorted({r["scene"] for r in baseline})
fig, ax = plt.subplots(figsize=(7.4, 3.1), layout="constrained")
for rows, name, label, marker in (
    (baseline, "rgb_teacher", "RGB teacher", "o"),
    (baseline, "event_student", "Token supervision", "s"),
    (geometry, "event_student", "Token + geometry supervision", "^"),
):
    averages = [
        mean(v for row in rows if row["scene"] == scene
             for v in row[name]["foreground_depth_spearman"] if v is not None)
        for scene in scenes
    ]
    ax.scatter(range(len(scenes)), averages, label=label, marker=marker, s=45)
ax.set_xticks(range(len(scenes)), labels=scenes)
ax.set_ylabel("Foreground depth Spearman ρ")
ax.axhline(0, linewidth=0.7, color="gray")
ax.set_ylim(-1, 1.03)
ax.legend(loc="lower center", ncols=3, fontsize=8)
fig.savefig(paper / "geometry_transfer.png", dpi=180)
