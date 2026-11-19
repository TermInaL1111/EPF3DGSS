"""Plot held-out scene means for the fixed-spacing EV3DS pilot."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def draw():
    here = Path(__file__).parent
    rows = json.loads((here.parent/"results/baseline_shift_fg10.json").read_text())
    fig,axes = plt.subplots(1,2,figsize=(7.6,3.0),layout="constrained")
    for reverse,color,marker,label in [(False,"#2166ac","o","Chronological"),
                                       (True,"#b35806","s","Reversed")]:
        for ax,metric in zip(axes,("matched_rotation_deg","matched_psnr")):
            gaps = [2,4,8,16]
            groups = [[r[metric] for r in rows if r["gap"]==gap and r["reverse"]==reverse]
                      for gap in gaps]
            ax.plot(range(4),np.mean(groups,axis=1),color=color,marker=marker,label=label)
            ax.set_xticks(range(4),gaps)
            ax.set_xlabel("Source spacing (frames)")
            ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Rotation error (degrees)")
    axes[1].set_ylabel("Full-frame PSNR (dB)")
    axes[0].legend(frameon=False)
    fig.savefig(here/"spacing_diagnostic.pdf")
    plt.close(fig)


if __name__ == "__main__":
    draw()
