"""Audit the shared trajectory against per-scene camera metadata on pilot triplets."""
import argparse
import json
from pathlib import Path
import numpy as np
from ev3ds_pose import camera, OPENGL_TO_OPENCV


def audit(root):
    rows = []
    for split,n in [("train",12),("val",6)]:
        for scene in (root/f"{split}_scenes.txt").read_text().split()[:n]:
            for i in range(4,60,7):
                for gap in [2,4]:
                    frames = (i,i+gap)
                    local = np.linalg.solve(*[camera(root,scene,j) for j in frames])
                    shared = np.linalg.solve(*[np.loadtxt(root/"Poses"/f"r_{j+1:04}.txt")@OPENGL_TO_OPENCV for j in frames])
                    relative = local[:3,:3].T @ shared[:3,:3]
                    angle = np.rad2deg(np.arccos(np.clip((relative.trace()-1)/2,-1,1)))
                    rows.append(dict(split=split,scene=scene,start=i,gap=gap,rotation_deg=angle,
                                     translation_l2=np.linalg.norm(local[:3,3]-shared[:3,3])))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("out",type=Path)
    args = p.parse_args()
    args.out.write_text(json.dumps(audit(args.root),indent=2)+"\n")
