"""Evaluate a trained event student across unseen source spacings and orderings."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from model import EventGaussians
from pilot import packet


@torch.inference_mode()
def evaluate(root, run):
    report = json.loads((run/"scores.json").read_text())
    net = EventGaussians().cuda().eval()
    net.load_state_dict(torch.load(run/"student_0.1.pt",weights_only=True))
    flip = np.diag([1,-1,-1,1])
    output = []
    for gap in (2,4,8,16):
        for reverse in (False,True):
            for scene in report["validation"]:
                packets, query, right, labels = [],[],[],[]
                for i in range(4,60,7):
                    left_id, right_id = (i+gap,i) if reverse else (i,i+gap)
                    mid = i+gap//2
                    poses = [np.loadtxt(root/"Poses"/f"r_{j+1:04}.txt") @ flip
                             for j in (left_id,mid,right_id)]
                    query.append(np.linalg.solve(poses[0],poses[1]))
                    right.append(np.linalg.solve(poses[0],poses[2]))
                    packets.append(torch.stack([packet(root/"Event"/scene/f"{j:04}.npy")
                                                for j in (left_id,right_id)]))
                    with Image.open(root/"Scenes"/scene/f"{mid:04}.png") as image:
                        gray = image.convert("L").resize((32,24),Image.Resampling.BOX)
                        labels.append(torch.tensor(np.array(gray))/255)
                target = torch.tensor(np.stack(query),device="cuda",dtype=torch.float32)
                truth = torch.tensor(np.stack(right),device="cuda",dtype=torch.float32)
                event = torch.stack(packets).cuda()
                label = torch.stack(labels).cuda()
                stats = dict(scene=scene,gap=gap,reverse=reverse)
                for key, inp in [("matched",event),("zero",torch.zeros_like(event))]:
                    cloud = net(inp)
                    residual = net.render(cloud,target)-label
                    stats[key+"_psnr"] = float((-10*residual.square().mean((1,2)).log10()).mean())
                    rotation = cloud["rotation"].transpose(1,2) @ truth[:,:3,:3]
                    cosine = (rotation.diagonal(dim1=1,dim2=2).sum(1)-1)/2
                    stats[key+"_rotation_deg"] = float(cosine.clamp(-1,1).acos().rad2deg().mean())
                    stats[key+"_translation_l2"] = float((cloud["translation"]-truth[:,:3,3]).norm(dim=1).mean())
                output.append(stats)
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("root","run","out"):
        p.add_argument(arg,type=Path)
    args = p.parse_args()
    args.out.write_text(json.dumps(evaluate(args.root,args.run),indent=2)+"\n")
