"""Measure local source depths where the Gaussian grid samples a foreground surface."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model import EventGaussians
from pilot import dataset


@torch.inference_mode()
def main(args):
    report = json.loads((args.run/"scores.json").read_text())
    names = report["validation"]
    fields = torch.load(args.event_fields,weights_only=True) if args.event_fields else None
    scale = report.get("scale",1)
    cell = 80//scale
    mid = cell//2
    events = dataset(args.root,names,None,fields=fields,scale=scale)[0]
    gt = []
    for scene in names:
        for i in range(4,60,7):
            pair = []
            for j in (i,i+4):
                with np.load(args.root/"Scenes"/scene/f"{j:04}.npz") as archive:
                    # Require all four neighbors around the ray center to see a surface.
                    d = archive["depth_map"].reshape(6*scale,cell,8*scale,cell)[:,mid-1:mid+1,:,mid-1:mid+1]
                    pair.append(np.where((d>0).all((1,3)),d.mean((1,3)),0))
            gt.append(pair)
    gt = torch.tensor(np.array(gt),device="cuda")
    net = EventGaussians(events.shape[2],scale=scale,fusion_mode=report.get("fusion","none")).cuda().eval()
    net.load_state_dict(torch.load(args.run/"student_0.1.pt",weights_only=True))
    actual = net(events)["source_depth"]
    empty = net(torch.zeros_like(events))["source_depth"]
    rows = []
    for n,reference in enumerate(gt):
        mask = reference>0
        if not mask.any():
            continue
        truth = reference[mask]
        row = dict(scene=names[n//8],start=4+7*(n%8),valid_rays=int(mask.sum()))
        for name,d in [("matched",actual[n]),("zero",empty[n]),("plane",torch.ones_like(reference))]:
            depth = d[mask]
            scale = (truth/depth).median()
            row[name+"_aligned_abs_rel"] = float(((scale*depth-truth).abs()/truth).mean())
        rows.append(row)
    (args.run/"student_depth.json").write_text(json.dumps(rows,indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("run",type=Path)
    parser.add_argument("--event-fields",type=Path)
    main(parser.parse_args())
