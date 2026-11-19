"""Re-evaluate frozen checkpoints with per-frame camera records and isolate identity sentinels."""
import argparse
import json
from pathlib import Path
import torch
from model import EventGaussians
from pilot import dataset,foreground
from ev3ds_pose import camera, OPENGL_TO_OPENCV
import numpy as np


@torch.inference_mode()
def sensitivity(args):
    fields = torch.load(args.event_fields,weights_only=True)
    report = json.loads((args.run/"scores.json").read_text())
    names = report["validation"]
    scale = report.get("scale",1)
    events,gray,target,right,_ = dataset(args.root,names,None,fields=fields,scale=scale)
    masks = foreground(args.root,names,scale=scale)
    net = EventGaussians(36,scale,report.get("fusion","none")).cuda().eval()
    net.load_state_dict(torch.load(args.run/"student_0.1.pt",weights_only=True))
    rows=[]
    for n in range(len(events)):
        scene,start = names[n//8],4+7*(n%8)
        cloud = net(events[n:n+1])
        pred = net.render(cloud,target[n:n+1])[0]
        difference = cloud["rotation"][0].T @ right[n,:3,:3]
        angle = ((difference.trace()-1)/2).clamp(-1,1).acos().rad2deg()
        err = (pred-gray[n]).square()
        rows.append(dict(scene=scene,start=start,
                         identity_record=any(np.allclose(camera(args.root,scene,j),OPENGL_TO_OPENCV)
                                             for j in (start,start+2,start+4)),
                         rotation_error_deg=float(angle),psnr=float(-10*err.mean().log10()),
                         fg_psnr=float(-10*((err*masks[n]).sum()/masks[n].sum()).log10())))
    args.out.write_text(json.dumps(rows,indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root","run","event_fields","out"):
        p.add_argument(name,type=Path)
    sensitivity(p.parse_args())
