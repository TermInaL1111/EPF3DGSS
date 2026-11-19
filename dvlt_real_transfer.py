"""Evaluate the trained event-DVLT/GS architecture on held-out real DAVIS flights."""
import argparse
import json
from pathlib import Path
import torch
from event_dvlt import EventDVLT,render,voxelize
from event_field import EventField
from mvsec_transfer import calibration,flight_windows
from torch.nn import functional as F


@torch.inference_mode()
def main(args):
    model=EventDVLT(args.assets).cuda().eval()
    delta=torch.load(args.weights,weights_only=True)
    assert set(delta)=={k for k,p in model.named_parameters() if p.requires_grad}
    model.load_state_dict(delta,strict=False)
    maps=calibration(args.calibration)
    extract=EventField(args.f3_repository,args.f3_checkpoint,4)
    rows=[]
    for flight in ("indoor_flying1","indoor_flying2","indoor_flying3"):
        for start,events,(_,target),query,right,raw in flight_windows(args.root,flight,24,4,extract,*maps,raw_packets=True):
            vox=torch.stack([voxelize(packet) for packet in raw])[None].cuda()
            field=events[:,4:][None]
            row=dict(flight=flight,start=start)
            for key,v,f in [("matched",vox,field),("zero",torch.zeros_like(vox),torch.zeros_like(field))]:
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    scene=model(v,f)
                image=F.avg_pool2d(render(scene,query[None]),5)[0]
                rot=scene["rotation"][0].T @ right[:3,:3]
                row[key]=dict(psnr=float(-10*(image-target).square().mean().log10()),
                              rotation_deg=float(((rot.trace()-1)/2).clamp(-1,1).acos().rad2deg()))
            rows.append(row)
    args.out.write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","calibration","assets","weights","f3_repository","f3_checkpoint","out"):
        p.add_argument(name,type=Path)
    main(p.parse_args())
