"""Evaluation-only replacement of predicted Gaussian geometry and source intensities."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from event_dvlt import EventDVLT, render
from event_field import source_windows
from train_event_dvlt import get_inputs, labels


def reference_scene(root, scene, i, relative, cloud):
    depth, gray = [], []
    for j in (i,i+4):
        stem = root/"Scenes"/scene/f"{j:04}"
        with np.load(stem.with_suffix(".npz")) as archive:
            depth.append(archive["depth_map"])
        with Image.open(stem.with_suffix(".png")) as image:
            gray.append(np.array(image.convert("L")))
    z = torch.tensor(np.stack(depth),device="cuda",dtype=torch.float32)
    valid = z>0
    # Retain a far plane for the synthetic backdrop; its depth is unspecified in EV3DS.
    z = torch.where(valid,z,torch.full_like(z,10.))
    k = np.loadtxt(root/"Intrinsics"/"0000.txt")
    y,x = torch.meshgrid(torch.arange(480,device="cuda"),torch.arange(640,device="cuda"),indexing="ij")
    rays = torch.stack([(x-k[0,2])/k[0,0],(y-k[1,2])/k[1,1],torch.ones_like(x)],-1)
    def calibrated_centers(depth):
        local = (depth[...,None]*rays).reshape(2,-1,3)
        second = local[1] @ relative[:3,:3].T + relative[:3,3]
        return torch.cat([local[0],second])[None].contiguous()

    centers = calibrated_centers(z)
    cameras = dict(centers=calibrated_centers(cloud["source_depth"][0]))
    scales = (z.flatten()/float(k[0,0]))[None,:,None].expand_as(centers).contiguous()
    quats = torch.zeros_like(cloud["quats"])
    quats[...,0] = 1.
    geometry = dict(centers=centers,scales=scales,quats=quats,opacity=torch.ones_like(cloud["opacity"]))
    appearance = torch.tensor(np.stack(gray),device="cuda").float().reshape(1,-1,1)/255
    return cameras,geometry,appearance


@torch.inference_mode()
def main(args):
    model = EventDVLT(args.assets).cuda().eval()
    delta = torch.load(args.student,weights_only=True)
    assert set(delta) == {key for key,p in model.named_parameters() if p.requires_grad}
    model.load_state_dict(delta,strict=False)
    fields = torch.load(args.fields,weights_only=True,mmap=True)
    rows = []
    for scene,i in source_windows(args.root,"val",args.full_split):
        target,mask,poses = labels(args.root,scene,i)
        v,f = get_inputs(args.root,fields,scene,i)
        torch.manual_seed(10)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            cloud = model(v,f)
        cams,geom,gray = reference_scene(args.root,scene,i,poses[1],cloud)
        row = dict(scene=scene,start=i)
        for name,geometry,appearance in (("student",{},cloud["gray"]),("source_gray",{},gray),
                                         ("cameras",cams,cloud["gray"]),("cameras_gray",cams,gray),
                                         ("reference_geometry",geom,cloud["gray"]),("both",geom,gray)):
            pred = render({**cloud,**geometry,"gray":appearance},poses[:1])
            err = (pred-target).square()
            a,b = pred[mask],target[mask]
            correlation = F.cosine_similarity(a-a.mean(),b-b.mean(),dim=0)
            row[name] = dict(foreground=float(-10*err[mask].mean().log10()),correlation=float(correlation))
        rows.append(row)
    args.out.write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("root","fields","assets","student","out"):
        parser.add_argument(key,type=Path)
    parser.add_argument("--full-split",action="store_true")
    main(parser.parse_args())
