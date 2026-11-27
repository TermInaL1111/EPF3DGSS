"""Check source depth, relative pose, and query-camera dependence on each held-out flight."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from event_field import EventField
from model import EventGaussians
from mvsec_transfer import calibration, flight_windows


def depth_lookup(calibration_path, height=24, width=32):
    cam = yaml.safe_load(calibration_path.read_text())["cam0"]
    rot = np.asarray(cam["rectification_matrix"])
    proj = np.asarray(cam["projection_matrix"])[:,:3]
    y,x = np.indices((height,width))
    rays = np.stack([(x-(width-1)/2)/(888.8889*width/640),
                     (y-(height-1)/2)/(888.8889*height/480),np.ones_like(x)],-1)
    rectified = rays @ rot.T
    coords = rectified @ proj.T
    xy = np.rint(coords[...,:2]/coords[...,2:]).astype(int)
    # The narrow virtual training camera is fully inside the DAVIS rectangle.
    w,h = cam["resolution"]
    assert (xy.min(axis=(0,1))>=0).all() and (xy.max(axis=(0,1))<[w,h]).all()
    return xy,rectified[...,2]


def depth_pair(gt, image_ts, depth_ts, start, xy, z_factor):
    samples = []
    offsets = []
    for frame in (start,start+4):
        idx = int(np.argmin(np.abs(depth_ts-image_ts[frame])))
        dt = abs(depth_ts[idx]-image_ts[frame])
        offsets.append(float(dt))
        d = gt["depth_image_rect"][idx][xy[...,1],xy[...,0]]/z_factor
        if dt>.025:
            d.fill(np.nan)
        samples.append(d)
    return torch.tensor(np.stack(samples),dtype=torch.float32,device="cuda"),offsets


@torch.inference_mode()
def main(args):
    report = json.loads((args.run/"scores.json").read_text())
    names = {flight for flight,_ in report["validation"]}
    assert len(names)==1
    flight = names.pop()
    assert flight not in {key[0] for key in report["train"]}
    field = EventField(args.f3_repository,args.f3_checkpoint,4)
    maps = calibration(args.calibration)
    xy,factor = depth_lookup(args.calibration)
    nets = {}
    for name in report["arms"]:
        net = EventGaussians(36,4).cuda().eval()
        net.load_state_dict(torch.load(args.run/f"{name}.pt",weights_only=True))
        nets[name] = net
    rows = []
    with h5py.File(args.root/f"{flight}_data.hdf5") as df, h5py.File(args.root/f"{flight}_gt.hdf5") as gf:
        ts = df["davis/left/image_raw_ts"][:]
        gt = gf["davis/left"]
        depth_ts = gt["depth_image_rect_ts"][:]
        for n,(start,ev,(left,gray),query,right) in enumerate(flight_windows(args.root,flight,32,4,field,*maps)):
            assert report["validation"][n]==[flight,start]
            ref,dt = depth_pair(gt,ts,depth_ts,start,xy,factor)
            valid = torch.isfinite(ref)&(ref>.1)&(ref<80)
            z = ref[valid]
            row = dict(start=start,valid_rays=int(valid.sum()),depth_dt_s=dt,arms={})
            row["source_copy_psnr"] = float(-10*(left-gray).square().mean().log10())
            row["identity_rotation_deg"] = float(((right[:3,:3].trace()-1)/2).clamp(-1,1).acos().rad2deg())
            if z.numel():
                row["plane_abs_rel"] = float(((z.median()-z).abs()/z).mean())
            for key,net in nets.items():
                scene = net(ev[None])
                fixed = net.render(scene,torch.eye(4,device="cuda")[None])
                normal = net.render(scene,query[None])
                arm = dict(source_camera_psnr=float(-10*(fixed-gray).square().mean().log10()),
                           query_psnr=float(-10*(normal-gray).square().mean().log10()))
                if z.numel():
                    pred = scene["source_depth"][0][valid]
                    arm["metric_abs_rel"] = float(((pred-z).abs()/z).mean())
                    scale = (z/pred).median()
                    arm["aligned_abs_rel"] = float(((scale*pred-z).abs()/z).mean())
                row["arms"][key] = arm
            rows.append(row)
    args.out.write_text(json.dumps(dict(flight=flight,rows=rows),indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root","calibration","f3_repository","f3_checkpoint","run","out"):
        p.add_argument(name,type=Path)
    main(p.parse_args())
