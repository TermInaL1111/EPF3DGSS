"""Evaluate the trained event-DVLT/GS architecture on held-out real DAVIS flights."""
import argparse
import json
from pathlib import Path
import torch
from event_dvlt import EventDVLT,load_scene_weights,render,voxelize
from event_field import EventField
from image_metrics import PerceptualMetrics, gray_metrics
from dvlt.common.rays import rays_to_pose
from mvsec_transfer import calibration,flight_windows
from torch.nn import functional as F


@torch.inference_mode()
def main(args):
    model=EventDVLT(args.assets).cuda().eval()
    delta=torch.load(args.weights,weights_only=True)
    if args.geometry_only:
        use_field = bool(delta['config']['f3'])
        delta = delta['model']
        expected = {k for k,p in model.named_parameters() if p.requires_grad
                    and not k.startswith(('gs_head.', 'geometry_features.', 'field_features.'))
                    and k != 'log_metric_scale' and (use_field or not k.startswith('event_adapter.'))}
        if set(delta) != expected:
            raise ValueError('Checkpoint parameter set does not match the geometry evaluation mode')
        model.load_state_dict(delta,strict=False)
    else:
        render_steps,use_field = load_scene_weights(model,delta)
    maps=calibration(args.calibration)
    extract=EventField(args.f3_repository,args.f3_checkpoint,4) if use_field else None
    perceptual = PerceptualMetrics("cuda") if args.lpips and not args.geometry_only else None
    rows=[]
    for flight in ("indoor_flying1","indoor_flying2","indoor_flying3"):
        for start,events,(source,target),query,right,raw in flight_windows(args.root,flight,args.count,4,extract,*maps,raw_packets=True):
            vox=torch.stack([voxelize(packet) for packet in raw])[None].cuda()
            field=events[:,4:][None] if use_field else vox.new_zeros((1,2,32,96,128))
            row=dict(flight=flight,start=start,
                     source_baseline_m=float(right[:3,3].norm()),
                     query_baseline_m=float(query[:3,3].norm()))
            baseline_cos = ((right[:3,:3].trace()-1)/2).clamp(-1,1)
            row['identity_rotation_deg'] = float(baseline_cos.acos().rad2deg())
            if not args.geometry_only:
                copy_psnr,copy_ssim = gray_metrics(source[None],target[None])
                row['source_copy_psnr'] = float(copy_psnr[0])
                row['source_copy_ssim'] = float(copy_ssim[0])
            for key,v,f in [("matched",vox,field),("zero",torch.zeros_like(vox),torch.zeros_like(field))]:
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    if args.geometry_only:
                        tokens, pos, _ = model.geometry(v,f,steps=16)
                        raw = model.backbone._decode(tokens,168,224,1,2,pos)
                    else:
                        scene=model(v,f,steps=render_steps)
                if args.geometry_only:
                    rays = raw['rays'].float()
                    poses, _ = rays_to_pose(rays,torch.ones_like(rays[...,0]),168,224,14)
                    relative = torch.linalg.solve(poses[0,0],poses[0,1])
                    rot = relative[:3,:3].T @ right[:3,:3]
                    row[key] = dict(rotation_deg=float(((rot.trace()-1)/2).clamp(-1,1).acos().rad2deg()))
                    continue
                image=F.avg_pool2d(render(scene,query[None]),5)[0]
                rot=scene["rotation"][0].T @ right[:3,:3]
                psnr,ssim = gray_metrics(image[None],target[None])
                row[key]=dict(psnr=float(psnr[0]),ssim=float(ssim[0]),
                              rotation_deg=float(((rot.trace()-1)/2).clamp(-1,1).acos().rad2deg()))
                row[key]["ssim_evggs"] = float(gray_metrics(image[None],target[None],padded=True)[1][0])
                if perceptual:
                    row[key].update(perceptual(image[None],target[None]))
            rows.append(row)
    args.out.write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","calibration","assets","weights","f3_repository","f3_checkpoint","out"):
        p.add_argument(name,type=Path)
    p.add_argument("--lpips",action="store_true")
    p.add_argument("--count",type=int,default=24,help="Windows per flight")
    p.add_argument("--geometry-only",action="store_true",help="Evaluate pose from a DSEC geometry checkpoint")
    main(p.parse_args())
