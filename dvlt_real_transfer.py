"""Evaluate the trained event-DVLT/GS architecture on held-out real DAVIS flights."""
import argparse
import json
from pathlib import Path
import torch
from event_dvlt import EventDVLT,load_scene_weights,render,voxelize
from event_field import EventField
from event_depth_boundary import (ARMS, EventDepthBoundary, evidence, event_bins,
                                  load_temporal_boundary, restore_depth_scale, temporal_corrector)
from image_metrics import PerceptualMetrics, gray_metrics
from dvlt.common.rays import rays_to_pose
from mvsec_transfer import calibration,flight_windows
from torch.nn import functional as F


RENDER_KEYS = ('centers', 'quats', 'scales', 'opacity', 'gray')


def image_scores(scene, query, target, perceptual):
    image = F.avg_pool2d(render(scene, query[None]), 5)
    target = target[None]
    psnr, ssim = gray_metrics(image, target)
    scores = dict(psnr=float(psnr[0]), ssim=float(ssim[0]),
                  ssim_evggs=float(gray_metrics(image, target, padded=True)[1][0]))
    if perceptual:
        scores.update(perceptual(image, target))
    return scores


def cap_scene(scene, budget):
    count = scene['opacity'].shape[1]
    if not 0 < budget <= count:
        raise ValueError('Gaussian budget must be positive and at most the scene count')
    if budget == count:
        return scene
    keep = torch.argsort(scene['opacity'][0], descending=True, stable=True)[:budget]
    keep = keep.sort().values
    return {key: scene[key][:, keep] for key in RENDER_KEYS}


@torch.inference_mode()
def main(args):
    if args.preserve_boundary_scale and not args.boundary:
        raise ValueError('Scale preservation requires an external boundary checkpoint')
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
    training_boundary = load_temporal_boundary(delta['boundary'], 'cuda') if 'boundary' in delta else None
    if training_boundary and args.boundary:
        raise ValueError('Evaluate the embedded matched-budget model without an external boundary head')
    boundary_heads = {}
    if args.boundary:
        if args.geometry_only:
            raise ValueError('Boundary refinement is evaluated with Gaussian rendering')
        boundary_state = torch.load(args.boundary, weights_only=True, map_location='cpu')
        for arm in ARMS:
            head = EventDepthBoundary().cuda().eval()
            head.load_state_dict(boundary_state['model'][arm])
            boundary_heads[arm] = head
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
            variants = [("matched",vox,field,None,False),
                        ("zero",torch.zeros_like(vox),torch.zeros_like(field),None,False)]
            if training_boundary:
                counts = torch.stack([event_bins(packet, (480,640)) for packet in raw]).cuda()
                variants = [(key, v, f, temporal_corrector(training_boundary, counts, empty=key=='zero',
                             preserve_scale=delta.get('preserve_boundary_scale',False)),
                             delta['event_leaves']) for key, v, f, _, _ in variants]
            if boundary_heads:
                counts = torch.stack([event_bins(packet, (480,640)) for packet in raw]).cuda()
                boundary_inputs = evidence(counts)
                for arm, head in boundary_heads.items():
                    def refine(tokens, depth, confidence, head=head, features=boundary_inputs[arm], arm=arm):
                        log_depth, _ = head(tokens.float(), depth, confidence, features)
                        if args.preserve_boundary_scale:
                            corrected, offset = restore_depth_scale(log_depth,depth)
                            row.setdefault('boundary_log_scale_offsets',{})[arm] = offset.tolist()
                            return corrected
                        return log_depth.exp()
                    variants.append((f'boundary_{arm}',vox,field,refine,False))
                if model.head_type == 'octree':
                    def refine_and_rank(tokens, depth, confidence):
                        log_depth, edges = boundary_heads['temporal'](
                            tokens.float(), depth, confidence, boundary_inputs['temporal'])
                        if args.preserve_boundary_scale:
                            return restore_depth_scale(log_depth,depth)[0], edges
                        return log_depth.exp(), edges
                    variants.append(('boundary_temporal_event_leaves',vox,field,refine_and_rank,True))
            budget_scenes = {}
            for key,v,f,refinement,rank_leaves in variants:
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    if args.geometry_only:
                        tokens, pos, _ = model.geometry(v,f,steps=16)
                        raw = model.backbone._decode(tokens,168,224,1,2,pos)
                    else:
                        scene=model(v,f,steps=render_steps,depth_refinement=refinement,
                                    event_leaf_ranking=rank_leaves, matched_leaf_budget=bool(training_boundary))
                if args.geometry_only:
                    rays = raw['rays'].float()
                    poses, _ = rays_to_pose(rays,torch.ones_like(rays[...,0]),168,224,14)
                    relative = torch.linalg.solve(poses[0,0],poses[0,1])
                    rot = relative[:3,:3].T @ right[:3,:3]
                    row[key] = dict(rotation_deg=float(((rot.trace()-1)/2).clamp(-1,1).acos().rad2deg()))
                    continue
                rot = scene["rotation"][0].T @ right[:3,:3]
                row[key] = image_scores(scene, query, target, perceptual)
                row[key]['rotation_deg'] = float(((rot.trace()-1)/2).clamp(-1,1).acos().rad2deg())
                if 'gaussian_count' in scene:
                    row[key]['gaussian_count'] = int(scene['gaussian_count'][0])
                if boundary_heads and key != 'zero':
                    budget_scenes[key] = {attr: scene[attr] for attr in RENDER_KEYS}
            if budget_scenes:
                budget = min(scene['opacity'].shape[1] for scene in budget_scenes.values())
                row['matched_budget'] = {
                    key: dict(gaussian_count=budget,
                              **image_scores(cap_scene(scene, budget), query, target, perceptual))
                    for key, scene in budget_scenes.items()
                }
            rows.append(row)
    args.out.write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","calibration","assets","weights","f3_repository","f3_checkpoint","out"):
        p.add_argument(name,type=Path)
    p.add_argument("--lpips",action="store_true")
    p.add_argument("--count",type=int,default=24,help="Windows per flight")
    p.add_argument("--boundary",type=Path,help="Evaluate frozen depth refinement arms on the same matched windows")
    p.add_argument("--preserve-boundary-scale",action="store_true",
                   help="Center each boundary correction in the original DVLT log-depth gauge")
    p.add_argument("--geometry-only",action="store_true",help="Evaluate pose from a DSEC geometry checkpoint")
    main(p.parse_args())
