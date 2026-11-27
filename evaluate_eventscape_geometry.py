"""Compare Town05 geometry on the same RGB-D camera references."""

import argparse
from contextlib import closing
import json
from pathlib import Path
from statistics import fmean

import cv2
import numpy as np
import torch

from dsec_data import resize_intrinsics
from dvlt.common.geometry import umeyama_alignment
from dvlt.common.rays import rays_to_pose
from dvlt.metric.pose import compute_pose_metrics
from evaluate_dsec_geometry import align_pointmap, lift
from evaluate_eventscape_depth import depth_scores
from event_dvlt import EventDVLT, load_rgb, voxelize
from eventscape_data import EventScapeArchive, EventScapeSequence
from eventscape_gaussian_data import NATIVE_K


def scores(raw, depth, reference_pose, reference_k, valid):
    estimate = raw['depth'][0,...,0].float()
    rays = raw['rays'].float()
    poses, intrinsics = rays_to_pose(rays,torch.ones_like(rays[...,0]),168,224,14)
    poses = torch.linalg.solve(poses[0,0],poses[0])
    fitted = [(lift(z,k)[mask] @ cam[:3,:3].T + cam[:3,3])
              for z,k,cam,mask in zip(estimate,intrinsics[0],poses,valid,strict=True)]
    reference = [(lift(z,reference_k)[mask] @ cam[:3,:3].T + cam[:3,3])
                 for z,cam,mask in zip(depth,reference_pose,valid,strict=True)]
    target = torch.cat(reference)
    point_error = (align_pointmap(torch.cat(fitted),target)-target).norm(dim=-1)
    point_error = point_error/target.norm(dim=-1).clamp_min(1e-8)
    xyz, target_xyz = poses[:,:3,3].double(), reference_pose[:,:3,3].double()
    rotation, translation, scale = umeyama_alignment(xyz.T,target_xyz.T)
    offset = scale*(xyz @ rotation.T)+translation-target_xyz
    angles = compute_pose_metrics(poses.double(),reference_pose.double(),thresholds=(3,20,30))
    return dict(**depth_scores(estimate[valid],depth[valid]),
                ate_m=float(offset.square().sum(-1).mean().sqrt()),
                point_absrel=float(point_error.mean()),
                point_inlier25=float((point_error<.25).float().mean()),
                **{f'pose_auc{a}':float(angles[f'Auc_{a}']) for a in (3,20,30)})


@torch.inference_mode()
def evaluate(args, archive, manifest):
    teacher = load_rgb(args.assets)
    teacher.inference_steps = 16
    student = EventDVLT(args.assets).cuda().eval()
    states = {arm: torch.load(path,map_location='cpu',weights_only=True) for arm,path in
              (('direct',args.direct),('realm',args.realm))}
    expected = {name for name,p in student.named_parameters() if p.requires_grad
                and not name.startswith(('gs_head.','geometry_features.','field_features.','event_adapter.'))
                and name != 'log_metric_scale'}
    for state in states.values():
        if state['config']['f3'] or set(state['model']) != expected or state['step'] != 20000:
            raise ValueError('Expected matched 20,000-step EventScape geometry checkpoints without F³')
        if set(archive.sequences).intersection(state['config']['train']):
            raise ValueError('EventScape training and test sequences overlap')
    k = torch.as_tensor(resize_intrinsics(NATIVE_K,(512,256),(224,168)),
                        device='cuda',dtype=torch.float32)
    rows = []
    for name,cameras in manifest['sequences'].items():
        source = EventScapeSequence(archive,name)
        for start,clips in cameras.items():
            indices = range(int(start),int(start)+18)
            poses = torch.as_tensor(np.stack([np.eye(4)]+[np.asarray(item['c2w']) for item in clips]),
                                    device='cuda',dtype=torch.float32)
            if poses.shape != (18,4,4):
                raise ValueError(f'Expected 18 accepted cameras in {name}/{start}')
            packets,frames = zip(*(source.frame(i) for i in indices),strict=True)
            reference = [cv2.resize(source.depth(i),(224,168),interpolation=cv2.INTER_NEAREST_EXACT)
                         for i in indices]
            reference = torch.as_tensor(np.stack(reference),device='cuda')
            rgb = torch.as_tensor(np.stack(frames),device='cuda')[None].permute(0,1,4,2,3).float()/255
            voxels = torch.stack([voxelize(packet) for packet in packets])[None].cuda()
            field = voxels.new_zeros((1,18,32,96,128))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                raw = {'teacher':teacher.forward_inference(rgb)}
                for arm,state in states.items():
                    student.load_state_dict(state['model'],strict=False)
                    tokens,pos,_ = student.geometry(voxels,field,steps=16)
                    raw[arm] = student.backbone._decode(tokens,168,224,1,18,pos)
            valid = (reference > 0) & (reference <= 200)
            for output in raw.values():
                estimate = output['depth'][0,...,0]
                valid &= torch.isfinite(estimate) & (estimate > 0)
            if valid.sum()<10:
                raise ValueError(f'Insufficient common depth support in {name}/{start}')
            row = dict(sequence=name,start=int(start),valid_pixels=int(valid.sum()))
            row.update({arm:scores(output,reference,poses,k,valid) for arm,output in raw.items()})
            rows.append(row)
            if args.limit and len(rows)==args.limit:
                return rows
        print(name,len(rows),'windows scored',flush=True)
    return rows


def summarize(rows):
    direction = dict(absrel=-1,delta1=1,ate_m=-1,point_absrel=-1,point_inlier25=1,
                     pose_auc3=1,pose_auc20=1,pose_auc30=1)
    summary = {arm:{metric:fmean(row[arm][metric] for row in rows) for metric in direction}
               for arm in ('teacher','direct','realm')}
    relative = {}
    for metric,sign in direction.items():
        baseline,estimate = (summary[arm][metric] for arm in ('direct','realm'))
        relative[metric] = 100*sign*(estimate-baseline)/abs(baseline) if baseline else None
    return dict(summary=summary,realm_vs_direct_improvement_pct=relative)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive','assets','cameras','direct','realm','out'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--limit',type=int)
    args = parser.parse_args()
    manifest = json.loads(args.cameras.read_text())
    with closing(EventScapeArchive(args.archive)) as archive:
        rows = evaluate(args,archive,manifest)
    if not rows or (args.limit is None and len(rows) != manifest['windows']):
        raise ValueError('Expected all independently validated reference windows')
    report = dict(clip_views=18,recurrent_steps=16,max_depth_m=200,windows=len(rows),
                  candidate_windows=manifest['attempted'],camera_reference=manifest['reference'],
                  pointmap_grid=[224,168],**summarize(rows),rows=rows)
    temporary = args.out.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.out)
