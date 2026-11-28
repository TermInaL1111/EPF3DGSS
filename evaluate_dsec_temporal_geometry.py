"""Score event DVLT on the same nonoverlapping 18-frame monocular DSEC clips."""

import argparse
from contextlib import closing
import json
from pathlib import Path
from statistics import fmean

import cv2
import numpy as np
import torch
from torch.nn.functional import interpolate

from dsec_data import DSECSequence, camera_matrix
from dvlt.common.geometry import umeyama_alignment
from dvlt.common.rays import rays_to_pose
from dvlt.metric.pose import compute_pose_metrics
from evaluate_dsec_geometry import align_pointmap, lift, native_intrinsics
from evaluate_eventscape_depth import depth_scores
from event_dvlt import EventDVLT, voxelize
from event_field import EventField


ARM_NAMES = ('eventscape_direct','eventscape_saag','eventscape_realm','dsec_no_f3','dsec_f3')
DEPTH_METRICS = dict(absrel=-1,delta1=1,unscaled_absrel=-1,unscaled_delta1=1)
GEOMETRY_METRICS = dict(point_absrel=-1,point_inlier25=1,ate_m=-1,
                        pose_auc3=1,pose_auc20=1,pose_auc30=1)


def load_states(args, model, sequences):
    states = {}
    for name,path in zip(ARM_NAMES,args.checkpoints,strict=True):
        state = torch.load(path,map_location='cpu',weights_only=True)
        use_field = name == 'dsec_f3'
        expected_steps = 22522 if name.startswith('dsec_') else 20000
        expected = {key for key,p in model.named_parameters() if p.requires_grad
                    and not key.startswith(('gs_head.','geometry_features.','field_features.'))
                    and key != 'log_metric_scale' and (use_field or not key.startswith('event_adapter.'))}
        if (bool(state['config']['f3']) != use_field or state['step'] != expected_steps
                or set(state['model']) != expected or sequences.intersection(state['config']['train'])):
            raise ValueError(f'Checkpoint mismatches the held-out evaluation for {name}')
        states[name] = state['model']
    return states


def reference_depth(sequence, index):
    path = sequence.root/f'disparity/event/{index:06}.png'
    disparity = cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if disparity is None or disparity.shape != (480,640):
        raise ValueError(f'Invalid DSEC event disparity {path}')
    q = np.asarray(sequence.calib['disparity_to_depth']['cams_03'])
    disparity = disparity.astype(np.float32)/256
    depth = cv2.reprojectImageTo3D(disparity,q)[...,2]
    return np.where(disparity>0,depth,0)


def project_points(depth, cameras, intrinsics, valid):
    return torch.cat([lift(z,k)[mask] @ c[:3,:3].T + c[:3,3]
                      for z,c,k,mask in zip(depth,cameras,intrinsics,valid,strict=True)])


def geometry_metrics(raw, estimate, valid, truth, truth_pose):
    rays = raw['rays'].float()
    c2w,intrinsics = rays_to_pose(rays,torch.ones_like(rays[...,0]),168,224,14)
    c2w = torch.linalg.solve(c2w[0,0],c2w[0])
    xyz = project_points(estimate,c2w,[native_intrinsics(k) for k in intrinsics[0]],valid)
    relative_error = (align_pointmap(xyz,truth)-truth).norm(dim=-1)/truth.norm(dim=-1).clamp_min(1e-8)
    estimated_position,true_position = c2w[:,:3,3].double(),truth_pose[:,:3,3].double()
    rot,shift,scale = umeyama_alignment(estimated_position.T,true_position.T)
    residual = scale*(estimated_position @ rot.T)+shift-true_position
    pose = compute_pose_metrics(c2w.double(),truth_pose.double(),thresholds=(3,20,30))
    return dict(point_absrel=float(relative_error.mean()),
                point_inlier25=float((relative_error<.25).float().mean()),
                ate_m=float(residual.square().sum(-1).mean().sqrt()),
                **{f'pose_auc{a}':float(pose[f'Auc_{a}']) for a in (3,20,30)})


@torch.inference_mode()
def evaluate(args, manifest, poses):
    model = EventDVLT(args.assets).cuda().eval()
    states = load_states(args,model,set(manifest['sequences']))
    encoder = EventField(args.f3_repository,args.assets/'f3.pth',4)
    rows = []
    for name,sequence_manifest in manifest['sequences'].items():
        with closing(DSECSequence(args.root,name,window_us=manifest['event_window_us'],
                                  monocular=True)) as sequence:
            k = torch.as_tensor(camera_matrix(sequence.calib['intrinsics']['camRect0']['camera_matrix']),
                                device='cuda',dtype=torch.float32).expand(18,-1,-1)
            for indices in sequence_manifest['windows']:
                start = indices[0]
                packets = [sequence.packet(0,i) for i in indices]
                voxel = torch.stack([voxelize(packet) for packet in packets])[None].cuda()
                field = torch.stack([encoder(packet) for packet in packets])[None].float()
                zero_field = torch.zeros_like(field)
                depth = torch.as_tensor(np.stack([reference_depth(sequence,i) for i in indices]),device='cuda')
                labelled = (depth>=.1) & (depth<=manifest['depth_cap_m'])
                valid = labelled.clone()
                results = {}
                for arm,state in states.items():
                    model.load_state_dict(state,strict=False)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        features = field if arm=='dsec_f3' else zero_field
                        tokens,pos,_ = model.geometry(voxel,features,steps=16)
                        raw = model.backbone._decode(tokens,168,224,1,18,pos)
                    estimate = interpolate(raw['depth'][0].float().permute(0,3,1,2),size=(480,640),
                                           mode='bilinear',align_corners=False)[:,0]
                    valid &= torch.isfinite(estimate) & (estimate>0)
                    results[arm] = (estimate,raw)
                if int(valid.sum()) < 16:
                    raise ValueError(f'Insufficient shared depth support at {name}/{start}')
                row = dict(sequence=name,start=start,frames=indices,labelled_pixels=int(labelled.sum()),
                           scored_pixels=int(valid.sum()))
                if pose := poses.get(name,{}).get(str(start)):
                    if pose['frames'] != indices:
                        raise ValueError(f'LiDAR frames differ at {name}/{start}')
                    truth_pose = torch.as_tensor(pose['event_c2w'],device='cuda',dtype=torch.float32)
                    truth = project_points(depth,truth_pose,k,valid)
                gt = depth[valid]
                for arm,(estimate,raw) in results.items():
                    values = estimate[valid]
                    row[arm] = dict(depth_scores(values,gt),
                                    unscaled_absrel=float(((values-gt).abs()/gt).mean()),
                                    unscaled_delta1=float((torch.maximum(values/gt,gt/values)<1.25).float().mean()))
                    if pose:
                        row[arm].update(geometry_metrics(raw,estimate,valid,truth,truth_pose))
                rows.append(row)
                if args.limit and len(rows)>=args.limit:
                    return rows
        print(name,len(rows),'windows scored',flush=True)
    return rows


def summarize(rows):
    if not rows:
        raise ValueError('No DSEC windows scored')
    lidar = [row for row in rows if 'ate_m' in row[ARM_NAMES[0]]]
    scores = {arm:{metric:fmean(row[arm][metric] for row in rows) for metric in DEPTH_METRICS}
              for arm in ARM_NAMES}
    if lidar:
        for arm in ARM_NAMES:
            scores[arm].update({metric:fmean(row[arm][metric] for row in lidar) for metric in GEOMETRY_METRICS})
    comparisons = {}
    for after,before in (('eventscape_saag','eventscape_direct'),('eventscape_realm','eventscape_direct'),
                         ('dsec_no_f3','eventscape_realm'),('dsec_f3','dsec_no_f3')):
        changes = {}
        for metric,sign in (DEPTH_METRICS | GEOMETRY_METRICS).items():
            if metric in scores[before]:
                base = scores[before][metric]
                changes[metric] = 100*sign*(scores[after][metric]-base)/abs(base) if base else None
        comparisons[f'{after}_vs_{before}'] = changes
    return dict(windows=len(rows),lidar_windows=len(lidar),scores=scores,improvement_pct=comparisons)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','assets','clips','f3_repository','out'):
        parser.add_argument(name,type=Path)
    parser.add_argument('checkpoints',type=Path,nargs=len(ARM_NAMES),metavar='CHECKPOINT',
                        help=', '.join(ARM_NAMES)+' in this order')
    parser.add_argument('--lidar-reference',type=Path)
    parser.add_argument('--limit',type=int)
    args = parser.parse_args()
    manifest = json.loads(args.clips.read_text())
    lidar = json.loads(args.lidar_reference.read_text()) if args.lidar_reference else {}
    if (manifest['clip_views'],manifest['event_window_us'],manifest['depth_cap_m']) != (18,50000,50):
        parser.error('Expected 18 views, 50 ms event windows and 50 m depth cap')
    rows = evaluate(args,manifest,lidar.get('sequences',{}))
    expected = {(name,tuple(window)) for name,source in manifest['sequences'].items() for window in source['windows']}
    if args.limit is None and (len(rows)!=len(expected) or {(r['sequence'],tuple(r['frames'])) for r in rows}!=expected):
        raise ValueError('DSEC report differs from the fixed evaluation windows')
    report = dict(**summarize(rows),clip_views=18,event_window_ms=50,max_depth_m=50,recurrent_steps=16,
                  depth_alignment='median ratio and unscaled, same valid pixels for all models',
                  pose_reference=lidar.get('reference'),rows=rows)
    temporary = args.out.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.out)
