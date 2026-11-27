"""Evaluate event DVLT on nonoverlapping temporal MVSEC clips with depth and pose."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from dvlt.common.geometry import umeyama_alignment
from dvlt.common.rays import rays_to_pose
from dvlt.metric.pose import compute_pose_metrics
from evaluate_dsec_geometry import align_pointmap, lift
from event_dvlt import EventDVLT, voxelize
from event_field import EventField
from mvsec_transfer import calibration, lower_bound_after
from real_geometry import depth_lookup


@torch.inference_mode()
def evaluate(args):
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = EventDVLT(args.assets).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=False)
    encoder = (EventField(args.f3_repository, args.assets/'f3.pth', 4)
               if checkpoint['config']['f3'] else None)
    mapping, _ = calibration(args.calibration)
    xy, z_factor = depth_lookup(args.calibration, 168, 224)
    k = torch.tensor([[888.8889*224/640, 0, 111.5], [0, 888.8889*168/480, 83.5],
                      [0, 0, 1]], device='cuda')
    results = []
    for name in args.flights:
        with h5py.File(args.root/f'{name}_data.hdf5') as data, h5py.File(args.root/f'{name}_gt.hdf5') as truth:
            camera, gt = data['davis/left'], truth['davis/left']
            times = camera['image_raw_ts'][:]
            pose_times, pose = gt['pose_ts'][:], gt['pose'][:]
            depth_times = gt['depth_image_rect_ts'][:]
            interpolate = Slerp(pose_times, Rotation.from_matrix(pose[:, :3, :3]))
            eligible = np.flatnonzero((times >= pose_times[0]) & (times <= pose_times[-1]) &
                                      (times >= depth_times[0]) & (times <= depth_times[-1]))
            eligible = eligible[eligible > 0]
            starts = eligible[:len(eligible)//18*18:18]
            if args.count is not None and len(starts) > args.count:
                starts = starts[np.linspace(0, len(starts)-1, args.count, dtype=int)]
            for start in starts:
                indices = np.arange(start, start+18)
                at = times[indices]
                gt_poses = np.repeat(np.eye(4)[None], 18, axis=0)
                gt_poses[:, :3, :3] = interpolate(at).as_matrix()
                gt_poses[:, :3, 3] = np.stack([np.interp(at, pose_times, pose[:, j, 3]) for j in range(3)], axis=-1)
                gt_poses = torch.as_tensor(np.linalg.solve(gt_poses[0], gt_poses), device='cuda', dtype=torch.float32)
                packets, depths = [], []
                events = camera['events']
                for i in indices:
                    block = events[lower_bound_after(events, times[i]-.05):lower_bound_after(events, times[i])]
                    coords = mapping[block[:,1].astype(int), block[:,0].astype(int)]
                    packets.append(np.column_stack([coords, block[:,2:]]).T)
                    d = int(np.argmin(np.abs(depth_times-times[i])))
                    depth = gt['depth_image_rect'][d][xy[...,1],xy[...,0]] / z_factor
                    if abs(depth_times[d]-times[i]) > .025:
                        depth.fill(np.nan)
                    depths.append(depth)
                voxels = torch.stack([voxelize(packet) for packet in packets])[None].cuda()
                field = (torch.stack([encoder(packet) for packet in packets])[None] if encoder
                         else voxels.new_zeros((1,18,32,96,128)))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    tokens, pos, _ = model.geometry(voxels, field, steps=16)
                    raw = model.backbone._decode(tokens,168,224,1,18,pos)
                rays = raw['rays'].float()
                poses, intrinsics = rays_to_pose(rays,torch.ones_like(rays[...,0]),168,224,14)
                poses = torch.linalg.solve(poses[0,0], poses[0])
                estimate = raw['depth'][0,...,0].float()
                reference = torch.as_tensor(np.stack(depths),device='cuda',dtype=torch.float32)
                valid = (reference >= .1) & (reference <= 50) & torch.isfinite(estimate) & (estimate > 0)
                p, t = estimate[valid], reference[valid]
                if not len(p):
                    continue
                scale = (t/p).median()
                aligned_depth = p*scale
                point_sets = []
                for values, cameras, ks in ((estimate,poses,intrinsics[0]), (reference,gt_poses,k.expand(18,-1,-1))):
                    point_sets.append(torch.cat([lift(z,ki)[mask] @ cam[:3,:3].T + cam[:3,3]
                        for z,cam,ki,mask in zip(values,cameras,ks,valid)]))
                xyz, truth_xyz = point_sets
                point_error = (align_pointmap(xyz,truth_xyz)-truth_xyz).norm(dim=-1)/truth_xyz.norm(dim=-1).clamp_min(1e-8)
                origins, true_origins = poses[:,:3,3], gt_poses[:,:3,3]
                rot, trans, distance_scale = umeyama_alignment(origins.T,true_origins.T)
                offset = distance_scale*(origins @ rot.T)+trans-true_origins
                pose_metrics = compute_pose_metrics(poses.double(),gt_poses.double(),thresholds=(3,20,30))
                results.append(dict(flight=name,start=int(start),frames=indices.tolist(),
                    absrel=float(((aligned_depth-t).abs()/t).mean()),
                    delta1=float((torch.maximum(aligned_depth/t,t/aligned_depth)<1.25).float().mean()),
                    ate_m=float(offset.square().sum(-1).mean().sqrt()),
                    **{f'pose_auc{a}':float(pose_metrics[f'Auc_{a}']) for a in (3,20,30)},
                    point_absrel=float(point_error.mean()),
                    point_inlier25=float((point_error<.25).float().mean()),
                    point_inlier3=float((point_error<.03).float().mean())))
            print(name,len(starts),'windows attempted',flush=True)
    if not results:
        raise ValueError('No windows with synchronized pose and depth')
    args.out.write_text(json.dumps(dict(step=checkpoint['step'],clip_views=18,
        depth_scale='median ratio per window', event_window_ms=50, rows=results),
        indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root','calibration','assets','f3_repository','checkpoint','out'):
        parser.add_argument(key,type=Path)
    parser.add_argument('--flights',nargs='+',default=['indoor_flying1','indoor_flying2','indoor_flying3'])
    parser.add_argument('--count',type=int,help='Maximum number of clips per flight')
    evaluate(parser.parse_args())
