"""Evaluate event DVLT on nonoverlapping temporal MVSEC clips with depth and pose."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import h5py
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from dvlt.common.geometry import umeyama_alignment
from dvlt.common.rays import rays_to_pose
from dvlt.metric.pose import compute_pose_metrics
from evaluate_dsec_geometry import align_pointmap, lift
from event_dvlt import EventDVLT, load_rgb, voxelize
from event_field import EventField
from mvsec_transfer import calibration, lower_bound_after
from real_geometry import depth_lookup


@torch.inference_mode()
def evaluate(args):
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = load_rgb(args.assets) if args.teacher else EventDVLT(args.assets).cuda().eval()
    if not args.teacher:
        model.load_state_dict(checkpoint['model'], strict=False)
    encoder = (EventField(args.f3_repository, args.assets/'f3.pth', 4)
               if not args.teacher and checkpoint['config']['f3'] else None)
    if not args.full_fov:
        mapping, _ = calibration(args.calibration)
        xy, z_factor = depth_lookup(args.calibration, 168, 224)
        k = torch.tensor([[888.8889*224/640, 0, 111.5], [0, 888.8889*168/480, 83.5],
                          [0, 0, 1]], device='cuda')
    results = []
    for name in args.flights:
        with ExitStack() as stack:
            truth = stack.enter_context(h5py.File(args.root/f'{name}_gt.hdf5'))
            gt = truth['davis/left']
            if args.full_fov:
                from mvsec_geometry_data import MVSECGeometrySequence

                source = MVSECGeometrySequence(args.root,name,args.calibration,stack)
                camera = source.camera
                k = torch.as_tensor(source.k, device='cuda', dtype=torch.float32)
            else:
                data = stack.enter_context(h5py.File(args.root/f'{name}_data.hdf5'))
                camera = data['davis/left']
            times = camera['image_raw_ts'][:]
            pose_times, pose = gt['pose_ts'][:], gt['pose'][:]
            depth_times = gt['depth_image_rect_ts'][:]
            interpolate = Slerp(pose_times, Rotation.from_matrix(pose[:, :3, :3]))
            eligible = np.flatnonzero((times >= pose_times[0]) & (times <= pose_times[-1]) &
                                      (times >= depth_times[0]) & (times <= depth_times[-1]))
            eligible = eligible[eligible > 0]
            if args.full_fov:
                eligible = np.intersect1d(eligible, source.eligible)
                gaps = (np.diff(eligible) != 1) | (np.diff(times[eligible]) > .2)
                groups = np.split(eligible, np.flatnonzero(gaps)+1)
                starts = np.concatenate([group[:len(group)//18*18:18] for group in groups])
            else:
                starts = eligible[:len(eligible)//18*18:18]
            if args.count is not None and len(starts) > args.count:
                starts = starts[np.linspace(0, len(starts)-1, args.count, dtype=int)]
            for start in starts:
                indices = np.arange(start, start+18)
                at = times[indices]
                gt_poses = np.repeat(np.eye(4)[None], 18, axis=0)
                gt_poses[:, :3, :3] = interpolate(at).as_matrix()
                gt_poses[:, :3, 3] = np.stack([np.interp(at, pose_times, pose[:, j, 3]) for j in range(3)], axis=-1)
                if args.full_fov:
                    gt_poses[:, :3, :3] = gt_poses[:, :3, :3] @ source.rect_rotation.T
                gt_poses = torch.as_tensor(np.linalg.solve(gt_poses[0], gt_poses), device='cuda', dtype=torch.float32)
                packets, depths = [], []
                events = camera['events']
                for i in indices:
                    if not args.teacher:
                        if args.full_fov:
                            packets.append(source.packet(i))
                        else:
                            block = events[lower_bound_after(events,times[i]-.05):lower_bound_after(events,times[i])]
                            coords = mapping[block[:,1].astype(int), block[:,0].astype(int)]
                            packets.append(np.column_stack([coords, block[:,2:]]).T)
                    d = int(np.argmin(np.abs(depth_times-times[i])))
                    depth = gt['depth_image_rect'][d]
                    depth = (cv2.resize(depth,(224,168),interpolation=cv2.INTER_NEAREST_EXACT)
                             if args.full_fov else depth[xy[...,1],xy[...,0]]/z_factor)
                    if abs(depth_times[d]-times[i]) > .025:
                        depth.fill(np.nan)
                    depths.append(depth)
                if args.teacher:
                    rgb = torch.as_tensor(np.stack([source.rgb(i) for i in indices]), device='cuda')
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        raw = model.forward_inference(rgb[None].permute(0,1,4,2,3).float()/255)
                else:
                    voxels = torch.stack([voxelize(packet) for packet in packets])[None].cuda()
                    field = (torch.stack([encoder(packet) for packet in packets])[None] if encoder
                             else voxels.new_zeros((1,18,32,96,128)))
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        tokens, pos, _ = model.geometry(voxels, field, steps=16)
                        raw = model.backbone._decode(tokens,168,224,1,18,pos)
                rays = raw['rays'].float()
                poses, intrinsics = rays_to_pose(rays,torch.ones_like(rays[...,0]),168,224,14)
                first_camera = poses[0,0]
                poses = torch.linalg.solve(first_camera, poses[0])
                estimate = raw['depth'][0,...,0].float()
                reference = torch.as_tensor(np.stack(depths),device='cuda',dtype=torch.float32)
                valid = (reference >= .1) & (reference <= args.max_depth) & torch.isfinite(estimate) & (estimate > 0)
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
                row = dict(flight=name,start=int(start),frames=indices.tolist(),
                    absrel=float(((aligned_depth-t).abs()/t).mean()),
                    delta1=float((torch.maximum(aligned_depth/t,t/aligned_depth)<1.25).float().mean()),
                    ate_m=float(offset.square().sum(-1).mean().sqrt()),
                    **{f'pose_auc{a}':float(pose_metrics[f'Auc_{a}']) for a in (3,20,30)},
                    point_absrel=float(point_error.mean()),
                    point_inlier25=float((point_error<.25).float().mean()),
                    point_inlier3=float((point_error<.03).float().mean()))
                if args.compare_pointmaps:
                    # DVLT's decoder points are in world coordinates. Apply the
                    # same first-camera gauge as the fitted-camera pointmap.
                    world = raw['world_points'][0].float()[valid]
                    direct = (world-first_camera[:3,3]) @ first_camera[:3,:3]
                    common = torch.isfinite(direct).all(-1) & torch.isfinite(xyz).all(-1)
                    if common.sum() < 10:
                        raise ValueError(f'Insufficient common pointmap support in {name}/{start}')
                    target_points = truth_xyz[common]
                    norm = target_points.norm(dim=-1).clamp_min(1e-8)
                    for source, points in (('native', direct), ('camera', xyz)):
                        aligned = align_pointmap(points[common], target_points)
                        error = (aligned-target_points).norm(dim=-1)/norm
                        row[f'{source}_rel_l2'] = float(error.mean())
                        row[f'{source}_ir3'] = float((error<.03).float().mean())
                    row['pointmap_common_fraction'] = float(common.float().mean())
                results.append(row)
            print(name,len(starts),'windows attempted',flush=True)
    if not results:
        raise ValueError('No windows with synchronized pose and depth')
    args.out.write_text(json.dumps(dict(step=None if args.teacher else checkpoint['step'],clip_views=18,
        depth_scale='median ratio per window', max_depth_m=args.max_depth, event_window_ms=50,
        camera_grid='full rectified DAVIS' if args.full_fov else 'EV3DS virtual crop',
        input='gray APS' if args.teacher else 'left events', rows=results),
        indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root','calibration','assets','f3_repository','checkpoint','out'):
        parser.add_argument(key,type=Path)
    parser.add_argument('--flights',nargs='+',default=['indoor_flying1','indoor_flying2','indoor_flying3'])
    parser.add_argument('--count',type=int,help='Maximum number of clips per flight')
    parser.add_argument('--compare-pointmaps',action='store_true',
                        help='Compare direct decoder and fitted-camera points on common support')
    parser.add_argument('--full-fov',action='store_true',help='Use the official rectified DAVIS grid')
    parser.add_argument('--teacher',action='store_true',help='Run the frozen RGB teacher on left APS')
    parser.add_argument('--max-depth',type=float,default=50.)
    args = parser.parse_args()
    if args.teacher and not args.full_fov:
        parser.error('--teacher requires --full-fov')
    if not np.isfinite(args.max_depth) or args.max_depth < .1 or (args.count is not None and args.count < 1):
        parser.error('Provide a valid depth cap and positive window count')
    evaluate(args)
