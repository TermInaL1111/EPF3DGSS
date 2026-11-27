"""Evaluate frozen event geometry on held-out DSEC stereo frames with sparse depth."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dsec_data import DSECSequence, stereo_supervision
from dvlt.common.geometry import umeyama_alignment
from dvlt.common.rays import rays_to_pose
from dvlt.metric.pose import compute_pose_metrics
from event_dvlt import EventDVLT, voxelize
from event_field import EventField


def lift(depth, k):
    """Unproject image pixels to camera coordinates using native-grid intrinsics."""
    h, w = depth.shape
    y, x = torch.meshgrid(torch.arange(h,device=depth.device),
                          torch.arange(w,device=depth.device),indexing='ij')
    rays = torch.stack(((x-k[0,2])/k[0,0],(y-k[1,2])/k[1,1],torch.ones_like(depth)),-1)
    return rays*depth[...,None]


def native_intrinsics(k):
    k = k.clone()
    ratio = k.new_tensor([640/224,480/168])
    k[:2,:2] *= ratio[:,None]
    k[:2,2] = (k[:2,2]+.5)*ratio-.5
    return k


def align_pointmap(predicted, truth):
    # Match the upstream evaluator's single 95th percentile refinement.
    rotation, translation, scale = umeyama_alignment(predicted.T, truth.T)
    aligned = scale * (predicted @ rotation.T) + translation
    residual = (aligned-truth).norm(dim=-1)
    inliers = residual <= torch.quantile(residual, .95)
    if inliers.sum() >= 10:
        rotation, translation, scale = umeyama_alignment(predicted[inliers].T, truth[inliers].T)
        aligned = scale * (predicted @ rotation.T) + translation
    return aligned


def scores(raw, reference, intrinsics_truth, relative_truth, max_depth):
    estimate = raw['depth'].float()[0].permute(0,3,1,2)
    estimate = F.interpolate(estimate,(480,640),mode='bilinear',align_corners=False)[:,0]
    labelled = (reference>0) & (reference<=max_depth)
    valid = labelled & torch.isfinite(estimate) & (estimate>0)
    if not valid.any():
        raise ValueError('No finite, positive depth predictions on labelled pixels')
    pred, gt = estimate[valid], reference[valid]
    # One scale per stereo pair, shared across both views. No shift is fitted.
    scale = (gt/pred).median()
    depth = pred*scale
    rays = raw['rays'].float()
    poses,intrinsics = rays_to_pose(rays,torch.ones_like(rays[...,0]),168,224,14)
    relative = torch.linalg.solve(poses[0,0],poses[0,1])
    point_errors = []
    predicted_points, true_points = [], []
    for view in range(2):
        mask = valid[view]
        pred_xyz = lift(estimate[view]*scale,native_intrinsics(intrinsics[0,view]))[mask]
        gt_xyz = lift(reference[view],native_intrinsics(intrinsics_truth[view]))[mask]
        if view == 1:
            pred_xyz = pred_xyz @ relative[:3,:3].T + relative[:3,3]*scale
            gt_xyz = gt_xyz @ relative_truth[:3,:3].T + relative_truth[:3,3]
        point_errors.append((pred_xyz-gt_xyz).norm(dim=-1)/gt_xyz.norm(dim=-1).clamp_min(1e-8))
        predicted_points.append(pred_xyz/scale)
        true_points.append(gt_xyz)
    point_errors = torch.cat(point_errors)
    pred_xyz, gt_xyz = torch.cat(predicted_points), torch.cat(true_points)
    aligned = align_pointmap(pred_xyz, gt_xyz)
    rel_l2 = (aligned-gt_xyz).norm(dim=-1)/gt_xyz.norm(dim=-1).clamp_min(1e-8)
    gt_poses = torch.stack((torch.eye(4, device=poses.device), relative_truth))
    pose_metrics = compute_pose_metrics(poses[0].double(), gt_poses.double(), thresholds=(3, 20, 30))
    rotation = relative[:3,:3].T @ relative_truth[:3,:3]
    cos_rot = ((rotation.trace()-1)/2).clamp(-1,1)
    a,b = relative[:3,3],relative_truth[:3,3]
    translation_deg = None
    if a.norm()>1e-8 and b.norm()>1e-8:
        translation_deg = float(F.cosine_similarity(a,b,dim=0).clamp(-1,1).acos().rad2deg())
    return dict(point_absrel=float(point_errors.mean()),
                point_inlier25=float((point_errors<.25).float().mean()),
                dvlt_point_rel_l2=float(rel_l2.mean()),
                dvlt_point_ir3=float((rel_l2<.03).float().mean()),
                sim3_point_inlier25=float((rel_l2<.25).float().mean()),
                dvlt_pose_auc3=float(pose_metrics['Auc_3']),
                pose_auc20=float(pose_metrics['Auc_20']),
                dvlt_pose_auc30=float(pose_metrics['Auc_30']),
                depth_inlier3=float(((depth-gt).abs()/gt<.03).float().mean()),
                absrel=float(((depth-gt).abs()/gt).mean()),
                delta1=float((torch.maximum(depth/gt,gt/depth)<1.25).float().mean()),
                valid_prediction_fraction=float(valid.sum()/labelled.sum()),
                rotation_deg=float(cos_rot.acos().rad2deg()), translation_deg=translation_deg,
                aligned_scale=float(scale))


@torch.inference_mode()
def main(args):
    model = EventDVLT(args.assets).cuda().eval()
    checkpoint = torch.load(args.checkpoint,weights_only=True)
    use_field = bool(checkpoint['config']['f3'])
    expected = {n for n,p in model.named_parameters() if p.requires_grad
                and not n.startswith(('gs_head.', 'field_features.', 'geometry_features.'))
                and n != 'log_metric_scale' and (use_field or not n.startswith('event_adapter.'))}
    if set(checkpoint['model']) != expected:
        raise ValueError('Invalid geometry checkpoint')
    if set(args.sequences)&set(checkpoint['config']['train']):
        raise ValueError('Evaluation sequences overlap training')
    model.load_state_dict(checkpoint['model'],strict=False)
    encoder = EventField(args.f3_repository,args.assets/'f3.pth',4) if use_field else None
    rows = []
    for seq in args.sequences:
        source = DSECSequence(args.root,seq)
        try:
            n = len(source.indices)
            selected = np.linspace(0,n-1,min(n,args.count),dtype=int)
            if not len(selected):
                raise ValueError(f'No labelled stereo frames in {seq}')
            for offset in selected:
                index = source.indices[offset]
                packets = [source.packet(v,index) for v in (0,1)]
                voxel = torch.stack([voxelize(p) for p in packets])[None].cuda()
                field = (torch.stack([encoder(p) for p in packets])[None] if encoder
                         else voxel.new_zeros((1,2,32,96,128)))
                depth,ki,poses,*_ = stereo_supervision(source,index)
                depth = torch.as_tensor(depth,device='cuda')
                ki = torch.as_tensor(ki,device='cuda',dtype=torch.float32)
                pose = torch.as_tensor(poses[1],device='cuda',dtype=torch.float32)
                row = dict(sequence=seq,frame=index)
                for name,v,f in [('matched',voxel,field),
                                 ('zero',torch.zeros_like(voxel),torch.zeros_like(field))]:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        tokens,pos,_ = model.geometry(v,f,steps=16)
                        raw = model.backbone._decode(tokens,168,224,1,2,pos)
                    row[name] = scores(raw,depth,ki,pose,args.max_depth)
                rows.append(row)
            print(seq,len(selected),'frames',flush=True)
        finally:
            source.close()
    report = dict(depth_cap_m=args.max_depth,epoch=checkpoint['epoch'],step=checkpoint['step'],rows=rows,
                  dvlt_protocol='Stereo pointmaps use upstream Sim(3) with one 95th percentile refinement; pose uses upstream AUC on each stereo pair',
                  protocol='20 ms rectified stereo, sparse camera depth, one median scale per pair')
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root','assets','f3_repository','checkpoint','out'):
        parser.add_argument(key,type=Path)
    parser.add_argument('--sequences',required=True,nargs='+')
    parser.add_argument('--max-depth',type=float,default=50.)
    parser.add_argument('--count',type=int,default=24)
    main(parser.parse_args())
