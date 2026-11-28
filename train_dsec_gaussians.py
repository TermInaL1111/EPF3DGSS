"""Train a grayscale Gaussian head with a frozen geometry backbone."""
import argparse
from contextlib import ExitStack, closing
from functools import lru_cache
import fcntl
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dsec_data import DSECSequence, resize_intrinsics, stereo_supervision
from evaluate_eventscape_depth import depth_scores
from event_depth_boundary import event_bins, load_temporal_boundary, temporal_corrector
from event_dvlt import EventDVLT, load_rgb, load_scene_weights, render, voxelize
from event_field import EventField
from image_metrics import PerceptualMetrics, gray_metrics


def geometry_audit(cloud, depth, intrinsics, poses, query, query_k):
    estimate = cloud['source_depth']
    valid = (depth > 0) & torch.isfinite(estimate) & (estimate > 0)
    predicted, reference = estimate[valid], depth[valid]
    if not len(reference):
        raise ValueError('No common source depth support')
    aligned = depth_scores(predicted,reference)
    fitted_k, fitted_pose = cloud['fitted_intrinsics'], cloud['fitted_c2w']
    rays = cloud['raw_geometry']['rays'][...,:3].float()
    height,width = rays.shape[-3:-1]
    y,x = torch.meshgrid(torch.arange(height,device=rays.device),torch.arange(width,device=rays.device),indexing='ij')
    pixels = torch.stack((x,y,torch.ones_like(x)),-1).float()
    reference_rays = (pixels.flatten(0,1) @ torch.linalg.inv(intrinsics).transpose(-1,-2)).reshape_as(rays)
    local_rays = (rays.flatten(2,3) @ fitted_pose[...,:3,:3]).reshape_as(rays)
    angle = F.cosine_similarity(local_rays,reference_rays,dim=-1).clamp(-1,1).acos().rad2deg()
    h,w = depth.shape[-2:]
    y,x = torch.meshgrid(torch.arange(h,device=rays.device),torch.arange(w,device=rays.device),indexing='ij')
    sensor_pixels = torch.stack(((x+.5)*width/w-.5,(y+.5)*height/h-.5,torch.ones_like(x)),-1).float()
    source_rays = sensor_pixels.flatten(0,1) @ torch.linalg.inv(intrinsics).transpose(-1,-2)
    target_from_source = torch.linalg.solve(query[:,None],poses)
    def project(z):
        local = source_rays*z.flatten(2)[...,None]
        target = local @ target_from_source[...,:3,:3].transpose(-1,-2) + target_from_source[...,:3,3].unsqueeze(-2)
        projected = target @ query_k[:,None].transpose(-1,-2)
        return projected[...,:2]/projected[...,2:].clamp_min(1e-6), projected[...,2]
    estimated_px, estimated_z = project(estimate)
    reference_px, reference_z = project(depth)
    shared = valid.flatten(2) & (estimated_z>0) & (reference_z>0)
    shared &= ((reference_px>=0)&(reference_px<reference_px.new_tensor([w,h]))).all(-1)
    error_px = (estimated_px-reference_px).norm(dim=-1)[shared]
    fitted_relative = torch.linalg.solve(fitted_pose[:,0],fitted_pose[:,1])
    rotation = fitted_relative[:,:3,:3].transpose(-1,-2) @ poses[:,1,:3,:3]
    rotation_cos = ((rotation.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
    offset = fitted_relative[:,:3,3] * cloud['metric_scale']
    reference_offset = poses[:,1,:3,3]
    return dict(
        source_depth_fraction=float(valid.float().mean()),
        metric_depth_absrel=float(((predicted-reference).abs()/reference).mean()),
        aligned_depth_absrel=aligned['absrel'], reference_depth_scale=aligned['scale'],
        focal_x_ratio=float((fitted_k[...,0,0]/intrinsics[...,0,0]).mean()),
        focal_y_ratio=float((fitted_k[...,1,1]/intrinsics[...,1,1]).mean()),
        principal_point_error_px=float((fitted_k[...,:2,2]-intrinsics[...,:2,2]).norm(dim=-1).mean()),
        local_ray_angle_deg=float(angle.mean()),
        rotation_error_deg=float(rotation_cos.acos().rad2deg().mean()),
        translation_error_m=float((offset-reference_offset).norm(dim=-1).mean()),
        depth_reprojection_px=float(error_px.mean()) if len(error_px) else None,
    )


def rgb_appearance_loss(teacher, source_gray, source_rgb):
    """Match the frozen DVLT image encoder on paired source-camera images."""
    rgb = source_rgb.permute(0, 3, 1, 2)[None].float() / 255
    rgb = F.interpolate(rgb[0], (168, 224), mode='area')[None]
    render_rgb = F.interpolate(source_gray[:, None], (168, 224), mode='area')
    render_rgb = render_rgb.expand(-1, 3, -1, -1)[None]
    with torch.autocast(source_gray.device.type, dtype=torch.bfloat16):
        with torch.no_grad():
            target_tokens = teacher._encode_images(rgb)
        predicted_tokens = teacher._encode_images(render_rgb)
    return (1 - F.cosine_similarity(predicted_tokens.float(), target_tokens.float(), dim=-1)).mean()


def run(args, data, checkpoint, test):
    model = EventDVLT(args.assets).cuda().eval()
    weights = checkpoint['model']
    geometry_names = {n for n,p in model.named_parameters() if p.requires_grad
                      and not n.startswith(('gs_head.', 'field_features.', 'geometry_features.'))
                      and n != 'log_metric_scale'}
    use_field = bool(checkpoint['config']['f3'])
    expected = geometry_names if use_field else {n for n in geometry_names if not n.startswith('event_adapter.')}
    if set(weights) != expected:
        raise ValueError('Unexpected parameters in distilled geometry checkpoint')
    model.load_state_dict(weights, strict=False)
    model.configure_head(args.head)
    if args.init_head:
        previous = torch.load(args.init_head,map_location='cpu',weights_only=True)
        if (previous.get('head') != args.head or previous.get('boundary') or
                any(name not in previous['model'] or not torch.equal(tensor,previous['model'][name])
                    for name,tensor in weights.items())):
            raise ValueError('Initial head must use the same frozen geometry and head type')
        load_scene_weights(model,previous)
    boundary = None
    if args.boundary:
        boundary_source = torch.load(args.boundary, weights_only=True, map_location='cpu')
        boundary = load_temporal_boundary(boundary_source['model']['temporal'], 'cuda')
    for name, parameter in model.named_parameters():
        trainable = name.startswith(('gs_head.', 'field_features.', 'geometry_features.'))
        parameter.requires_grad_(trainable or (name == 'log_metric_scale' and args.geometry == 'predicted'
                                                and args.head == 'legacy'))
    params = {n:p for n,p in model.named_parameters() if p.requires_grad}
    perceptual = PerceptualMetrics("cuda") if args.lpips or args.lpips_weight else None
    appearance_teacher = load_rgb(args.assets) if args.rgb_teacher_weight else None
    optimizer = torch.optim.AdamW(params.values(), lr=args.lr)
    encoder = EventField(args.f3_repository,args.assets/'f3.pth',4) if use_field else None
    detail_encoder = EventField(args.f3_repository,args.assets/'f3.pth',8) if use_field else None
    eval_only = args.steps == 0 and args.init_head is not None
    train = [] if eval_only else [(s,i) for s in args.train for i in data[s].indices]
    val = [] if eval_only else [(s,i) for s in args.val for i in data[s].indices]
    if not eval_only and args.mvsec_calibration:
        val = [(seq, indices[i]) for seq in args.val
               if (indices := data[seq].indices)
               for i in np.linspace(0,len(indices)-1,min(len(indices),args.val_limit),dtype=int)]
    elif not eval_only and len(val)>args.val_limit:
        val = [val[i] for i in np.linspace(0,len(val)-1,args.val_limit,dtype=int)]
    if not eval_only and (not train or not val):
        raise ValueError('Both splits require usable source and target windows')
    mixed = bool(args.eventscape_root)
    train_rng = torch.Generator().manual_seed(args.seed) if mixed else None
    print('training windows',len(train),'validation windows',len(val),flush=True)
    temporal = bool(args.mvsec_calibration or args.eventscape_cameras or args.dsec_temporal_cameras)

    @lru_cache(maxsize=32)
    def sample(seq,index):
        source = data[seq]
        packets = [source.packet(v,index) for v in (0,1)]
        voxel = torch.stack([voxelize(p) for p in packets]).half()
        field = (torch.stack([encoder(p).half().cpu() for p in packets]) if encoder
                 else torch.zeros((2,32,96,128),dtype=torch.float16))
        detail = (torch.stack([detail_encoder(p).half().cpu() for p in packets]) if detail_encoder
                  else torch.zeros((2,32,240,320),dtype=torch.float16))
        supervision = source.supervision(index) if temporal else stereo_supervision(source,index)
        labels = tuple(torch.as_tensor(x,dtype=torch.float32) for x in supervision)
        return voxel,field,detail,*labels

    @torch.no_grad()
    def initialize_scale():
        # DVLT scale is ambiguous, but the target camera baseline is in metres.
        # Fit a single scalar on training frames only.
        selected = [train[i] for i in np.linspace(0,len(train)-1,min(12,len(train)),dtype=int)]
        rows = []
        for sequence,index in selected:
            voxel,field,_,depth,*_ = sample(sequence,index)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                tokens,pos,_ = model.geometry(voxel[None].cuda().float(),field[None].cuda().float(),steps=16)
                raw = model.backbone._decode(tokens,168,224,1,2,pos)
            estimate = F.interpolate(raw['depth'][0].float().permute(0,3,1,2),
                                     (480,640),mode='bilinear',align_corners=False)[:,0]
            target = depth.cuda()
            valid = (target>0)&(estimate>0)&torch.isfinite(estimate)
            if not valid.any() and temporal:
                continue
            if not valid.any():
                raise ValueError(f'No valid depth for scale calibration in {sequence}/{index}')
            rows.append(dict(sequence=sequence,frame=index,
                             scale=float((target[valid]/estimate[valid]).median())))
        if not rows:
            raise ValueError('No usable training depths for scale calibration')
        scale = float(np.median([row['scale'] for row in rows]))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid training scale')
        model.log_metric_scale.fill_(np.log(scale))
        (args.out/'scale_initialization.json').write_text(json.dumps(
            dict(scale=scale,training_frames=rows),indent=2,allow_nan=False)+'\n')
        print('training-only initial scale',scale,flush=True)

    if args.geometry == 'predicted' and not args.init_head:
        initialize_scale()

    @lru_cache(maxsize=32)
    def event_observation(seq,index):
        return tuple(torch.as_tensor(x, dtype=torch.float32) for x in data[seq].event_observation(index))

    @lru_cache(maxsize=4)
    def boundary_counts(seq, index):
        source = data[seq]
        return torch.stack([event_bins(source.packet(view, index), (480,640)) for view in (0,1)])

    def predict(item, empty=False, training=False, source_control=None):
        voxel,field,detail,depth,ki,poses,query,k,gray = (x[None].cuda() for x in sample(*item))
        if empty:
            voxel,field,detail = torch.zeros_like(voxel),torch.zeros_like(field),torch.zeros_like(detail)
        calibrated = (depth,ki,poses) if args.geometry == 'calibrated' else None
        cameras = None
        if source_control in ('both','reference_depth'):
            cameras = (ki,poses)
        elif source_control == 'intrinsics' or args.known_source_intrinsics:
            cameras = (ki,None)
        elif source_control == 'poses':
            cameras = (None,poses)
        elif training and args.head != 'legacy':
            cameras = (ki,poses)
        refine = (temporal_corrector(boundary,boundary_counts(*item).cuda(),empty,args.preserve_boundary_scale)
                  if boundary else None)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            cloud = model(voxel.float(),field.float(),detail_field=detail.float(),
                          steps=16,calibrated_geometry=calibrated,
                          source_cameras=cameras, depth_refinement=refine,
                          return_geometry=args.source_depth_control and source_control == 'both' and not empty,
                          reference_metric_depth=depth if source_control == 'reference_depth' else None,
                          event_leaf_ranking=args.event_leaves, matched_leaf_budget=bool(boundary))
        coverage = None
        if args.source_camera_control:
            image,alpha = render(cloud,query,k,return_alpha=True)
            coverage = dict(alpha_mean=float(alpha.mean()),
                            alpha_covered_fraction=float((alpha>=.5).float().mean()))
            if 'fitted_intrinsics' in cloud:
                coverage.update(geometry_audit(cloud,depth,ki,poses,query,k))
        else:
            image = render(cloud,query,k)
        count = cloud.get('gaussian_count')
        if (args.eventscape_cameras or args.dsec_temporal_cameras) and count is None:
            count = (cloud['centers'].shape[1],)
        event_loss = None
        if args.event_weight and training:
            previous,grid,counts = (x.cuda() for x in event_observation(*item))
            if len(counts):
                earlier = render(cloud,previous[None],k)
                images = torch.stack((earlier,image),dim=1)
                values = F.grid_sample(images,grid[None,None],align_corners=False)[0,:,0]
                before,after = (values+1/255).log()
                event_loss = F.smooth_l1_loss(after-before,args.event_threshold*counts,
                                              beta=args.event_threshold)
        source_loss = teacher_loss = None
        if args.source_view_weight and training:
            queries,source_k,labels = (torch.as_tensor(x,device=image.device,dtype=torch.float32)
                                       for x in data[item[0]].source_supervision(item[1]))
            source_images = torch.cat([render(cloud,pose[None],source_k[None]) for pose in queries])
            labels = labels/255
            source_l1 = (source_images-labels).abs().mean()
            _,source_ssim = gray_metrics(source_images,labels)
            source_loss = (1-args.ssim_weight)*source_l1 + args.ssim_weight*(1-source_ssim.mean())
            if appearance_teacher is not None:
                rgb = torch.as_tensor(data[item[0]].source_rgb(item[1]), device=image.device)
                teacher_loss = rgb_appearance_loss(appearance_teacher, source_images, rgb)
        return image,gray/255,count,event_loss,coverage,source_loss,teacher_loss

    @lru_cache(maxsize=32)
    def overlap(seq,index):
        # Nominal angular overlap from calibration, independent of model output.
        _,_,_,_,ki,poses,query,k,_ = sample(seq,index)
        y,x = np.indices((480,640))
        pixels = np.stack([x,y,np.ones_like(x)],-1)
        rays = pixels @ np.linalg.inv(k.numpy()).T
        mask = np.zeros((480,640),dtype=bool)
        for source_k,pose in zip(ki,poses):
            transform = np.linalg.solve(pose.numpy(),query.numpy())
            projected = rays @ transform[:3,:3].T @ resize_intrinsics(
                source_k.numpy(),(224,168),(640,480)).T
            z = projected[...,2]
            mask |= ((z>0)&(projected[...,0]>=0)&(projected[...,0]<640*z)&
                     (projected[...,1]>=0)&(projected[...,1]<480*z))
        if not mask.any():
            raise ValueError(f'No nominal camera overlap for {seq}')
        return torch.from_numpy(mask).cuda()

    @torch.no_grad()
    def evaluate(windows, label, native=False):
        rows = []
        for index,item in enumerate(windows):
            row = dict(sequence=item[0], frame=item[1])
            preview = []
            mask = overlap(*item)
            row["overlap_fraction"] = float(mask.float().mean())
            native_gray = None
            if native:
                sequence = data[item[0]]
                native_gray = torch.as_tensor(sequence.native_gray(item[1]), device=mask.device)[None]/255
                source = cv2.cvtColor(sequence.sequence.rgb(item[1]), cv2.COLOR_RGB2GRAY)
                source = torch.as_tensor(source, device=mask.device)[None]/255
                psnr,ssim = gray_metrics(source,native_gray)
                row.update(source_copy_psnr=float(psnr[0]),source_copy_ssim=float(ssim[0]))
            if label.startswith('test') and args.dsec_temporal_cameras and not native:
                source = torch.as_tensor(data[item[0]].gray(item[1]),device=mask.device)[None]/255
                target = sample(*item)[-1][None].to(mask.device)/255
                psnr,ssim = gray_metrics(source,target)
                row.update(source_copy_psnr=float(psnr[0]),source_copy_ssim=float(ssim[0]))
            variants = [('matched',False,None),('zero',True,None)]
            if args.source_camera_control:
                variants += [('matched_calibrated',False,'both'),('zero_calibrated',True,'both')]
            if args.source_camera_components:
                variants += [('matched_intrinsics',False,'intrinsics'),('matched_poses',False,'poses')]
            if args.source_depth_control:
                variants.append(('matched_reference_depth',False,'reference_depth'))
            for key,empty,source_control in variants:
                image,gray,count,_,coverage,_,_ = predict(item,empty,source_control=source_control)
                if coverage is not None:
                    row.update({f'{key}_{metric}':value for metric,value in coverage.items()})
                if count is not None:
                    row[key+'_gaussian_count'] = int(count[0])
                mse_overlap = (image-gray)[:,mask].square().mean()
                row[key+'_overlap_psnr'] = float(-10*mse_overlap.clamp_min(1e-12).log10())
                if native:
                    image = F.interpolate(image[:,None], size=(256,512), mode='area')[:,0]
                    gray = native_gray
                psnr,ssim = gray_metrics(image,gray)
                row[key+'_psnr'],row[key+'_ssim'] = float(psnr[0]),float(ssim[0])
                row[key+"_ssim_evggs"] = float(gray_metrics(image,gray,padded=True)[1][0])
                if args.lpips:
                    row.update({f"{key}_{metric}":value for metric,value in perceptual(image,gray).items()})
                if index == 0:
                    if not preview:
                        preview.append(gray[0])
                    preview.append(image[0])
            if preview:
                comparison = torch.cat(preview,dim=1).clamp(0,1).mul(255).round().byte().cpu().numpy()
                if not cv2.imwrite(str(args.out/f'preview_{label}.png'),comparison):
                    raise OSError('Could not save the validation preview')
            rows.append(row)
        (args.out/f'val_{label}.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        print('evaluation',label,{k:sum(r[k] for r in rows)/len(rows)
                                  for k in ('matched_psnr','zero_psnr','matched_ssim','zero_ssim')},flush=True)

    if not eval_only:
        evaluate(val,'000000')
    for step in range(1,args.steps+1):
        if mixed:
            position = (step-1) % len(train)
            if position == 0:
                order = torch.randperm(len(train),generator=train_rng).tolist()
            item = train[order[position]]
        else:
            item = train[int(torch.randint(len(train),(1,)))]
        image,gray,_,event_loss,_,source_loss,teacher_loss = predict(item,training=True)
        l1 = (image-gray).abs().mean()
        _,ssim = gray_metrics(image,gray)
        loss = (1-args.ssim_weight)*l1 + args.ssim_weight*(1-ssim.mean())
        if event_loss is not None:
            loss = loss + args.event_weight*event_loss
        if source_loss is not None:
            loss = loss + args.source_view_weight*source_loss
        if teacher_loss is not None:
            loss = loss + args.rgb_teacher_weight*teacher_loss
        perceptual_loss = None
        if args.lpips_weight:
            reduced = F.interpolate(torch.stack((image,gray),dim=1),size=(192,256),mode='area')
            perceptual_loss = perceptual.vgg_loss(*reduced.unbind(1))
            loss = loss + args.lpips_weight*perceptual_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params.values(),1.,error_if_nonfinite=True)
        optimizer.step()
        if mixed:
            with (args.out/'train.jsonl').open('a') as stream:
                row = dict(step=step, sequence=item[0], frame=item[1],
                           loss=float(loss.detach()), grad_norm=float(norm))
                if perceptual_loss is not None:
                    row['lpips_loss'] = float(perceptual_loss.detach())
                if source_loss is not None:
                    row['source_view_loss'] = float(source_loss.detach())
                if teacher_loss is not None:
                    row['rgb_teacher_loss'] = float(teacher_loss.detach())
                stream.write(json.dumps(row,allow_nan=False)+'\n')
        if step==1 or step%20==0:
            print('step',step,'loss',float(loss.detach()),'l1',float(l1.detach()),
                  'ssim',float(ssim.detach().mean()),'grad',float(norm),
                  'event',float(event_loss.detach()) if event_loss is not None else None,flush=True)
        if step==args.steps or step%100==0:
            tmp = args.out/'latest.pt.tmp'
            state = weights | {n:p.detach().cpu() for n,p in params.items()}
            if args.head != 'legacy':
                state['log_metric_scale'] = model.log_metric_scale.detach().cpu()
            payload = dict(model=state, step=step, geometry=args.geometry, f3=use_field, head=args.head)
            if boundary:
                payload.update(boundary={key: p.detach().cpu() for key,p in boundary.state_dict().items()},
                               event_leaves=args.event_leaves,
                               preserve_boundary_scale=args.preserve_boundary_scale)
            torch.save(payload,tmp)
            tmp.replace(args.out/'latest.pt')
            if args.save_every and step % args.save_every == 0:
                archive = args.out/f'step_{step:06}.pt.tmp'
                torch.save(payload,archive)
                archive.replace(args.out/f'step_{step:06}.pt')
            if step == args.steps:
                archive = args.out/f'step_{step:06}.pt.tmp'
                torch.save(payload,archive)
                archive.replace(args.out/f'step_{step:06}.pt')
        if mixed and step % len(train) == 0 and step < args.steps:
            evaluate(val,f'{step:06}')
    if args.steps:
        evaluate(val,f'{args.steps:06}')
    for label, (names, native) in test.items():
        evaluate([(name,index) for name in names for index in data[name].indices],label,native)


def main(args):
    if not np.isfinite(args.source_view_weight) or args.source_view_weight < 0:
        raise ValueError('Source-view weight must be finite and nonnegative')
    if args.source_view_weight and not (args.eventscape_cameras or args.dsec_temporal_cameras):
        raise ValueError('Source-view supervision requires EventScape or DSEC temporal cameras')
    if not np.isfinite(args.rgb_teacher_weight) or args.rgb_teacher_weight < 0:
        raise ValueError('RGB teacher weight must be finite and nonnegative')
    if args.rgb_teacher_weight and (not args.source_view_weight or not args.steps):
        raise ValueError('RGB appearance distillation requires training with source-view supervision')
    if not 0 <= args.ssim_weight <= 1:
        raise ValueError('SSIM weight must be between 0 and 1')
    if args.steps < 0 or args.val_limit < 1:
        raise ValueError("Steps must be nonnegative and validation limit must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.lpips_weight) or args.lpips_weight < 0:
        raise ValueError('Learning rate must be positive and LPIPS weight nonnegative')
    if args.init_head and (args.geometry != 'predicted' or args.boundary):
        raise ValueError('Head initialization requires predicted geometry without a boundary head')
    if args.source_camera_control and (args.steps or not args.init_head):
        raise ValueError('Source-camera comparison requires evaluation of a frozen trained head')
    if args.source_camera_components and not args.source_camera_control:
        raise ValueError('Source-camera components require the full source-camera comparison')
    if args.source_depth_control and (not args.source_camera_control or args.head != 'octree'):
        raise ValueError('Source depth comparison requires the frozen octree camera control')
    if args.known_source_intrinsics and (args.geometry != 'predicted' or args.source_camera_control):
        raise ValueError('Known source intrinsics require predicted geometry and the standard render protocol')
    checkpoint = torch.load(args.geometry_checkpoint, map_location='cpu', weights_only=True)
    eventscape_splits, dsec_splits = {}, {}
    mixed = bool(args.eventscape_cameras and args.dsec_temporal_cameras)
    if args.dsec_temporal_cameras and mixed != bool(args.eventscape_root):
        raise ValueError('Specify --eventscape-root only when mixing DSEC and EventScape')
    if args.mvsec_calibration and args.eventscape_cameras:
        raise ValueError('MVSEC calibration cannot be mixed with EventScape manifests')
    if args.eventscape_cameras:
        from eventscape_gaussian_data import load_camera_splits

        eventscape_splits = load_camera_splits(args.eventscape_cameras)
    if args.dsec_temporal_cameras:
        from dsec_temporal_gaussian_data import load_dsec_camera_splits

        dsec_splits = load_dsec_camera_splits(args.dsec_temporal_cameras)
    test = {}
    if eventscape_splits:
        test['test_eventscape' if mixed else 'test'] = (eventscape_splits['test'],True)
    if dsec_splits:
        test['test_dsec' if mixed else 'test'] = (dsec_splits['test'],False)
    if test:
        if args.train or args.val:
            raise ValueError('Train and val splits come from the camera manifests')
        args.train = [*dsec_splits.get('train',{}),*eventscape_splits.get('train',{})]
        args.val = [*dsec_splits.get('val',{}),*eventscape_splits.get('val',{})]
        if mixed:
            for split in dsec_splits:
                if dsec_splits[split].keys() & eventscape_splits[split].keys():
                    raise ValueError(f'A DSEC and EventScape sequence share a name in {split}')
        if args.steps == 0 and args.init_head is not None:
            eventscape_splits = {'test': eventscape_splits['test']} if eventscape_splits else {}
            dsec_splits = {'test': dsec_splits['test']} if dsec_splits else {}
            args.train, args.val = [], []
    elif not args.train or not args.val:
        raise ValueError('Specify both --train and --val')
    if args.head is None:
        args.head = ('fixed' if args.geometry == 'predicted' and checkpoint['config'].get('cover_train')
                     else 'legacy')
    if args.head != 'legacy' and args.geometry != 'predicted':
        raise ValueError('Fixed and voxel heads use student depth')
    if (args.mvsec_calibration or test) and args.head == 'legacy':
        raise ValueError('Temporal training supports fixed, voxel and octree heads')
    if (args.boundary and args.head != 'octree') or (args.event_leaves and not args.boundary):
        raise ValueError('Boundary-controlled training requires an octree head and temporal weights')
    if args.preserve_boundary_scale and not args.boundary:
        raise ValueError('Scale preservation requires a boundary checkpoint')
    if not np.isfinite(args.event_weight) or args.event_weight < 0:
        raise ValueError('Event loss weight must be finite and nonnegative')
    if args.event_weight and (not args.mvsec_calibration or args.head != 'voxel' or
                              not np.isfinite(args.event_threshold) or args.event_threshold <= 0):
        raise ValueError('Event supervision requires an MVSEC voxel head and a positive threshold')
    if set(args.train)&set(args.val):
        raise ValueError('Specify nonoverlapping training and validation sequences')
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    if args.mvsec_calibration is None:
        config.pop('mvsec_calibration')
    for key in ('eventscape_cameras','dsec_temporal_cameras','eventscape_root'):
        if config[key] is None:
            config.pop(key)
    if not args.event_weight:
        config.pop('event_weight')
        config.pop('event_threshold')
    if not args.boundary:
        config.pop('boundary')
        config.pop('event_leaves')
    if not args.preserve_boundary_scale:
        config.pop('preserve_boundary_scale')
    if not args.init_head:
        config.pop('init_head')
    if not args.lpips_weight:
        config.pop('lpips_weight')
    if not args.source_view_weight:
        config.pop('source_view_weight')
    if not args.rgb_teacher_weight:
        config.pop('rgb_teacher_weight')
    if not args.source_camera_control:
        config.pop('source_camera_control')
    if not args.source_camera_components:
        config.pop('source_camera_components')
    if not args.source_depth_control:
        config.pop('source_depth_control')
    if not args.known_source_intrinsics:
        config.pop('known_source_intrinsics')
    if args.lr == 3e-4:
        config.pop('lr')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    lock_path = args.out.with_name(args.out.name+'.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.out.exists():
            recorded = json.loads((args.out/'config.json').read_text())
            validation = args.out/f'val_{args.steps:06}.json'
            checkpoint = args.out/'latest.pt'
            if (dict({'head':'legacy'}, **recorded) == config and validation.is_file() and
                    all((args.out/f'val_{label}.json').is_file() for label in test) and
                    (args.steps == 0 or checkpoint.is_file())):
                rows = json.loads(validation.read_text())
                state = (torch.load(checkpoint,map_location='cpu',weights_only=True)
                         if args.steps else {'step':0})
                if rows and state['step'] == args.steps:
                    print('Using completed run',args.out,flush=True)
                    return
            raise FileExistsError(f'Output already contains another or incomplete run: {args.out}')
        args.out.mkdir()
        torch.manual_seed(args.seed)
        (args.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        with ExitStack() as stack:
            data = {}
            if args.eventscape_cameras:
                from eventscape_gaussian_data import open_gaussian_sequences

                data = open_gaussian_sequences(args.eventscape_root or args.root,eventscape_splits,stack,
                                               target_offset=args.target_offset)
            if args.dsec_temporal_cameras:
                from dsec_temporal_gaussian_data import DSECTemporalGaussianSequence

                for split in dsec_splits.values():
                    for sequence, cameras in split.items():
                        try:
                            source = DSECTemporalGaussianSequence(args.root,sequence,cameras,
                                                                  target_offset=args.target_offset)
                        except ValueError:
                            if args.target_offset == 2:
                                raise
                            continue
                        data[sequence] = stack.enter_context(closing(source))
            if test:
                present = set(data)
                args.train = [name for name in args.train if name in present]
                args.val = [name for name in args.val if name in present]
                for label,(names,native) in list(test.items()):
                    test[label] = ([name for name in names if name in present],native)
            if not test:
                for sequence in args.train+args.val:
                    if args.mvsec_calibration:
                        from mvsec_gaussian_data import MVSECTemporalGaussianSequence

                        source = MVSECTemporalGaussianSequence(args.root,sequence,args.mvsec_calibration)
                    else:
                        source = DSECSequence(args.root,sequence)
                    data[sequence] = stack.enter_context(closing(source))
            run(args,data,checkpoint,test)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root','assets','f3_repository','geometry_checkpoint','out'):
        parser.add_argument(key,type=Path)
    parser.add_argument('--train',nargs='+')
    parser.add_argument('--val',nargs='+')
    parser.add_argument('--geometry',choices=('calibrated','predicted'),required=True)
    parser.add_argument('--head',choices=('legacy','fixed','voxel','octree','octree_pixel','octree_pixel_center','octree_pixel_fine','octree_pixel_deep','vg2gt','vg2gt_pixel','vg2gt_octree'),
                        help='Defaults to fixed for predicted geometry from full-coverage distillation')
    parser.add_argument('--boundary', type=Path, help='Frozen EventScape temporal boundary checkpoint')
    parser.add_argument('--preserve-boundary-scale', action='store_true')
    parser.add_argument('--event-leaves', action='store_true', help='Use event priority for octree selection')
    parser.add_argument('--lpips',action='store_true',help='Report VGG and AlexNet LPIPS during validation')
    parser.add_argument('--lpips-weight',type=float,default=0.,help='Train with additional VGG LPIPS supervision')
    parser.add_argument('--init-head',type=Path,help='Start from a trained head with identical frozen geometry')
    parser.add_argument('--source-view-weight',type=float,default=0.,
                        help='Weight of the mean photometric loss on the two source RGB views')
    parser.add_argument('--rgb-teacher-weight',type=float,default=0.,
                        help='Training-only weight of DVLT RGB encoder token loss on the two source renders')
    parser.add_argument('--source-camera-control',action='store_true',
                        help='Compare predicted and calibrated source cameras and report alpha coverage')
    parser.add_argument('--source-camera-components',action='store_true',
                        help='Also compare calibrated intrinsics and poses separately for real events')
    parser.add_argument('--source-depth-control',action='store_true',
                        help='Replace measured source depth with camera and feature inputs held fixed')
    parser.add_argument('--target-offset',type=int,choices=(2,4,6,8),default=2,
                        help='Target frame offset from the first source; 2 is the midpoint, 4 extrapolates')
    parser.add_argument('--known-source-intrinsics',action='store_true',
                        help='Use sensor intrinsics and event-predicted source poses in training and evaluation')
    parser.add_argument('--lr',type=float,default=3e-4)
    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument('--mvsec-calibration',type=Path,
                         help='Train from temporal windows of the left MVSEC camera')
    parser.add_argument('--eventscape-cameras',type=Path,
                         help='Train from EventScape RGB-D camera manifests (train/val/test.json)')
    parser.add_argument('--eventscape-root',type=Path,help='EventScape archive root in mixed training')
    dataset.add_argument('--dsec-temporal-cameras',type=Path,
                         help='Train from DSEC temporal RGB-D camera manifests (train/val/test.json)')
    parser.add_argument('--event-weight',type=float,default=0.,
                        help='Weight for target-side native-pixel event supervision during MVSEC voxel-head training')
    parser.add_argument('--event-threshold',type=float,default=.1,
                        help='Effective symmetric threshold in log APS intensity units')
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--ssim-weight',type=float,default=.2)
    parser.add_argument('--save-every',type=int,default=0,
                        help='Also keep step_XXXXXX.pt every N optimizer steps')
    parser.add_argument('--seed',type=int,default=10)
    parser.add_argument('--val-limit',type=int,default=24)
    main(parser.parse_args())
