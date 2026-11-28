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
from event_depth_boundary import event_bins, load_temporal_boundary, temporal_corrector
from event_dvlt import EventDVLT, load_scene_weights, render, voxelize
from event_field import EventField
from image_metrics import PerceptualMetrics, gray_metrics


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
    optimizer = torch.optim.AdamW(params.values(), lr=args.lr)
    encoder = EventField(args.f3_repository,args.assets/'f3.pth',4) if use_field else None
    train = [(s,i) for s in args.train for i in data[s].indices]
    val = [(s,i) for s in args.val for i in data[s].indices]
    if args.mvsec_calibration:
        val = [(seq, indices[i]) for seq in args.val
               if (indices := data[seq].indices)
               for i in np.linspace(0,len(indices)-1,min(len(indices),args.val_limit),dtype=int)]
    elif len(val)>args.val_limit:
        val = [val[i] for i in np.linspace(0,len(val)-1,args.val_limit,dtype=int)]
    if not train or not val:
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
        supervision = source.supervision(index) if temporal else stereo_supervision(source,index)
        labels = tuple(torch.as_tensor(x,dtype=torch.float32) for x in supervision)
        return voxel,field,*labels

    @torch.no_grad()
    def initialize_scale():
        # DVLT scale is ambiguous, but the target camera baseline is in metres.
        # Fit a single scalar on training frames only.
        selected = [train[i] for i in np.linspace(0,len(train)-1,min(12,len(train)),dtype=int)]
        rows = []
        for sequence,index in selected:
            voxel,field,depth,*_ = sample(sequence,index)
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

    def predict(item, empty=False, training=False, calibrated_source=False):
        voxel,field,depth,ki,poses,query,k,gray = (x[None].cuda() for x in sample(*item))
        if empty:
            voxel,field = torch.zeros_like(voxel),torch.zeros_like(field)
        calibrated = (depth,ki,poses) if args.geometry == 'calibrated' else None
        cameras = (ki,poses) if calibrated_source or (training and args.head != 'legacy') else None
        refine = (temporal_corrector(boundary,boundary_counts(*item).cuda(),empty,args.preserve_boundary_scale)
                  if boundary else None)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            cloud = model(voxel.float(),field.float(),steps=16,calibrated_geometry=calibrated,
                          source_cameras=cameras, depth_refinement=refine,
                          event_leaf_ranking=args.event_leaves, matched_leaf_budget=bool(boundary))
        coverage = None
        if args.source_camera_control:
            image,alpha = render(cloud,query,k,return_alpha=True)
            coverage = dict(alpha_mean=float(alpha.mean()),
                            alpha_covered_fraction=float((alpha>=.5).float().mean()))
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
        return image,gray/255,count,event_loss,coverage

    @lru_cache(maxsize=32)
    def overlap(seq,index):
        # Nominal angular overlap from calibration, independent of model output.
        _,_,_,ki,poses,query,k,_ = sample(seq,index)
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
            variants = [('matched',False,False),('zero',True,False)]
            if args.source_camera_control:
                variants += [('matched_calibrated',False,True),('zero_calibrated',True,True)]
            for key,empty,calibrated_source in variants:
                image,gray,count,_,coverage = predict(item,empty,calibrated_source=calibrated_source)
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

    evaluate(val,'000000')
    for step in range(1,args.steps+1):
        if mixed:
            position = (step-1) % len(train)
            if position == 0:
                order = torch.randperm(len(train),generator=train_rng).tolist()
            item = train[order[position]]
        else:
            item = train[int(torch.randint(len(train),(1,)))]
        image,gray,_,event_loss,_ = predict(item,training=True)
        l1 = (image-gray).abs().mean()
        _,ssim = gray_metrics(image,gray)
        loss = (1-args.ssim_weight)*l1 + args.ssim_weight*(1-ssim.mean())
        if event_loss is not None:
            loss = loss + args.event_weight*event_loss
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
        if mixed and step % len(train) == 0 and step < args.steps:
            evaluate(val,f'{step:06}')
    if args.steps:
        evaluate(val,f'{args.steps:06}')
    for label, (names, native) in test.items():
        evaluate([(name,index) for name in names for index in data[name].indices],label,native)


def main(args):
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
    checkpoint = torch.load(args.geometry_checkpoint, map_location='cpu', weights_only=True)
    eventscape_splits, dsec_splits = {}, {}
    mixed = bool(args.eventscape_cameras and args.dsec_temporal_cameras)
    if mixed != bool(args.eventscape_root):
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
    if not args.source_camera_control:
        config.pop('source_camera_control')
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

                data = open_gaussian_sequences(args.eventscape_root or args.root,eventscape_splits,stack)
            if args.dsec_temporal_cameras:
                from dsec_temporal_gaussian_data import DSECTemporalGaussianSequence

                for split in dsec_splits.values():
                    for sequence, cameras in split.items():
                        source = DSECTemporalGaussianSequence(args.root,sequence,cameras)
                        data[sequence] = stack.enter_context(closing(source))
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
    parser.add_argument('--head',choices=('legacy','fixed','voxel','octree'),
                        help='Defaults to fixed for predicted geometry from full-coverage distillation')
    parser.add_argument('--boundary', type=Path, help='Frozen EventScape temporal boundary checkpoint')
    parser.add_argument('--preserve-boundary-scale', action='store_true')
    parser.add_argument('--event-leaves', action='store_true', help='Use event priority for octree selection')
    parser.add_argument('--lpips',action='store_true',help='Report VGG and AlexNet LPIPS during validation')
    parser.add_argument('--lpips-weight',type=float,default=0.,help='Train with additional VGG LPIPS supervision')
    parser.add_argument('--init-head',type=Path,help='Start from a trained head with identical frozen geometry')
    parser.add_argument('--source-camera-control',action='store_true',
                        help='Compare predicted and calibrated source cameras and report alpha coverage')
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
    parser.add_argument('--seed',type=int,default=10)
    parser.add_argument('--val-limit',type=int,default=24)
    main(parser.parse_args())
