"""Train a grayscale Gaussian head with a frozen geometry backbone."""
import argparse
from functools import lru_cache
import fcntl
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dsec_data import DSECSequence, resize_intrinsics, stereo_supervision
from event_dvlt import EventDVLT, render, voxelize
from event_field import EventField
from image_metrics import PerceptualMetrics, gray_metrics


def run(args, data, checkpoint):
    model = EventDVLT(args.assets).cuda().eval()
    weights = checkpoint['model']
    geometry_names = {n for n,p in model.named_parameters() if p.requires_grad
                      and not n.startswith(('gs_head.', 'field_features.', 'geometry_features.'))
                      and n != 'log_metric_scale'}
    use_field = bool(checkpoint['config']['f3'])
    expected = geometry_names if use_field else {n for n in geometry_names if not n.startswith('event_adapter.')}
    if set(weights) != expected:
        raise ValueError('Unexpected parameters in DSEC geometry checkpoint')
    model.load_state_dict(weights, strict=False)
    model.configure_head(args.head)
    for name, parameter in model.named_parameters():
        trainable = name.startswith(('gs_head.', 'field_features.', 'geometry_features.'))
        parameter.requires_grad_(trainable or (name == 'log_metric_scale' and args.geometry == 'predicted'
                                                and args.head == 'legacy'))
    params = {n:p for n,p in model.named_parameters() if p.requires_grad}
    perceptual = PerceptualMetrics("cuda") if args.lpips else None
    optimizer = torch.optim.AdamW(params.values(), lr=3e-4)
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
        raise ValueError('Both splits require stereo frames with disparity timestamps')

    @lru_cache(maxsize=32)
    def sample(seq,index):
        source = data[seq]
        packets = [source.packet(v,index) for v in (0,1)]
        voxel = torch.stack([voxelize(p) for p in packets]).half()
        field = (torch.stack([encoder(p).half().cpu() for p in packets]) if encoder
                 else torch.zeros((2,32,96,128),dtype=torch.float16))
        supervision = source.supervision(index) if args.mvsec_calibration else stereo_supervision(source,index)
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
            if not valid.any() and args.mvsec_calibration:
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

    if args.geometry == 'predicted':
        initialize_scale()

    def predict(item, empty=False, training=False):
        voxel,field,depth,ki,poses,query,k,gray = (x[None].cuda() for x in sample(*item))
        if empty:
            voxel,field = torch.zeros_like(voxel),torch.zeros_like(field)
        calibrated = (depth,ki,poses) if args.geometry == 'calibrated' else None
        cameras = (ki,poses) if training and args.head != 'legacy' else None
        with torch.autocast('cuda',dtype=torch.bfloat16):
            cloud = model(voxel.float(),field.float(),steps=16,calibrated_geometry=calibrated,
                          source_cameras=cameras)
        return render(cloud,query,k),gray/255,cloud.get('gaussian_count')

    @lru_cache(maxsize=None)
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
    def evaluate(step):
        rows = []
        for index,item in enumerate(val):
            row = dict(sequence=item[0], frame=item[1])
            preview = []
            mask = overlap(*item)
            row["overlap_fraction"] = float(mask.float().mean())
            for key,empty in [('matched',False),('zero',True)]:
                image,gray,count = predict(item,empty)
                if count is not None:
                    row[key+'_gaussian_count'] = int(count[0])
                psnr,ssim = gray_metrics(image,gray)
                row[key+'_psnr'],row[key+'_ssim'] = float(psnr[0]),float(ssim[0])
                row[key+"_ssim_evggs"] = float(gray_metrics(image,gray,padded=True)[1][0])
                if perceptual:
                    row.update({f"{key}_{metric}":value for metric,value in perceptual(image,gray).items()})
                mse_overlap = (image-gray)[:,mask].square().mean()
                row[key+'_overlap_psnr'] = float(-10*mse_overlap.clamp_min(1e-12).log10())
                if index == 0:
                    if not preview:
                        preview.append(gray[0])
                    preview.append(image[0])
            if preview:
                comparison = torch.cat(preview,dim=1).clamp(0,1).mul(255).round().byte().cpu().numpy()
                if not cv2.imwrite(str(args.out/f'preview_{step:06}.png'),comparison):
                    raise OSError('Could not save the validation preview')
            rows.append(row)
        (args.out/f'val_{step:06}.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        print('validation',step,{k:sum(r[k] for r in rows)/len(rows)
                                  for k in ('matched_psnr','zero_psnr','matched_ssim','zero_ssim')},flush=True)

    evaluate(0)
    if args.steps == 0:
        return
    for step in range(1,args.steps+1):
        item = train[int(torch.randint(len(train),(1,)))]
        image,gray,_ = predict(item,training=True)
        l1 = (image-gray).abs().mean()
        _,ssim = gray_metrics(image,gray)
        loss = (1-args.ssim_weight)*l1 + args.ssim_weight*(1-ssim.mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params.values(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step==1 or step%20==0:
            print('step',step,'loss',float(loss.detach()),'l1',float(l1.detach()),
                  'ssim',float(ssim.detach().mean()),'grad',float(norm),flush=True)
        if step==args.steps or step%100==0:
            tmp = args.out/'latest.pt.tmp'
            state = weights | {n:p.detach().cpu() for n,p in params.items()}
            if args.head != 'legacy':
                state['log_metric_scale'] = model.log_metric_scale.detach().cpu()
            torch.save({'model':state,'step':step,'geometry':args.geometry,'f3':use_field,
                        'head':args.head},tmp)
            tmp.replace(args.out/'latest.pt')
    evaluate(args.steps)


def main(args):
    if not 0 <= args.ssim_weight <= 1:
        raise ValueError('SSIM weight must be between 0 and 1')
    if args.steps < 0 or args.val_limit < 1:
        raise ValueError("Steps must be nonnegative and validation limit must be positive")
    checkpoint = torch.load(args.geometry_checkpoint, map_location='cpu', weights_only=True)
    if args.head is None:
        args.head = ('fixed' if args.geometry == 'predicted' and checkpoint['config'].get('cover_train')
                     else 'legacy')
    if args.head != 'legacy' and args.geometry != 'predicted':
        raise ValueError('Fixed and voxel heads use student depth')
    if args.mvsec_calibration and args.head == 'legacy':
        raise ValueError('MVSEC temporal training supports fixed and voxel heads')
    if set(args.train)&set(args.val):
        raise ValueError('Specify nonoverlapping training and validation sequences')
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    if args.mvsec_calibration is None:
        config.pop('mvsec_calibration')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    lock_path = args.out.with_name(args.out.name+'.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.out.exists():
            recorded = json.loads((args.out/'config.json').read_text())
            validation = args.out/f'val_{args.steps:06}.json'
            checkpoint = args.out/'latest.pt'
            if (dict({'head':'legacy'}, **recorded) == config and validation.is_file() and
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
        data = {}
        try:
            for sequence in args.train+args.val:
                if args.mvsec_calibration:
                    from mvsec_gaussian_data import MVSECTemporalGaussianSequence

                    data[sequence] = MVSECTemporalGaussianSequence(args.root,sequence,args.mvsec_calibration)
                else:
                    data[sequence] = DSECSequence(args.root,sequence)
            run(args,data,checkpoint)
        finally:
            for source in data.values():
                source.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root','assets','f3_repository','geometry_checkpoint','out'):
        parser.add_argument(key,type=Path)
    parser.add_argument('--train',nargs='+',required=True)
    parser.add_argument('--val',nargs='+',required=True)
    parser.add_argument('--geometry',choices=('calibrated','predicted'),required=True)
    parser.add_argument('--head',choices=('legacy','fixed','voxel'),
                        help='Defaults to fixed for predicted geometry from full-coverage distillation')
    parser.add_argument('--lpips',action='store_true',help='Report VGG and AlexNet LPIPS during validation')
    parser.add_argument('--mvsec-calibration',type=Path,
                        help='Train from temporal windows of the left MVSEC camera')
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--ssim-weight',type=float,default=.2)
    parser.add_argument('--seed',type=int,default=10)
    parser.add_argument('--val-limit',type=int,default=24)
    main(parser.parse_args())
