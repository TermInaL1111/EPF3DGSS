"""Cache event-only stereo inputs and calibrated RGB-teacher supervision."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from dsec_data import DSECSequence
from event_dvlt import load_rgb, voxelize
from event_field import EventField


def main(args):
    args.out.mkdir(parents=True,exist_ok=True)
    teacher=load_rgb(args.assets)
    field=EventField(args.f3_repository,args.assets/'f3.pth',4)
    features=[]
    hook=teacher.depth_decoder.register_forward_pre_hook(lambda module,inputs: features.append(inputs[0].detach()))
    manifest={'protocol':'DSEC stereo; RGB tokens mapped by calibrated sparse disparity; inputs only events',
              'window_us':args.window_us,'sequences':args.sequences,'samples':[]}
    try:
        for sequence in args.sequences:
            data=DSECSequence(args.root,sequence,args.window_us)
            try:
                indices=data.indices if not args.limit else [data.indices[i] for i in np.unique(np.linspace(0,len(data.indices)-1,min(args.limit,len(data.indices)),dtype=int))]
                for index in indices:
                    rgb,mapping=data.supervision(index)
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        teacher.forward_inference(torch.from_numpy(rgb).permute(0,3,1,2)[None].cuda().float()/255)
                    tokens=features.pop()[:,teacher.patch_start_idx:].transpose(1,2).reshape(2,768,12,16).float()
                    refs,weights=[],[]
                    for view,correspondence in enumerate(mapping):
                        grid=torch.from_numpy(correspondence['grid']).cuda()[None,None]
                        samples=F.grid_sample(tokens[view:view+1],grid,align_corners=False,padding_mode='border')[0,:,0].T
                        patch=torch.from_numpy(correspondence['patch']).cuda()
                        sums=samples.new_zeros(192,768).index_add_(0,patch,samples)
                        count=samples.new_zeros(192).index_add_(0,patch,torch.ones_like(patch,dtype=samples.dtype))
                        refs.append((sums/count.clamp_min(1)[:,None]).half().cpu())
                        # Coverage fraction per 40x40 event patch; invalid patches never contribute.
                        weights.append((count/1600).clamp(max=1).cpu())
                    packets=[data.packet(v,index) for v in (0,1)]
                    if min(p.shape[1] for p in packets)==0:
                        raise ValueError(f'Empty packet: {sequence}/{index}')
                    sample={'sequence':sequence,'frame':index,'timestamp_us':int(data.times[index]),
                            'voxel':torch.stack([voxelize(p) for p in packets]).half(),
                            'field':torch.stack([field(p).half().cpu() for p in packets]),
                            'tokens':torch.stack(refs),'coverage':torch.stack(weights)}
                    name=f'{sequence}_{index:06}.pt'
                    temp=args.out/(name+'.tmp'); torch.save(sample,temp); temp.replace(args.out/name)
                    manifest['samples'].append({'file':name,'sequence':sequence,'frame':index,
                        'event_counts':[p.shape[1] for p in packets], 'coverage':float(sample['coverage'].mean())})
                    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                    print(sequence,index,'coverage',float(sample['coverage'].mean()),flush=True)
            finally:
                data.close()
    finally:
        hook.remove()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','assets','f3_repository','out'): p.add_argument(key,type=Path)
    p.add_argument('--sequences',nargs='+',required=True)
    p.add_argument('--limit',type=int,default=0,help='0 processes all eligible disparity timestamps')
    p.add_argument('--window-us',type=int,default=20000)
    with torch.inference_mode(): main(p.parse_args())
