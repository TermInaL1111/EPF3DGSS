"""Evaluate the published EPF3DGSS checkpoint on the full EV3DS validation split."""
import argparse
import json
from pathlib import Path

import torch

from event_dvlt import EventDVLT, load_scene_weights
from event_field import source_windows
from image_metrics import PerceptualMetrics
from train_event_dvlt import evaluate


def main(args):
    model = EventDVLT(args.assets).cuda().eval()
    checkpoint = torch.load(args.weights,map_location='cpu',weights_only=True)
    steps,use_f3 = load_scene_weights(model,checkpoint)
    if steps!=12 or not use_f3:
        raise ValueError('Use the published EV3DS reference checkpoint')
    fields = torch.load(args.fields,map_location='cpu',weights_only=True)
    keys = source_windows(args.root,'val',True)
    metrics = PerceptualMetrics('cuda')
    rows = []
    for offset in range(0,len(keys),8):
        rows.extend(evaluate(model,args.root,fields,keys[offset:offset+8],metrics,crop_evggs=args.crop_evggs))
        print(f'{len(rows)}/{len(keys)} validation windows',flush=True)
    temporary = args.out.with_name(args.out.name+'.tmp')
    temporary.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.out)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','assets','weights','fields','out'):
        p.add_argument(name,type=Path)
    p.add_argument('--crop-evggs', action='store_true', help='Score the same central crop as EvGGS')
    main(p.parse_args())
