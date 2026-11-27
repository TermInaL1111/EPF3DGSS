"""Score EventScape depth on consecutive, nonoverlapping Town05 test clips."""

import argparse
import json
from contextlib import closing
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from event_dvlt import EventDVLT, load_rgb, voxelize
from event_field import EventField
from eventscape_data import EventScapeArchive, EventScapeSequence


def depth_scores(estimate, reference):
    scale = (reference / estimate).median()
    depth = estimate * scale
    ratio = torch.maximum(depth / reference, reference / depth)
    return dict(absrel=float(((depth - reference).abs() / reference).mean()),
                delta1=float((ratio < 1.25).float().mean()), scale=float(scale))


@torch.inference_mode()
def evaluate(args, archive, checkpoint):
    if set(archive.sequences) & set(checkpoint['config']['train']):
        raise ValueError('EventScape test sequences overlap training')
    student = EventDVLT(args.assets).cuda().eval()
    use_field = bool(checkpoint['config']['f3'])
    expected = {name for name, p in student.named_parameters() if p.requires_grad
                and not name.startswith(('gs_head.', 'field_features.', 'geometry_features.'))
                and name != 'log_metric_scale' and (use_field or not name.startswith('event_adapter.'))}
    if set(checkpoint['model']) != expected:
        raise ValueError('Unexpected parameters in the geometry checkpoint')
    student.load_state_dict(checkpoint['model'], strict=False)
    teacher = load_rgb(args.assets)
    teacher.inference_steps = 16
    encoder = EventField(args.f3, args.assets/'f3.pth', 4) if use_field else None
    rows = []
    for name in archive.sequences:
        source = EventScapeSequence(archive, name)
        for start in range(0, len(source.ends) - 17, 18):
            indices = range(start, start + 18)
            packets, frames = zip(*(source.frame(i) for i in indices))
            rgb = torch.as_tensor(np.stack(frames), device='cuda').permute(0, 3, 1, 2)[None].float()/255
            voxel = torch.stack([voxelize(packet) for packet in packets])[None].cuda()
            field = (torch.stack([encoder(packet) for packet in packets])[None] if encoder
                     else voxel.new_zeros((1, 18, 32, 96, 128)))
            reference = torch.as_tensor(np.stack([source.depth(i) for i in indices]), device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tokens, pos, _ = student.geometry(voxel, field, steps=16)
                student_depth = student.backbone._decode(tokens, 168, 224, 1, 18, pos)['depth']
                teacher_depth = teacher.forward_inference(rgb)['depth']
            estimates = [F.interpolate(depth[0].float().permute(0, 3, 1, 2), reference.shape[-2:],
                                       mode='bilinear', align_corners=False)[:, 0]
                         for depth in (student_depth, teacher_depth)]
            labelled = (reference > 0) & (reference <= 200)
            valid = labelled.clone()
            for estimate in estimates:
                valid &= torch.isfinite(estimate) & (estimate > 0)
            if not valid.any():
                raise ValueError(f'No common labelled pixels at {name}/{start}')
            rows.append(dict(sequence=name, start=start, labelled_pixels=int(labelled.sum()),
                             scored_pixels=int(valid.sum()), student=depth_scores(estimates[0][valid], reference[valid]),
                             teacher=depth_scores(estimates[1][valid], reference[valid])))
            if args.limit is not None and len(rows) >= args.limit:
                return rows
        print(name, 'windows scored', len(rows), flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('archive', 'assets', 'checkpoint', 'out'):
        parser.add_argument(key, type=Path)
    parser.add_argument('--f3', type=Path)
    parser.add_argument('--limit', type=int, help='Limit the number of windows for a smoke test')
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    with closing(EventScapeArchive(args.archive)) as archive:
        rows = evaluate(args, archive, checkpoint)
    if not rows:
        raise ValueError('No complete 18-view clips')
    summary = {arm: {key: sum(row[arm][key] for row in rows)/len(rows) for key in ('absrel', 'delta1')}
               for arm in ('student', 'teacher')}
    report = dict(checkpoint=str(args.checkpoint), step=checkpoint['step'], windows=len(rows),
                  sequences=len({row['sequence'] for row in rows}),
                  max_depth_m=200, clip_views=18, recurrent_steps=16,
                  depth_alignment='median ratio per clip on jointly valid native pixels', summary=summary, rows=rows)
    temporary = args.out.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.out)
    print(summary, flush=True)
