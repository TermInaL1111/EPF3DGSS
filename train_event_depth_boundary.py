"""Train matched depth-boundary ablations on EventScape with frozen event DVLT."""

import argparse
from contextlib import closing
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from event_depth_boundary import EventDepthBoundary, adjacent, depth_loss, event_bins
from event_dvlt import EventDVLT, voxelize
from eventscape_data import EventScapeArchive, EventScapeSequence


ARMS = ('geometry', 'accumulated', 'fixed_edge', 'temporal')


def fixed_edge(counts):
    # E2EGS Eqs. 1–2 with overlapping stride-one patches and a continuous confidence map.
    signed = counts[:, 1] - counts[:, 0]
    kernel = torch.exp(-.5 * torch.arange(-2, 3, device=counts.device).float().square())
    kernel = kernel / kernel.sum()
    blurred = F.conv2d(signed.flatten(0, 1)[:, None], (kernel[:, None] * kernel)[None, None],
                       padding=2).reshape_as(signed)
    delta = blurred.diff(dim=1).abs()
    average = F.avg_pool2d(delta, 5, stride=1, padding=2, count_include_pad=False)
    second = F.avg_pool2d(delta.square(), 5, stride=1, padding=2, count_include_pad=False)
    variance = (second - average.square()).clamp_min(0).amax(dim=1, keepdim=True)
    confidence = F.max_pool2d(variance, 5, stride=1, padding=2)
    return confidence / confidence.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def evidence(counts):
    accumulated = counts.sum(dim=2, keepdim=True).expand_as(counts)
    return dict(geometry=torch.zeros_like(counts).flatten(1, 2),
                accumulated=accumulated.flatten(1, 2).log1p(),
                fixed_edge=fixed_edge(counts).expand(-1, 10, -1, -1),
                temporal=counts.flatten(1, 2).log1p())


@torch.no_grad()
def sample(model, sequence, index, device):
    packets = [sequence.packet(i) for i in (index-1, index)]
    scaled = [packet.astype(np.float64) * np.array([640/512, 480/256, 1, 1])[:, None]
              for packet in packets]
    voxel = torch.stack([voxelize(packet) for packet in scaled])[None].to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        tokens, pos, _ = model.geometry(voxel, voxel.new_zeros(1, 2, 32, 96, 128), steps=16)
        geometry = model.backbone._decode(tokens, 168, 224, 1, 2, pos)
    tokens = tokens[1:, model.backbone.patch_start_idx:].transpose(1, 2).reshape(1, 768, 12, 16).float()
    depth = geometry['depth'][0, 1, ..., 0].float()[None, None]
    confidence = geometry['depth_conf'][0, 1].float()[None, None]
    label = torch.from_numpy(sequence.depth(index))[None, None].to(device)
    inputs = evidence(event_bins(packets[-1], label.shape[-2:])[None].to(device))
    return tokens, depth, confidence, inputs, label


def score(pred, label):
    valid = torch.isfinite(label) & (label >= .1) & (label <= 200)
    values = pred.exp()[valid]
    target = label[valid]
    values = values * (target / values).median()
    hit, predicted, truth = 0, 0, 0
    near_reference, near_prediction = 0, 0
    masks = (valid[..., :, 1:] & valid[..., :, :-1], valid[..., 1:, :] & valid[..., :-1, :])
    band = torch.zeros_like(valid)
    target_log = torch.where(valid, label, 1).log()
    for axis, (a, b, mask) in enumerate(zip(adjacent(pred), adjacent(target_log), masks, strict=True)):
        estimated = (a.abs() >= math.log1p(.10)) & mask
        reference = (b.abs() >= math.log1p(.10)) & mask
        hit += int((estimated & reference).sum())
        predicted += int(estimated.sum())
        truth += int(reference.sum())
        near_reference += int((estimated & F.max_pool2d(reference.float(), 3, stride=1, padding=1).bool()).sum())
        near_prediction += int((reference & F.max_pool2d(estimated.float(), 3, stride=1, padding=1).bool()).sum())
        if axis == 0:
            band[..., :, :-1] |= reference
            band[..., :, 1:] |= reference
        else:
            band[..., :-1, :] |= reference
            band[..., 1:, :] |= reference
    band = F.max_pool2d(band.float(), 5, stride=1, padding=2).bool()[valid]
    err = (values-target).abs()/target
    return dict(absrel=float(err.mean()), delta1=float((torch.maximum(values/target,target/values)<1.25).float().mean()),
                edge_hits=hit, edge_predicted=predicted, edge_reference=truth,
                edge_predicted_near_reference=near_reference, edge_reference_near_predicted=near_prediction,
                boundary_absrel=float(err[band].mean()) if band.any() else None,
                interior_absrel=float(err[~band].mean()) if (~band).any() else None)


def sequences(archive):
    return [EventScapeSequence(archive, name) for name in archive.sequences]


def summarize(rows):
    result = {}
    for arm in ('dvlt', *ARMS):
        stats = [row[arm] for row in rows]
        entry = {}
        for key in ('absrel','delta1','boundary_absrel','interior_absrel'):
            values = [r[key] for r in stats if r[key] is not None]
            entry[key] = float(np.mean(values)) if values else None
        support = sum(s['edge_reference'] + s['edge_predicted'] for s in stats)
        entry['boundary_f1_exact'] = 2 * sum(s['edge_hits'] for s in stats) / support if support else None
        total_predicted = sum(s['edge_predicted'] for s in stats)
        total_reference = sum(s['edge_reference'] for s in stats)
        precision = sum(s['edge_predicted_near_reference'] for s in stats) / max(1, total_predicted)
        recall = sum(s['edge_reference_near_predicted'] for s in stats) / max(1, total_reference)
        entry['boundary_f1_1px'] = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        result[arm] = entry
    return result


def main(args):
    torch.manual_seed(10)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    if checkpoint['config']['f3']:
        raise ValueError('Use a no-F³ EventScape geometry checkpoint')
    model = EventDVLT(args.assets, device=device).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'], strict=False)
    heads = {key: EventDepthBoundary().to(device) for key in ARMS}
    starting = heads['geometry'].state_dict()
    for head in heads.values():
        head.load_state_dict(starting)
    optimizers = {key: torch.optim.AdamW(head.parameters(), lr=3e-4) for key, head in heads.items()}

    with (closing(EventScapeArchive(args.root/'Town01-03_train.zip')) as training,
          closing(EventScapeArchive(args.root/'Town05_val.zip')) as validation):
        train, val = sequences(training), sequences(validation)
        if set(training.sequences) & set(validation.sequences):
            raise ValueError('Training and validation must be disjoint')
        starts = [(source, index) for source in train for index in source.ends[1:]]
        if args.steps > len(starts):
            raise ValueError(f'{args.steps} training updates requested, but only {len(starts)} unique starts exist')
        rng = np.random.default_rng(10)
        for step, row in enumerate(rng.permutation(len(starts))[:args.steps], 1):
            tokens, depth, conf, inputs, label = sample(model, *starts[row], device)
            losses = {}
            for key, head in heads.items():
                opt = optimizers[key]
                opt.zero_grad(set_to_none=True)
                loss = depth_loss(*head(tokens, depth, conf, inputs[key]), label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                losses[key] = float(loss.detach())
            if step == 1 or step % 100 == 0:
                print(step, losses, flush=True)
            if step % 1000 == 0 or step == args.steps:
                tmp = args.out/'latest.tmp'
                torch.save(dict(step=step, model={key: head.state_dict() for key, head in heads.items()}),tmp)
                tmp.replace(args.out/'latest.pt')
        rows = []
        selected = np.linspace(0, len(val)-1, min(args.val_windows, len(val)), dtype=int)
        with torch.no_grad():
            for row in selected:
                sequence = val[row]
                index = sequence.ends[len(sequence.ends)//2]
                tokens, depth, conf, inputs, label = sample(model, sequence, index, device)
                base = F.interpolate(depth, label.shape[-2:], mode='bilinear', align_corners=False).log()
                scores = {key: score(head(tokens, depth, conf, inputs[key])[0],label) for key,head in heads.items()}
                rows.append(dict(sequence=sequence.sequence, index=index, dvlt=score(base,label), **scores))
        summary = summarize(rows)
        relative = {}
        for arm in ARMS:
            relative[arm] = {}
            for metric, baseline in summary['geometry'].items():
                value = summary[arm][metric]
                sign = -1 if metric.endswith('absrel') else 1
                relative[arm][metric] = (100 * sign * (value - baseline) / baseline
                                         if baseline and value is not None else None)
        result = dict(checkpoint=str(args.checkpoint), geometry_step=checkpoint['step'],
                      boundary_steps=step, summary=summary,
                      relative_improvement_vs_geometry_pct=relative, rows=rows)
        (args.out/'val.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(result['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('root','assets','checkpoint','out'):
        parser.add_argument(field,type=Path)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--val-windows', type=int, default=96)
    parser.add_argument('--device', choices=('cpu','cuda'),default='cuda')
    args = parser.parse_args()
    if args.steps < 1 or args.val_windows < 1:
        parser.error('Training and validation budgets must be positive')
    args.out.mkdir(parents=True)
    main(args)
