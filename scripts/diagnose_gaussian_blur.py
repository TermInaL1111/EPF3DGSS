"""Measure whether frozen-head blur comes from Gaussian size, colour, or geometry.

Runs temporal test windows under the calibrated source camera, then reports
Gaussian statistics and re-renders with depth / scale / opacity / colour
interventions.  The target frame is used only by the marked oracle arms.
"""

import argparse
import json
from contextlib import closing, ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F

from dsec_temporal_gaussian_data import DSECTemporalGaussianSequence
from event_dvlt import EventDVLT, load_scene_weights, render, voxelize
from event_field import EventField
from eventscape_gaussian_data import EventScapeArchive, EventScapeGaussianSequence, ARCHIVES
from image_metrics import gray_metrics


def percentiles(x, qs=(1, 10, 50, 90, 99)):
    x = x.flatten().float()
    return {f'p{q}': float(torch.quantile(x, q / 100.)) for q in qs}


def projected_radius(cloud, query, k):
    centers = cloud['centers'][0].float()
    cam = torch.linalg.inv(query)[0] @ torch.cat((centers, torch.ones_like(centers[:, :1])), 1).T
    radius = cloud['scales'][0].float().amax(-1) * k[0, 0, 0] / cam[2].clamp_min(1e-6)
    return radius


def score(prediction, target):
    psnr = -10 * (prediction - target).square().mean().clamp_min(1e-12).log10()
    return dict(psnr=float(psnr), ssim=float(gray_metrics(prediction, target)[1][0]))


def paint_target_colours(cloud, target, query, k):
    """Keep geometry, replace each Gaussian colour with the target at its projection."""
    centers = cloud['centers'][0].float()
    cam = torch.linalg.inv(query)[0] @ torch.cat((centers, torch.ones_like(centers[:, :1])), 1).T
    z = cam[2]
    uv = k[0] @ cam[:3]
    u, v = uv[0] / uv[2].clamp_min(1e-6), uv[1] / uv[2].clamp_min(1e-6)
    valid = (z > 0) & (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
    grid = torch.stack([(u + .5) / 640 * 2 - 1, (v + .5) / 480 * 2 - 1], -1)
    sampled = F.grid_sample(target[:, None], grid[None, None], align_corners=False)[0, 0, 0]
    fallback = cloud['gray'][0].float().squeeze(-1)
    painted = {**cloud, 'gray': torch.where(valid, sampled, fallback)[None, :, None],
               'opacity': cloud['opacity'] * valid.float()[None, :]}
    return painted, float(valid.float().mean())


def render_arms(cloud, target, query, k):
    arms = {}
    for name, factor in (('scale_half', .5), ('scale_quarter', .25), ('scale_double', 2.)):
        arms[name] = score(render({**cloud, 'scales': cloud['scales'] * factor}, query, k), target)
    for name, factor in (('opacity_half', .5), ('opacity_double', 2.)):
        arms[name] = score(render({**cloud, 'opacity': (cloud['opacity'] * factor).clamp(0, 1)},
                                  query, k), target)
    painted, visible = paint_target_colours(cloud, target, query, k)
    arms['target_colour_oracle'] = score(render(painted, query, k), target)
    arms['target_colour_oracle']['visible_fraction'] = visible
    smaller, _ = paint_target_colours(
        {**cloud, 'scales': cloud['scales'] * .5, 'opacity': (cloud['opacity'] * 2).clamp(0, 1)},
        target, query, k)
    arms['target_colour_oracle_scale_half'] = score(render(smaller, query, k), target)
    return arms


@torch.inference_mode()
def evaluate(model, encoder, steps, source, sequence, indices):
    reports = []
    for index in indices:
        packets = [source.packet(v, index) for v in (0, 1)]
        voxel = torch.stack([voxelize(p) for p in packets]).half()[None].cuda().float()
        field = (torch.stack([encoder(p).half().cpu() for p in packets])[None].cuda().float()
                 if encoder else torch.zeros((1, 2, 32, 96, 128), device='cuda'))
        depth, ki, poses, query, k, gray = source.supervision(index)
        depth, ki, poses, query, k = (torch.as_tensor(x)[None].cuda().float()
                                      for x in (depth, ki, poses, query, k))
        target = torch.as_tensor(gray)[None].cuda().float() / 255
        with torch.autocast('cuda', dtype=torch.bfloat16):
            cloud = model(voxel, field, steps=steps,
                          source_cameras=(ki, poses), return_geometry=True)
            predictor = model(voxel, field, steps=steps, return_geometry=True)
            cloud_ref = model(voxel, field, steps=steps, source_cameras=(ki, poses),
                              reference_metric_depth=depth, return_geometry=True)

        image, alpha = render(cloud, query, k, return_alpha=True)
        predicted, _ = render(predictor, query, k, return_alpha=True)
        measured = render(cloud_ref, query, k)
        painted_ref, _ = paint_target_colours(cloud_ref, target, query, k)
        report = dict(sequence=sequence, window=int(index),
                      gaussians=int(cloud['centers'].shape[1]),
                      base=score(image, target),
                      predicted_camera=score(predicted, target),
                      reference_depth=score(measured, target),
                      reference_depth_colour_oracle=score(render(painted_ref, query, k), target),
                      scale_m=percentiles(cloud['scales'][0]),
                      radius_px=percentiles(projected_radius(cloud, query, k)),
                      opacity=percentiles(cloud['opacity'][0]),
                      gray=percentiles(cloud['gray'][0]),
                      alpha=dict(mean=float(alpha.mean()),
                                 covered=float((alpha >= .5).float().mean())),
                      arms=render_arms(cloud, target, query, k))
        reports.append(report)
        print(json.dumps(report), flush=True)
    return reports


@torch.inference_mode()
def main(args):
    torch.manual_seed(args.seed)
    model = EventDVLT(args.assets).cuda().eval()
    payload = torch.load(args.init_head, map_location='cpu', weights_only=True)
    steps, use_f3 = load_scene_weights(model, payload)
    if getattr(args, 'max_scale_voxels', None) and hasattr(model.gs_head, 'max_scale_voxels'):
        model.gs_head.max_scale_voxels = args.max_scale_voxels
    if getattr(args, 'min_points', None) and hasattr(model.gs_head, 'min_points'):
        model.gs_head.min_points = args.min_points
    if getattr(args, 'min_views', None) and hasattr(model.gs_head, 'min_views'):
        model.gs_head.min_views = args.min_views
    model.eval()
    encoder = EventField(args.f3_repository, args.assets / 'f3.pth', 4) if use_f3 else None

    reports = []
    with ExitStack() as stack:
        if args.dataset == 'eventscape':
            splits = json.loads((args.cameras / f'{args.split}.json').read_text())['sequences']
            archive = stack.enter_context(
                closing(EventScapeArchive(args.eventscape_root / ARCHIVES[args.split])))
            names = args.sequences or [args.sequence]
            for name in names:
                source = EventScapeGaussianSequence(archive, name, splits[name],
                                                    target_offset=args.target_offset)
                selection = source.indices if args.windows == 'all' else [source.indices[i] for i in args.windows]
                reports += evaluate(model, encoder, steps, source, name, selection)
        else:
            record = json.loads((args.dsec_cameras / f'{args.split}.json').read_text())
            cameras = record['sequences'][args.sequence]['cameras']
            source = stack.enter_context(
                closing(DSECTemporalGaussianSequence(args.dsec_root, args.sequence, cameras,
                                                      target_offset=args.target_offset)))
            selection = source.indices if args.windows == 'all' else [source.indices[i] for i in args.windows]
            reports = evaluate(model, encoder, steps, source, args.sequence, selection)
    args.out.write_text(json.dumps(reports, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assets', type=Path)
    parser.add_argument('f3_repository', type=Path)
    parser.add_argument('--init-head', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dataset', choices=('eventscape', 'dsec'), default='eventscape')
    parser.add_argument('--eventscape-root', type=Path)
    parser.add_argument('--cameras', type=Path)
    parser.add_argument('--dsec-root', type=Path)
    parser.add_argument('--dsec-cameras', type=Path)
    parser.add_argument('--sequence')
    parser.add_argument('--sequences', nargs='+')
    parser.add_argument('--split', default='test')
    parser.add_argument('--target-offset', type=int, choices=(2,4,6,8), default=2)
    parser.add_argument('--max-scale-voxels', type=float)
    parser.add_argument('--min-points', type=int)
    parser.add_argument('--min-views', type=int)
    parser.add_argument('--windows', default='0',
                        help="Comma-separated window ordinals in the sequence manifest, or 'all'")
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()
    if args.windows != 'all':
        args.windows = [int(x) for x in args.windows.split(',')]
    if args.dataset == 'dsec' and not args.sequence:
        parser.error('DSEC needs --sequence')
    if not (args.sequence or args.sequences):
        parser.error('Specify --sequence or --sequences')
    main(args)
