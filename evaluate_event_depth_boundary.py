"""Evaluate frozen depth-boundary heads on held-out single-camera event sequences."""

import argparse
from contextlib import closing
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import yaml
from torch.nn import functional as F

from audit_voxel_boundaries import mvsec_packet, spread
from dsec_data import DSECSequence
from event_depth_boundary import EventDepthBoundary
from event_dvlt import EventDVLT
from eventscape_data import EventScapeArchive, EventScapeSequence
from train_event_depth_boundary import ARMS, boundary_inputs, relative_improvement, score, summarize


def windows(args):
    if args.dataset == 'dsec':
        split = json.loads(args.split.read_text())
        for name in split['val'] + split['test']:
            with closing(DSECSequence(args.root, name)) as source:
                paths = source.root / 'disparity/event'
                frames = [i for i in source.indices[1:] if (paths / f'{i:06}.png').is_file()]
                previous = dict(zip(source.indices[1:], source.indices))
                for i in spread(frames, args.frames):
                    disparity = cv2.imread(str(paths / f'{i:06}.png'), cv2.IMREAD_UNCHANGED)
                    if disparity is None or disparity.dtype != np.uint16:
                        raise ValueError(f'Invalid DSEC disparity at {name}/{i}')
                    q = np.asarray(source.calib['disparity_to_depth']['cams_03'])
                    depth = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 256, q)[..., 2]
                    yield name, i, [source.packet(0, j) for j in (previous[i], i)], np.where(disparity > 0, depth, 0)
    elif args.dataset == 'mvsec':
        camera = yaml.safe_load((args.root / 'calibration/camchain-imucam-indoor_flying.yaml').read_text())['cam0']
        k = np.asarray(camera['projection_matrix'])[:, :3]
        width, height = camera['resolution']
        native_scale = np.array([width/640, height/480])[:, None]
        for name in ('indoor_flying1', 'indoor_flying2', 'indoor_flying3'):
            with (h5py.File(args.root / f'{name}_data.hdf5') as data,
                  h5py.File(args.root / f'{name}_gt.hdf5') as gt):
                depths = gt['davis/left/depth_image_rect']
                times = gt['davis/left/depth_image_rect_ts'][:]
                events = data['davis/left/events']
                for i in spread(range(1, len(depths)), args.frames):
                    packets = [mvsec_packet(events, times[j], camera, k) for j in (i-1, i)]
                    for packet in packets:
                        packet[:2] = (packet[:2] + .5) * native_scale - .5
                    yield name, i, packets, depths[i]
    else:
        with closing(EventScapeArchive(args.root / 'Town05_test.zip')) as archive:
            for name in archive.sequences:
                source = EventScapeSequence(archive, name)
                for i in spread(source.ends[1:], args.frames):
                    yield name, i, [source.packet(j) for j in (i-1, i)], source.depth(i)


@torch.inference_mode()
def evaluate(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.geometry, map_location='cpu', weights_only=True)
    if checkpoint['config']['f3']:
        raise ValueError('Expected the no-F³ backbone used to train the boundary heads')
    model = EventDVLT(args.assets, device=device).to(device).eval()
    model.load_state_dict(checkpoint['model'], strict=False)
    heads = {}
    state = torch.load(args.boundary, map_location='cpu', weights_only=True)
    for key in ARMS:
        heads[key] = EventDepthBoundary().to(device).eval()
        heads[key].load_state_dict(state['model'][key])
    trained = set(checkpoint['config']['train']) | set(checkpoint['config']['val'])
    cap = 200. if args.dataset == 'eventscape' else 50.
    rows = []
    for name, index, packets, target in windows(args):
        if name in trained:
            raise ValueError(f'Evaluation sequence overlaps backbone training or validation: {name}')
        target = torch.as_tensor(target, device=device)[None, None]
        if not ((target >= .1) & (target <= cap)).any():
            continue
        tokens, depth, conf, evidence = boundary_inputs(model, packets, target.shape[-2:], device)
        baseline = F.interpolate(depth, target.shape[-2:], mode='bilinear', align_corners=False).log()
        scores = {key: score(head(tokens, depth, conf, evidence[key])[0], target, cap)
                  for key, head in heads.items()}
        rows.append(dict(sequence=name, index=index, dvlt=score(baseline, target, cap), **scores))
        if args.limit and len(rows) >= args.limit:
            break
    if not rows:
        raise ValueError('No valid depth windows')
    summary = summarize(rows)
    return dict(dataset=args.dataset, cap_m=cap, geometry_checkpoint=str(args.geometry),
                geometry_step=checkpoint['step'], boundary_checkpoint=str(args.boundary),
                boundary_steps=state['step'], summary=summary,
                relative_improvement_vs_geometry_pct=relative_improvement(summary), rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('dsec', 'mvsec', 'eventscape'))
    for key in ('root', 'assets', 'geometry', 'boundary', 'out'):
        parser.add_argument(key, type=Path)
    parser.add_argument('--split', type=Path, default=Path('scripts/dsec_full_split.json'))
    parser.add_argument('--frames', type=int, default=24, help='Sampled windows per sequence')
    parser.add_argument('--limit', type=int, help='Total valid window cap for integration checks')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.frames < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Window limits must be positive')
    report = evaluate(args)
    tmp = args.out.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    tmp.replace(args.out)
    print(args.dataset, len(report['rows']), report['summary'], flush=True)
