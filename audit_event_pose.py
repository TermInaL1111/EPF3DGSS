"""Measure how source-pose errors affect event-based depth candidate ranking."""

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from audit_event_brightness import unclipped
from audit_event_depth import DEPTH_FACTORS, ev3ds_windows, huber, log_differences, mvsec_windows, project
from event_observation import polarity_counts
from mvsec_transfer import calibration, lower_bound_after


def source_packets(root, dataset, frames):
    if dataset == 'ev3ds':
        for name, indices in frames.items():
            for index in indices:
                packets = [np.load(root/'Event'/name/f'{i:04}.npy') for i in (index, index+1)]
                yield name, index, packets
    else:
        name, indices = next(iter(frames.items()))
        mapping, _ = calibration(root/'calibration/camchain-imucam-indoor_flying.yaml')
        with h5py.File(root/f'{name}_data.hdf5') as data:
            source = data['davis/left']
            events, times = source['events'], source['image_raw_ts'][:]
            for index in indices:
                bounds = [lower_bound_after(events,t) for t in times[index-1:index+2]]
                packets = []
                for first, last in zip(bounds, bounds[1:]):
                    x, y, t, p = events[first:last].T
                    xy = mapping[y.astype(int), x.astype(int)]
                    packets.append(np.column_stack([xy, t, p]).T)
                yield name, index, packets


@torch.inference_mode()
def estimate(args, baseline):
    from dvlt.common.rays import rays_to_pose
    from event_dvlt import EventDVLT, voxelize

    torch.manual_seed(10)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if state['config']['f3']:
        raise ValueError('The CPU diagnostic uses an event-only DVLT checkpoint without F³')
    model = EventDVLT(args.assets, device='cpu').eval()
    model.load_state_dict(state['model'], strict=False)
    field = torch.zeros((1, 2, 32, 96, 128))
    rows = []
    for name, index, packets in source_packets(args.root, args.dataset, baseline['frames']):
        voxel = torch.stack([voxelize(p) for p in packets])[None]
        tokens, pos, _ = model.geometry(voxel, field, steps=16)
        rays = model.backbone._decode(tokens, 168, 224, 1, 2, pos)['rays'].float()
        poses, _ = rays_to_pose(rays, torch.ones_like(rays[..., 0]), 168, 224, 14)
        relative = torch.linalg.solve(poses[0, 0], poses[0, 1])
        rows.append(dict(sequence=name, frame=index, relative_c2w=relative.tolist()))
    print(f'Estimated {len(rows)} source poses', flush=True)
    return dict(dataset=args.dataset, checkpoint=str(args.checkpoint), step=state['step'], rows=rows)


FACTORS = DEPTH_FACTORS[[1, 2, 3]]
CONDITIONS = ('aligned', 'shifted', 'flipped')
GATE_THRESHOLDS = (1., 1.5, 2.)


def rotate(pose, axis, degrees):
    altered = pose.copy()
    altered[:3, :3] = pose[:3, :3] @ Rotation.from_euler(axis, degrees, degrees=True).as_matrix()
    return altered


def rescale(pose, factor):
    altered = pose.copy()
    altered[:3, 3] *= factor
    return altered


def test_poses(reference, predicted):
    cases = {'reference': [reference]}
    for degrees in (.025, .05, .075, .125, .25, .5, 1.):
        cases[f'yaw_{degrees}_deg'] = [rotate(reference, 'y', sign*degrees) for sign in (-1, 1)]
    cases['pitch_0.25_deg'] = [rotate(reference, 'x', sign*.25) for sign in (-1, 1)]
    cases['translation_10_pct'] = [rescale(reference, factor) for factor in (.9, 1.1)]
    cases['translation_20_pct'] = [rescale(reference, factor) for factor in (.8, 1.2)]
    r = reference.copy()
    r[:3, :3] = predicted[:3, :3]
    cases['dvlt_rotation_reference_translation'] = [r]
    t_norm = np.linalg.norm(predicted[:3, 3])
    full = (rescale(predicted, np.linalg.norm(reference[:3, 3])/t_norm) if t_norm > 1e-8 else predicted)
    cases['dvlt_pose_reference_baseline_length'] = [full]
    return cases


def gate_values(pose, rays, depth, k, distortion, coordinates):
    """Compare candidate separations to a fixed local envelope of pose uncertainty."""
    spread = np.linalg.norm(np.diff(coordinates, axis=0), axis=-1).min(0)
    noise = np.zeros_like(spread)
    nearby = [rotate(pose, axis, delta) for axis in ('x', 'y') for delta in (-.25, .25)]
    nearby += [rescale(pose, s) for s in (.95, 1.05)]
    for q in nearby:
        origin = q[:3, 3]
        direction = rays @ q[:3, :3].T
        distance = (FACTORS[:, None]*depth-origin[2])/direction[:, 2].clip(min=1e-8)
        points = origin + distance[..., None]*direction
        warped = project(points.reshape(-1, 3), k, distortion).reshape(coordinates.shape)
        displacement = np.linalg.norm(warped-coordinates, axis=-1).max(0)
        noise = np.maximum(noise, displacement)
    return spread, noise


def outcome(predicted, observations, contrast):
    costs = huber(predicted[None]-observations[:, None], contrast).mean(-1)
    winners = np.isclose(costs, costs.min(-1, keepdims=True), rtol=1e-7, atol=1e-9)
    return winners[:, 1]/winners.sum(-1)


def counters(gate):
    return dict(patches=0, reference_top1_sum=[0.]*len(CONDITIONS), top1_sum=[0.]*len(CONDITIONS),
                gates={str(limit): counters(False) for limit in GATE_THRESHOLDS} if gate else {})


def add(stats, score, oracle):
    stats['patches'] += 1
    stats['top1_sum'] = (np.asarray(stats['top1_sum']) + score).tolist()
    stats['reference_top1_sum'] = (np.asarray(stats['reference_top1_sum']) + oracle).tolist()


def measure(window, predicted, contrast):
    name, index, images, depths, packets, rays, k, distortion, reference, _ = window
    counts, later = (polarity_counts(p, images[0].shape) for p in packets)
    active = ((counts.sum(0) > 0) & unclipped(images[0]) & unclipped(images[1])
              & np.isfinite(depths[0]) & (depths[0] > 0) & (depths[1] > 0))
    y, x = np.where(active)
    nominal, mask, _, _ = log_differences(images[0], depths, rays, k, distortion, reference, y, x, False)
    y, x = y[mask], x[mask]
    nominal = nominal[1:4]
    rotation_error = float(Rotation.from_matrix(reference[:3, :3].T@predicted[:3, :3]).magnitude()*180/np.pi)
    base_t, pred_t = reference[:3, 3], predicted[:3, 3]
    t_norm = np.linalg.norm(base_t)*np.linalg.norm(pred_t)
    t_error = float(np.degrees(np.arccos(np.clip(base_t@pred_t/t_norm, -1., 1.)))) if t_norm > 1e-12 else None
    row = dict(sequence=name, frame=index, rotation_error_deg=rotation_error,
               translation_direction_error_deg=t_error, cases={})
    needs_gate = {'reference', 'yaw_0.25_deg', 'dvlt_rotation_reference_translation',
                  'dvlt_pose_reference_baseline_length'}
    for label, scenarios in test_poses(reference, predicted).items():
        stats = counters(label in needs_gate)
        row['cases'][label] = stats
        if not len(x):
            continue
        candidates = [log_differences(images[0], depths, rays, k, distortion, pose, y, x, False,
                                      check_ending_depth=False) for pose in scenarios]
        common = np.logical_and.reduce([candidate[1] for candidate in candidates])
        if not common.any():
            continue
        sy, sx = y[common], x[common]
        nominal_delta = nominal[:, common]
        signed = contrast*(counts[0]-counts[1])[sy, sx]
        next_signed = contrast*(later[0]-later[1])[sy, sx]
        observations = np.stack((signed, next_signed, -signed))
        patch = (sy//16)*((images[0].shape[1]+15)//16) + sx//16
        ids_by_patch = [np.flatnonzero(patch == block) for block in np.unique(patch)]
        blocks = [ids for ids in ids_by_patch if len(ids) >= 16]
        for pose, (changes, keep, _, coordinates) in zip(scenarios, candidates, strict=True):
            delta, uv = changes[1:4, common[keep]], coordinates[1:4, common[keep]]
            if stats['gates']:
                spread, noise = gate_values(pose, rays[sy, sx], depths[0][sy, sx], k, distortion, uv)
            for ids in blocks:
                actual = outcome(delta[:, ids], observations[:, ids], contrast)
                oracle = outcome(nominal_delta[:, ids], observations[:, ids], contrast)
                add(stats, actual, oracle)
                if stats['gates']:
                    ratio = np.median(spread[ids])/max(np.median(noise[ids]), 1e-6)
                    for limit, gate in stats['gates'].items():
                        if ratio >= float(limit):
                            add(gate, actual, oracle)
    return row


def aggregate(rows):
    def merge(items, parent_count=None):
        count = sum(item['patches'] for item in items)
        report = dict(patches=count)
        if parent_count is not None:
            report['coverage'] = count/parent_count if parent_count else 0.
        for output, field in (('top1', 'top1_sum'), ('reference_top1', 'reference_top1_sum')):
            values = np.asarray([item[field] for item in items]).sum(0)
            report[output] = dict(zip(CONDITIONS, (values/count).tolist())) if count else None
        if items[0]['gates']:
            report['gates'] = {limit: merge([item['gates'][limit] for item in items], count)
                               for limit in items[0]['gates']}
        return report

    reports = {label: merge([row['cases'][label] for row in rows]) for label in rows[0]['cases']}
    translation = [r['translation_direction_error_deg'] for r in rows if r['translation_direction_error_deg'] is not None]
    return dict(windows=len(rows), rotation_median_deg=float(np.median([r['rotation_error_deg'] for r in rows])),
                translation_direction_median_deg=float(np.median(translation)) if translation else None,
                translation_direction_windows=len(translation), cases=reports)


def score(args, baseline):
    raw = json.loads(args.poses.read_text())
    if raw['dataset'] != args.dataset:
        raise ValueError('Pose estimates and depth-ranking dataset differ')
    poses = {(row['sequence'], row['frame']): np.asarray(row['relative_c2w']) for row in raw['rows']}
    counts = {len(frames) for frames in baseline['frames'].values()}
    if len(counts) != 1:
        raise ValueError('Expected the same number of fixed diagnostic windows per sequence')
    count = counts.pop()
    windows = (ev3ds_windows(args.root, len(baseline['frames']), count) if args.dataset == 'ev3ds'
               else mvsec_windows(args.root, count))
    rows = []
    for window in windows:
        key = window[:2]
        if key not in poses:
            raise ValueError(f'Missing the DVLT pose for {key}')
        rows.append(measure(window, poses.pop(key), baseline['contrast']))
    if poses:
        raise ValueError('Unused pose estimates')
    return dict(dataset=args.dataset, step=raw['step'], contrast=baseline['contrast'],
                source_pose_file=str(args.poses), summary=aggregate(rows), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('ev3ds', 'mvsec'))
    parser.add_argument('root', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--assets', type=Path)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--checkpoint', type=Path, help='Estimate source poses from this checkpoint on CPU')
    choice.add_argument('--poses', type=Path, help='Score perturbations and the previously estimated DVLT poses')
    args = parser.parse_args()
    if args.checkpoint and not args.assets:
        parser.error('--checkpoint requires --assets')
    key = 'ev3ds_frontal' if args.dataset == 'ev3ds' else 'mvsec_frontal_c_010'
    baseline = json.loads(args.manifest.read_text())[key]
    cv2.setNumThreads(1)
    result = estimate(args, baseline) if args.checkpoint else score(args, baseline)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if 'summary' in result:
        print(json.dumps(result['summary'], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
