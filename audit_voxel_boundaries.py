"""Measure when native-depth discontinuities could share a 3D voxel."""

import argparse
from contextlib import closing
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml

from dsec_data import DSECSequence, camera_matrix
from eventscape_data import EventScapeArchive, EventScapeSequence


VOXEL_SIDES = (.05, .10, .20, .50)
DEPTH_JUMPS = (.05, .10)


def spread(items, count):
    if count == 1 and len(items):
        return [items[len(items) // 2]]
    return [items[i] for i in np.linspace(0, len(items) - 1, min(count, len(items)), dtype=int)]


def source_frames(args):
    if args.dataset == 'eventscape':
        with closing(EventScapeArchive(args.root)) as archive:
            k = camera_matrix([256., 256., 256., 128.])
            for town in ('Town01', 'Town02', 'Town03'):
                names = [n for n in archive.sequences if n.startswith(town + '/')]
                if not names:
                    raise ValueError(f'Use a training archive containing {town}')
                for name in spread(names, 4):
                    source = EventScapeSequence(archive, name)
                    for frame in spread(source.ends[1:], args.frames_per_sequence):
                        packets = None
                        if args.checkpoint:
                            packets = [source.frame(i)[0] for i in (frame - 1, frame)]
                        yield name, frame, source.depth(frame), k, 200., packets
    elif args.dataset == 'dsec':
        train = json.loads(args.split.read_text())['train']
        for name in spread(train, 8):
            with closing(DSECSequence(args.root, name)) as source:
                k = camera_matrix(source.calib['intrinsics']['camRect0']['camera_matrix'])
                q = np.asarray(source.calib['disparity_to_depth']['cams_03'])
                candidates = [frame for frame in source.indices
                              if (source.root / f'disparity/event/{frame:06}.png').is_file()]
                previous = dict(zip(source.indices[1:], source.indices))
                if args.checkpoint:
                    candidates = [i for i in candidates if i in previous]
                for frame in spread(candidates, args.frames_per_sequence):
                    disparity = cv2.imread(str(source.root / f'disparity/event/{frame:06}.png'),
                                           cv2.IMREAD_UNCHANGED)
                    if disparity is None or disparity.dtype != np.uint16:
                        raise ValueError(f'Invalid event disparity for {name}/{frame}')
                    depth = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 256, q)[..., 2]
                    packets = None
                    if args.checkpoint:
                        packets = [source.packet(0, i) for i in (previous[frame], frame)]
                    yield name, frame, np.where(disparity > 0, depth, 0), k, 50., packets
    else:
        name = 'indoor_flying4'
        calib = args.root / 'calibration/camchain-imucam-indoor_flying.yaml'
        cam = yaml.safe_load(calib.read_text())['cam0']
        k = np.asarray(cam['projection_matrix'])[:, :3]
        with (h5py.File(args.root / f'{name}_gt.hdf5', 'r') as archive,
              h5py.File(args.root / f'{name}_data.hdf5', 'r') as data):
            camera = archive['davis/left']
            depths = camera['depth_image_rect']
            frames = range(1 if args.checkpoint else 0, len(depths))
            times = camera['depth_image_rect_ts'][:]
            for frame in spread(frames, args.frames_per_sequence):
                depth = depths[frame]
                packets = None if not args.checkpoint else [
                    mvsec_packet(data['davis/left/events'], times[i], cam, k) for i in (frame - 1, frame)
                ]
                yield name, frame, depth, k, 50., packets


def mvsec_packet(events, stop, camera, k):
    from mvsec_transfer import lower_bound_after

    begin = lower_bound_after(events, stop - .025)
    end = lower_bound_after(events, stop)
    packet = events[begin:end].T.copy()
    if begin != end:
        xy = cv2.fisheye.undistortPoints(packet[:2].T.copy().reshape(-1, 1, 2),
                                         camera_matrix(camera['intrinsics']),
                                         np.asarray(camera['distortion_coeffs']),
                                         R=np.asarray(camera['rectification_matrix']), P=k)[:, 0]
        size = np.asarray(camera['resolution'])
        packet[:2] = ((xy + .5) * (np.array([640, 480]) / size) - .5).T
    return packet


def depth_predictions(args):
    import torch
    import torch.nn.functional as F
    from event_dvlt import EventDVLT, voxelize

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['config']['f3']:
        raise ValueError('CPU geometry audit requires a checkpoint without F³')
    model = EventDVLT(args.assets, device='cpu').eval()
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad
                and not name.startswith(('gs_head.', 'field_features.', 'geometry_features.', 'event_adapter.'))
                and name != 'log_metric_scale'}
    if set(checkpoint['model']) != expected:
        raise ValueError('Unexpected adapters in the geometry checkpoint')
    model.load_state_dict(checkpoint['model'], strict=False)
    field = torch.zeros((1, 2, 32, 96, 128))

    @torch.inference_mode()
    def predict(packets, depth, max_depth):
        voxel = torch.stack([voxelize(packet) for packet in packets])[None]
        with torch.autocast('cpu', dtype=torch.bfloat16):
            tokens, pos, _ = model.geometry(voxel, field, steps=16)
            raw = model.backbone._decode(tokens, 168, 224, 1, 2, pos)['depth'][0, 1, ..., 0].float()
        modes = {'bilinear': False, 'nearest-exact': None}
        estimates = {name: F.interpolate(raw[None, None], depth.shape, mode=name,
                                         align_corners=align)[0, 0].numpy()
                     for name, align in modes.items()}
        valid = (depth >= .1) & (depth <= max_depth)
        for estimate in estimates.values():
            valid &= np.isfinite(estimate) & (estimate > 0)
        if not valid.any():
            raise ValueError('No common valid prediction and reference depth pixels')
        scale = np.median(depth[valid] / estimates['bilinear'][valid])
        return {name: estimate * scale for name, estimate in estimates.items()}

    return predict, checkpoint['step']


def frame_counts(depth, k, max_depth, predictions=None):
    """Count adjacent labelled pairs and phase-averaged same-voxel probability."""
    y, x = np.indices(depth.shape)
    rays = np.stack((x, y, np.ones_like(x)), axis=-1) @ np.linalg.inv(k).T
    valid = np.isfinite(depth) & (depth >= .1) & (depth <= max_depth)
    z = np.where(valid, depth, 1.)
    xyz = rays * z[..., None]
    logz = np.log(z)
    predicted_xyz, pred_logs = None, {}
    if predictions is not None:
        for estimate in predictions.values():
            valid &= np.isfinite(estimate) & (estimate > 0)
        predicted_xyz = rays * predictions['bilinear'][..., None]
        pred_logs = {mode: np.log(np.where(valid, estimate, 1.))
                     for mode, estimate in predictions.items()}
    left, right = np.s_[:, :-1], np.s_[:, 1:]
    top, bottom = np.s_[:-1, :], np.s_[1:, :]
    neighbours = [(a, b, valid[a] & valid[b], np.abs(logz[a] - logz[b]))
                  for a, b in ((left, right), (top, bottom))]
    result = dict(labelled_pairs=sum(int(ok.sum()) for _, _, ok, _ in neighbours), jumps={})
    for jump in DEPTH_JUMPS:
        difference, predicted_difference = [], []
        detected = {mode: dict(true_positives=0, positives=0) for mode in pred_logs}
        for a, b, ok, magnitude in neighbours:
            keep = ok & (magnitude >= np.log1p(jump))
            difference.append(np.abs(xyz[a][keep] - xyz[b][keep]))
            if predicted_xyz is not None:
                predicted_difference.append(np.abs(predicted_xyz[a][keep] - predicted_xyz[b][keep]))
                for mode, pred_logz in pred_logs.items():
                    predicted = np.abs(pred_logz[a] - pred_logz[b]) >= np.log1p(jump)
                    detected[mode]['true_positives'] += int((predicted & keep).sum())
                    detected[mode]['positives'] += int((predicted & ok).sum())
        offsets = np.concatenate(difference)
        distance = np.linalg.norm(offsets, axis=1)
        estimated = np.concatenate(predicted_difference) if predicted_difference else None
        grid_counts = {}
        for side in VOXEL_SIDES:
            # A voxel's diagonal bounds the separation in any orientation.
            possible = distance < np.sqrt(3) * side
            # Uniform grid-origin offsets are independent along the camera axes.
            probability = np.maximum(1 - offsets / side, 0).prod(axis=1)
            counts = dict(
                possible_any_orientation=int(possible.sum()),
                same_voxel_expected=float(probability.sum()),
            )
            if estimated is not None:
                predicted_probability = np.maximum(1 - estimated / side, 0).prod(axis=1)
                counts['predicted_same_voxel_expected'] = float(predicted_probability.sum())
                counts['false_same_voxel_expected'] = float(predicted_probability[~possible].sum())
            grid_counts[f'{side:.2f}'] = counts
        counts = dict(pairs=len(distance), grid=grid_counts)
        if detected:
            counts['predicted_jumps'] = detected
        result['jumps'][f'{jump:.2f}'] = counts
    return result


def summarize(rows):
    output = {}
    for jump in DEPTH_JUMPS:
        key = f'{jump:.2f}'
        jumps = [row['jumps'][key] for row in rows]
        pairs = sum(row['pairs'] for row in jumps)
        grid = {}
        for side in VOXEL_SIDES:
            size = f'{side:.2f}'
            grid[size] = {
                field + '_fraction': sum(row['grid'][size][field] for row in jumps) / pairs if pairs else None
                for field in jumps[0]['grid'][size]
            }
        summary = dict(pairs=pairs, grid=grid)
        for mode in jumps[0].get('predicted_jumps', {}):
            tp = sum(row['predicted_jumps'][mode]['true_positives'] for row in jumps)
            positives = sum(row['predicted_jumps'][mode]['positives'] for row in jumps)
            summary.setdefault('predicted_jumps', {})[mode] = dict(
                true_positives=tp, positives=positives, recall=tp / pairs if pairs else None,
                precision=tp / positives if positives else None,
            )
        output[key] = summary
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', choices=('eventscape', 'dsec', 'mvsec'))
    p.add_argument('root', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--split', type=Path, default=Path('scripts/dsec_full_split.json'))
    p.add_argument('--frames-per-sequence', type=int, default=8)
    p.add_argument('--assets', type=Path)
    p.add_argument('--checkpoint', type=Path, help='Also measure DVLT predictions on the reference jumps')
    args = p.parse_args()
    if args.frames_per_sequence < 1:
        p.error('--frames-per-sequence must be positive')
    if args.checkpoint is not None and args.assets is None:
        p.error('--checkpoint requires --assets')
    cv2.setNumThreads(1)
    predict, step = depth_predictions(args) if args.checkpoint else (None, None)
    rows = [dict(sequence=name, frame=frame, max_depth_m=max_depth,
                 **frame_counts(depth, k, max_depth, predict(packets, depth, max_depth) if predict else None))
            for name, frame, depth, k, max_depth, packets in source_frames(args)]
    if not rows:
        raise ValueError('No usable depth frames')
    summary = {
        'frames': len(rows), 'sequences': len({row['sequence'] for row in rows}),
        'labelled_pairs': sum(row['labelled_pairs'] for row in rows), 'jumps': summarize(rows),
    }
    report = dict(dataset=args.dataset, source=str(args.root), summary=summary, rows=rows)
    if predict:
        report.update(checkpoint=str(args.checkpoint), checkpoint_step=step, recurrent_steps=16,
                      geometry='Calibrated rays and one reference/prediction median scale ratio per frame',
                      predicted_pairs='Exact native-pixel neighbour agreement, no localization tolerance')
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, allow_nan=False))


if __name__ == '__main__':
    main()
