"""Event-only source windows with separately estimated EventScape RGB-D cameras."""

import argparse
from contextlib import closing
import json
from pathlib import Path

import cv2
import numpy as np

from dsec_data import resize_intrinsics
from eventscape_data import EventScapeArchive, EventScapeSequence


NATIVE_K = np.array([[256., 0., 256.], [0., 256., 128.], [0., 0., 1.]])
ARCHIVES = {'train': 'Town01-03_train.zip', 'val': 'Town05_val.zip', 'test': 'Town05_test.zip'}


def reference_camera(first, target, depth, tiles, intrinsics=NATIVE_K, max_depth=200.):
    """Fit source-to-target PnP and validate it with separate spatial tiles."""
    valid = np.isfinite(depth) & (depth >= .1) & (depth <= max_depth)
    corners = cv2.goodFeaturesToTrack(first, maxCorners=1536, qualityLevel=.01,
                                      minDistance=5, mask=valid.astype(np.uint8))
    if corners is None or len(corners) < 50:
        return None
    end, forward, _ = cv2.calcOpticalFlowPyrLK(first, target, corners, None)
    if end is None:
        return None
    back, backward, _ = cv2.calcOpticalFlowPyrLK(target, first, end, None)
    if back is None:
        return None
    keep = ((forward[:, 0] > 0) & (backward[:, 0] > 0)
            & (np.linalg.norm((back-corners)[:, 0], axis=-1) <= .3))
    source, end = corners[keep, 0].astype(np.float64), end[keep, 0].astype(np.float64)
    x, y = np.rint(source).astype(np.intp).T
    k_inv = np.linalg.inv(intrinsics)
    rays = np.column_stack((source, np.ones(len(source)))) @ k_inv.T
    points = rays * depth[y, x, None]
    held = tiles[y, x]
    if held.sum() < 20 or (~held).sum() < 30:
        return None
    fit_points, fit_pixels = points[~held], end[~held]
    ok, rotation, translation, inliers = cv2.solvePnPRansac(fit_points, fit_pixels, intrinsics, None,
        iterationsCount=150, reprojectionError=1., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 25:
        return None
    fit = inliers[:, 0]
    rotation, translation = cv2.solvePnPRefineLM(fit_points[fit], fit_pixels[fit],
                                                intrinsics, None, rotation, translation)
    projected, _ = cv2.projectPoints(points[held], rotation, translation, intrinsics, None)
    residual = np.linalg.norm(projected[:, 0] - end[held], axis=-1)
    error = float(np.median(residual))
    if not np.isfinite(error) or error > 1.:
        return None
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = cv2.Rodrigues(rotation)[0], translation[:, 0]
    return dict(c2w=np.linalg.inv(matrix).tolist(), heldout_pixels=int(held.sum()),
                fits=int(len(fit)), median_reprojection_px=error)


def prepare(root, split, windows_per_sequence, sequence_limit=None):
    y, x = np.indices((256, 512))
    tiles = ((y // 32 + x // 32) % 2).astype(bool)
    rows = {}
    attempted = 0
    with closing(EventScapeArchive(root / ARCHIVES[split])) as archive:
        for name in archive.sequences[:sequence_limit]:
            sequence = EventScapeSequence(archive, name)
            if len(sequence.ends) < 6:
                continue
            candidates = np.linspace(1, len(sequence.ends)-5,
                                     min(windows_per_sequence, len(sequence.ends)-5), dtype=int)
            for i in candidates.tolist():
                attempted += 1
                first = cv2.cvtColor(sequence.rgb(i), cv2.COLOR_RGB2GRAY)
                depth = sequence.depth(i)
                cameras = [reference_camera(first, cv2.cvtColor(sequence.rgb(j), cv2.COLOR_RGB2GRAY),
                                            depth, tiles) for j in (i+2, i+4)]
                if all(cameras):
                    rows.setdefault(name, {})[str(i)] = cameras
    return dict(split=split, attempted=attempted, windows=sum(map(len,rows.values())), sequences=rows)


class EventScapeGaussianSequence:
    """Use event packets i/i+4, grey target i+2 and an offline camera reference."""

    def __init__(self, archive, name, cameras, target_offset=2):
        if target_offset not in (2,4,6,8):
            raise ValueError('EventScape target offset must be 2, 4, 6 or 8 frames')
        self.sequence = EventScapeSequence(archive, name)
        self.name = name
        self.cameras = cameras
        self.target_offset = target_offset
        self.source_offsets = (0,4) if target_offset != 4 else (0,2)
        self.indices = [i for i in sorted(map(int, cameras))
                        if i+self.target_offset < len(self.sequence.ends)
                        and self.has_offset(cameras[str(i)], self.target_offset)
                        and self.has_offset(cameras[str(i)], self.source_offsets[1])]
        if not self.indices:
            raise ValueError(f'No valid target offset {target_offset} in {name}')
        self.source_k = resize_intrinsics(NATIVE_K, (512,256), (224,168))
        self.render_k = resize_intrinsics(NATIVE_K, (512,256), (640,480))

    @staticmethod
    def has_offset(cameras, offset):
        if isinstance(cameras, dict):
            return str(offset) in cameras
        position = (offset - 2) // 2
        return 0 <= position < len(cameras)

    def camera(self, cameras, offset):
        if isinstance(cameras, dict):
            if str(offset) not in cameras:
                raise ValueError(f'Camera offset {offset} is absent from {self.name}')
            return np.asarray(cameras[str(offset)]['c2w'])
        position = (offset - 2) // 2
        if position < 0 or position >= len(cameras):
            raise ValueError(f'Camera offset {offset} is absent from {self.sequence.name}')
        return np.asarray(cameras[position]['c2w'])

    def packet(self, view, index):
        packet = self.sequence.packet(index+self.source_offsets[view]).astype(np.float64)
        packet[:2] *= np.array([640/512,480/256])[:,None]
        return packet

    def native_gray(self, index):
        return cv2.cvtColor(self.sequence.rgb(index+self.target_offset), cv2.COLOR_RGB2GRAY)

    def source_rgb(self, index):
        return np.stack([self.sequence.rgb(index+offset) for offset in self.source_offsets])

    def supervision(self, index):
        cameras = self.cameras[str(index)]
        query = self.camera(cameras,self.target_offset)
        later = self.camera(cameras,self.source_offsets[1])
        depths = np.stack([cv2.resize(self.sequence.depth(index+offset), (640,480),
                                     interpolation=cv2.INTER_NEAREST)
                           for offset in self.source_offsets])
        depths = np.where(np.isfinite(depths) & (depths >= .1) & (depths <= 200),depths,0)
        gray = cv2.resize(self.native_gray(index), (640,480), interpolation=cv2.INTER_LINEAR)
        return depths, np.stack([self.source_k]*2), np.stack((np.eye(4),later)), query, self.render_k, gray

    def source_supervision(self, index):
        later = self.camera(self.cameras[str(index)],self.source_offsets[1])
        gray = np.stack([
            cv2.resize(cv2.cvtColor(self.sequence.rgb(index+offset),cv2.COLOR_RGB2GRAY),
                       (640,480),interpolation=cv2.INTER_LINEAR)
            for offset in self.source_offsets
        ])
        return np.stack((np.eye(4),later)),self.render_k,gray


def load_camera_splits(directory):
    splits = {}
    seen = set()
    for split in ARCHIVES:
        record = json.loads((directory/f'{split}.json').read_text())
        sequences = record['sequences']
        if (record['split'] != split or not sequences or any(not rows for rows in sequences.values())
                or sum(map(len, sequences.values())) != record['windows'] or seen.intersection(sequences)):
            raise ValueError(f'Invalid or overlapping EventScape camera split: {split}')
        splits[split] = sequences
        seen.update(sequences)
    return splits


def open_gaussian_sequences(root, splits, stack, target_offset=2):
    data = {}
    for split, sequences in splits.items():
        archive = stack.enter_context(closing(EventScapeArchive(root / ARCHIVES[split])))
        if sequences.keys() - set(archive.sequences):
            raise ValueError(f'EventScape camera manifest contains unknown sequences: {split}')
        for name, rows in sequences.items():
            try:
                data[name] = EventScapeGaussianSequence(archive, name, rows,
                                                        target_offset=target_offset)
            except ValueError:
                if target_offset == 2:
                    raise
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('split', choices=tuple(ARCHIVES))
    p.add_argument('out', type=Path)
    p.add_argument('--windows', type=int, default=8)
    p.add_argument('--sequence-limit', type=int)
    args = p.parse_args()
    if args.windows < 1 or (args.sequence_limit is not None and args.sequence_limit < 1):
        p.error('Window and sequence limits must be positive')
    cv2.setNumThreads(1)
    cv2.setRNGSeed(7)
    report = prepare(args.root, args.split, args.windows, args.sequence_limit)
    if not report['windows']:
        raise ValueError('No independently validated camera windows')
    tmp = args.out.with_suffix('.tmp')
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    tmp.replace(args.out)
    print(args.split, report['windows'], '/', report['attempted'], 'windows')
