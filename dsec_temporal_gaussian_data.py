"""DSEC monocular temporal windows with fixed offline RGB-D reference cameras."""

import argparse
from contextlib import closing
import json
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from dsec_data import DSECSequence, camera_matrix, rectified_transform, resize_intrinsics
from eventscape_gaussian_data import reference_camera


class DSECTemporalGaussianSequence(DSECSequence):
    def __init__(self, root, name, cameras=None, target_offset=2):
        super().__init__(root,name,window_us=50000)
        if target_offset not in (2,4,6,8):
            raise ValueError('DSEC target offset must be 2, 4, 6 or 8 frames')
        self.target_offset = target_offset
        self.source_offsets = (0,4) if target_offset != 4 else (0,2)
        self.rgb_from_event = rectified_transform(self.calib,1,0)
        calibrations = self.calib['intrinsics']
        self.rgb_k = resize_intrinsics(camera_matrix(calibrations['camRect1']['camera_matrix']),
                                      calibrations['camRect1']['resolution'],(640,480))
        self.event_k = resize_intrinsics(camera_matrix(calibrations['camRect0']['camera_matrix']),
                                        (640,480),(224,168))
        self.cameras = cameras
        eligible = set(self.indices)
        self.indices = [i for i in self.indices
                        if all(i+offset in eligible for offset in self.source_offsets)
                        and i+self.target_offset < len(self.times)
                        and (self.root/f'images/left/rectified/{i+self.target_offset:06}.png').is_file()
                        and (cameras is None or not isinstance(cameras.get(str(i)), dict)
                             or str(self.target_offset) in cameras[str(i)])
                        and self.times[i+self.source_offsets[1]]-self.times[i] <= 500000]
        if cameras is not None:
            available = set(self.indices)
            requested = sorted(map(int,cameras))
            if self.target_offset == 2:
                if set(requested)-available:
                    raise ValueError(f'Invalid camera window in {name}')
                self.indices = requested
            else:
                # A larger target offset legitimately drops windows that do not
                # contain the requested future frame.
                self.indices = [index for index in requested if index in available]
                if not self.indices:
                    raise ValueError(f'No window supports target offset {self.target_offset} in {name}')

    def packet(self, view, index):
        return super().packet(0,index+self.source_offsets[view])

    def gray(self, index):
        path = self.root/f'images/left/rectified/{index:06}.png'
        frame = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError(f'Cannot decode {path}')
        return cv2.resize(frame,(640,480),interpolation=cv2.INTER_AREA)

    def depth(self, index, event=False):
        location, key = ('event','cams_03') if event else ('image','cams_12')
        path = self.root/f'disparity/{location}/{index:06}.png'
        disparity = cv2.imread(str(path),-1)
        if disparity is None:
            raise ValueError(f'Cannot read {path}')
        d = disparity.astype(np.float32)/256
        z = cv2.reprojectImageTo3D(d,np.asarray(self.calib['disparity_to_depth'][key]))[...,2]
        return np.where((d>0)&np.isfinite(z)&(z>=.1)&(z<=50),z,0).astype(np.float32)

    def camera(self, cameras, offset):
        if isinstance(cameras, dict):
            if str(offset) not in cameras:
                raise ValueError(f'Camera offset {offset} is absent from {self.sequence}')
            return np.asarray(cameras[str(offset)]['c2w'])
        position = (offset - 2) // 2
        if position < 0 or position >= len(cameras):
            raise ValueError(f'Camera offset {offset} is absent from {self.sequence}')
        return np.asarray(cameras[position]['c2w'])

    def supervision(self, index):
        cameras = self.cameras[str(index)]
        query = self.camera(cameras,self.target_offset)
        later = self.camera(cameras,self.source_offsets[1])
        # RGB-D PnP provides C2W in the first RGB camera. Express all views
        # in the first event camera, including the displaced RGB query.
        event_from_rgb = np.linalg.inv(self.rgb_from_event)
        later_event = event_from_rgb @ later @ self.rgb_from_event
        poses = np.stack((np.eye(4),later_event))
        depth = np.stack([self.depth(index+offset,event=True) for offset in self.source_offsets])
        return depth, np.stack([self.event_k]*2), poses, event_from_rgb@query, self.rgb_k, self.gray(index+self.target_offset)

    def source_supervision(self, index):
        event_from_rgb = np.linalg.inv(self.rgb_from_event)
        later_rgb = self.camera(self.cameras[str(index)], self.source_offsets[1])
        poses = event_from_rgb @ np.stack((np.eye(4),later_rgb))
        gray = np.stack([self.gray(index+offset) for offset in self.source_offsets])
        return poses,self.rgb_k,gray


def prepare(root, names, count):
    y,x = np.indices((480,640))
    tiles = ((y//32+x//32)%2).astype(bool)
    sequences = {}
    for name in names:
        with closing(DSECTemporalGaussianSequence(root,name)) as source:
            indices = source.indices
            candidates = [indices[j] for j in np.linspace(0,len(indices)-1,min(count,len(indices)),dtype=int)]
            rows = {}
            for i in candidates:
                reference = source.gray(i)
                depth = cv2.resize(source.depth(i),(640,480),interpolation=cv2.INTER_NEAREST_EXACT)
                poses = [reference_camera(reference,source.gray(j),depth,tiles,source.rgb_k,50.)
                         for j in (i+2,i+4)]
                if all(poses):
                    rows[str(i)] = poses
            errors = [cam['median_reprojection_px'] for poses in rows.values() for cam in poses]
            sequences[name] = dict(attempted=len(candidates),accepted=len(rows),cameras=rows,
                                   median_reprojection_px=median(errors) if errors else None)
            print(name,len(rows),'/',len(candidates),'windows',flush=True)
    return dict(reference='left rectified RGB-D PnP with held-out pixel tiles',
                target_offset_frames=2, second_source_offset_frames=4,sequences=sequences)


def load_dsec_camera_splits(directory):
    splits = {}
    seen = set()
    for split in ('train','val','test'):
        record = json.loads((directory/f'{split}.json').read_text())
        sequences = record['sequences']
        if not sequences or seen.intersection(sequences):
            raise ValueError(f'Invalid or overlapping DSEC temporal split: {split}')
        seen.update(sequences)
        splits[split] = {name:row['cameras'] for name,row in sequences.items() if row['cameras']}
        if not splits[split]:
            raise ValueError(f'No accepted DSEC cameras in split: {split}')
    return splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--sequences',nargs='+',required=True)
    parser.add_argument('--windows',type=int,default=24)
    args = parser.parse_args()
    if args.windows < 1:
        parser.error('Window count must be positive')
    cv2.setNumThreads(1)
    cv2.setRNGSeed(7)
    report = prepare(args.root,args.sequences,args.windows)
    temporary = args.out.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.out)
