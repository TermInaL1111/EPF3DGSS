"""Read same-camera APS and events on the official MVSEC rectified grid."""

from types import SimpleNamespace
from zipfile import ZipFile

import cv2
import h5py
import numpy as np
import yaml

from dsec_data import camera_matrix, resize_intrinsics
from mvsec_transfer import lower_bound_after


class MVSECGeometrySequence:
    window_s = .05

    def __init__(self, root, sequence, calibration, stack):
        if calibration.suffix == '.zip':
            with ZipFile(calibration) as archive:
                (member,) = [name for name in archive.namelist() if name.endswith('.yaml')]
                cam = yaml.safe_load(archive.read(member))['cam0']
        else:
            cam = yaml.safe_load(calibration.read_text())['cam0']
        size = tuple(cam['resolution'])
        native = camera_matrix(cam['intrinsics'])
        distortion = np.asarray(cam['distortion_coeffs'])
        self.rect_rotation = np.asarray(cam['rectification_matrix'])
        rectified = np.asarray(cam['projection_matrix'])[:, :3]
        self.k = resize_intrinsics(rectified, size, (224,168))
        y,x = np.indices(size[::-1])
        pixels = np.stack((x,y),-1).astype(np.float64).reshape(-1,1,2)
        coords = cv2.fisheye.undistortPoints(pixels, native, distortion,
                                               R=self.rect_rotation, P=rectified)
        self.mapping = (coords.reshape(*y.shape,2)+.5)*([640,480]/np.asarray(size))-.5
        self.gray_maps = cv2.fisheye.initUndistortRectifyMap(
            native, distortion, self.rect_rotation, self.k, (224,168), cv2.CV_32FC1)
        data = stack.enter_context(h5py.File(root/f'{sequence}_data.hdf5','r'))
        self.camera = data['davis/left']
        self.events = self.camera['events']
        self.times = self.camera['image_raw_ts'][:]
        if not len(self.times) or not np.all(np.diff(self.times)>0):
            raise ValueError(f'Invalid APS timestamps in {sequence}')
        self.eligible = np.flatnonzero((self.times >= self.events[0,2]+self.window_s)
                                      & (self.times <= self.events[-1,2]))

    def packet(self, index):
        time = self.times[index]
        block = self.events[lower_bound_after(self.events,time-self.window_s):
                            lower_bound_after(self.events,time)]
        rectified = self.mapping[block[:,1].astype(int),block[:,0].astype(int)]
        return np.column_stack((rectified,block[:,2:])).T

    def rgb(self, index):
        gray = cv2.remap(self.camera['image_raw'][index], *self.gray_maps, cv2.INTER_LINEAR)
        return np.repeat(gray[...,None],3,axis=-1)

    def frame(self, index):
        return self.packet(index), self.rgb(index)


def adaptation_splits(root, sequence, calibration, stack):
    data = MVSECGeometrySequence(root, sequence, calibration, stack)
    cut = int(.9*len(data.times))
    groups = {'train': data.eligible[data.eligible < cut-18],
              'val': data.eligible[data.eligible >= cut]}
    splits = {}
    for name, indices in groups.items():
        ends = set(indices.tolist())
        pairs = [(i,i+1) for i in sorted(ends) if i+1 in ends and data.times[i+1]-data.times[i] <= .2]
        # Break clips at APS gaps as well as at the split boundary.
        splits[f'{sequence}/{name}'] = SimpleNamespace(
            pairs=pairs, ends=ends, frame=data.frame,
            contiguous={i for i,_ in pairs})
    return splits
