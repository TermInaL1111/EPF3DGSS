"""Single-camera temporal training windows with the MVSEC shared render convention."""
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from dsec_data import resize_intrinsics
from mvsec_transfer import calibration, lower_bound_after
from real_geometry import depth_lookup


class MVSECTemporalGaussianSequence:
    def __init__(self, root: Path, sequence: str, calibration_path: Path):
        self.mapping, self.inverse = calibration(calibration_path)
        self.xy, self.z_factor = depth_lookup(calibration_path, 480, 640)
        self.data = h5py.File(root/f'{sequence}_data.hdf5')
        try:
            self.truth = h5py.File(root/f'{sequence}_gt.hdf5')
        except Exception:
            self.data.close()
            raise
        self.camera = self.data['davis/left']
        self.gt = self.truth['davis/left']
        self.times = self.camera['image_raw_ts'][:]
        self.pose_times = self.gt['pose_ts'][:]
        self.depth_times = self.gt['depth_image_rect_ts'][:]
        self.world_from_camera = self.gt['pose'][:]
        self.rotation = Slerp(self.pose_times, Rotation.from_matrix(self.world_from_camera[:,:3,:3]))
        self.indices = (np.flatnonzero((self.times[1:-4] >= self.pose_times[0]) &
                                      (self.times[5:] <= self.pose_times[-1])) + 1).tolist()
        self.intrinsics = np.array([[888.8889,0,319.5],[0,888.8889,239.5],[0,0,1.]])
        self.source_intrinsics = resize_intrinsics(self.intrinsics,(640,480),(224,168))

    def close(self):
        self.truth.close()
        self.data.close()

    def packet(self, view, index):
        frame = index+4*view
        events = self.camera['events']
        begin = lower_bound_after(events,self.times[frame-1])
        end = lower_bound_after(events,self.times[frame])
        block = events[begin:end]
        coords = self.mapping[block[:,1].astype(int),block[:,0].astype(int)]
        return np.column_stack([coords,block[:,2:]]).T

    def supervision(self, index):
        frames = [index,index+2,index+4]
        times = self.times[frames]
        poses = np.broadcast_to(np.eye(4),(3,4,4)).copy()
        poses[:,:3,:3] = self.rotation(times).as_matrix()
        poses[:,:3,3] = np.stack([
            np.interp(times,self.pose_times,self.world_from_camera[:,axis,3])
            for axis in range(3)],axis=-1)
        poses = np.linalg.solve(poses[0],poses)
        depths = []
        for time in times[::2]:
            closest = np.argmin(np.abs(self.depth_times-time))
            if abs(self.depth_times[closest]-time) > .025:
                depths.append(np.zeros((480,640),np.float32))
            else:
                depth = self.gt['depth_image_rect'][closest][self.xy[...,1],self.xy[...,0]]/self.z_factor
                depths.append(np.where((depth>=.1)&(depth<=50),depth,0))
        gray = cv2.remap(self.camera['image_raw'][frames[1]],*self.inverse,cv2.INTER_LINEAR)
        return (np.stack(depths),np.stack([self.source_intrinsics]*2),poses[::2],
                poses[1],self.intrinsics,gray)
