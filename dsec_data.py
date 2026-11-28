"""DSEC stereo samples: native events and calibrated RGB/token correspondences.

Only supervision uses disparity/calibration. Student inputs are event packets.
Official conventions: https://dsec.ifi.uzh.ch/data-format/
"""
from pathlib import Path
import cv2
import h5py
import hdf5plugin  # registers the official Blosc HDF5 filter
import numpy as np
import yaml


def camera_matrix(values):
    fx, fy, cx, cy = values
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], dtype=np.float64)


def rectified_transform(calib, target, source):
    poses = {0: np.eye(4)}
    for i in range(1, 4):
        poses[i] = np.asarray(calib['extrinsics'][f'T_{i}{i-1}']) @ poses[i-1]
    def rect(i):
        r = np.eye(4)
        r[:3, :3] = calib['extrinsics'][f'R_rect{i}']
        return r @ poses[i]
    return rect(target) @ np.linalg.inv(rect(source))


def lower_bound(dataset, value):
    lo, hi = 0, len(dataset)
    while lo < hi:
        mid = (lo + hi) // 2
        if dataset[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def indexed_lower_bound(timestamps, milliseconds, value):
    """Use the DSEC millisecond index and refine to the exact microsecond."""
    bucket = value//1000
    if bucket < 0 or bucket+1 >= len(milliseconds):
        return lower_bound(timestamps,value)
    lo,hi = map(int,milliseconds[bucket:bucket+2])
    if hi < lo or hi > len(timestamps):
        raise ValueError('Invalid DSEC millisecond index')
    return lo+int(np.searchsorted(timestamps[lo:hi],value,side='left'))


class DSECSequence:
    def __init__(self, root, sequence, window_us=20000, *, monocular=False):
        self.root = Path(root) / sequence
        self.sequence = sequence
        self.window_us = window_us
        self.calib = yaml.safe_load((self.root/'calibration/cam_to_cam.yaml').read_text())
        self.times = np.loadtxt(self.root/'images/timestamps.txt', dtype=np.int64)
        disparity_times = set(np.loadtxt(self.root/'disparity/timestamps.txt',dtype=np.int64,ndmin=1))
        sides = ('left',) if monocular else ('left','right')
        self.files = [h5py.File(self.root/f'events/{side}/events.h5', 'r') for side in sides]
        self.offsets = [int(f['t_offset'][()]) for f in self.files]
        self.millisecond_indices = [f['ms_to_idx'][:] if 'ms_to_idx' in f else () for f in self.files]
        bounds = [(offset+int(f['events/t'][0]),offset+int(f['events/t'][-1]))
                  for offset,f in zip(self.offsets,self.files)]
        self.maps = []
        for side in sides:
            with h5py.File(self.root/f'events/{side}/rectify_map.h5', 'r') as f:
                self.maps.append(f['rectify_map'][:])
        self.indices = []
        for p in sorted((self.root/'disparity/image').glob('*.png')):
            index = int(p.stem)
            if index >= len(self.times):
                raise ValueError(f'Image index out of range: {p}')
            at = int(self.times[index])
            if at not in disparity_times:
                raise ValueError(f'Disparity/image timestamp mismatch: {p}')
            if all(first+self.window_us <= at <= last for first,last in bounds):
                self.indices.append(index)

    def close(self):
        for f in self.files:
            f.close()

    def packet(self, view, index):
        f = self.files[view]
        end = int(self.times[index])-self.offsets[view]
        timestamps, milliseconds = f['events/t'], self.millisecond_indices[view]
        lo = indexed_lower_bound(timestamps,milliseconds,end-self.window_us)
        hi = indexed_lower_bound(timestamps,milliseconds,end)
        x, y = f['events/x'][lo:hi], f['events/y'][lo:hi]
        xy = self.maps[view][y, x]
        # Relative microseconds avoid precision loss with large absolute timestamps.
        return np.column_stack((xy, f['events/t'][lo:hi].astype(np.float64)-end,
                                f['events/p'][lo:hi])).T

    def supervision(self, index):
        disparity = cv2.imread(str(self.root/f'disparity/image/{index:06}.png'), -1)
        if disparity is None:
            raise FileNotFoundError(index)
        disparity = disparity.astype(np.float32)/256
        xyz = cv2.reprojectImageTo3D(disparity, np.asarray(self.calib['disparity_to_depth']['cams_12']))
        valid = (disparity > 0) & np.isfinite(xyz).all(-1) & (xyz[...,2] > 0)
        points = xyz[valid].astype(np.float64)
        rgb, mappings = [], []
        for event_camera, image_camera, side in ((0,1,'left'), (3,2,'right')):
            im = cv2.imread(str(self.root/f'images/{side}/rectified/{index:06}.png'))
            if im is None:
                raise FileNotFoundError(f'{side}/{index:06}.png')
            rgb.append(cv2.cvtColor(cv2.resize(im, (224,168), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            projected = []
            for camera in (event_camera, image_camera):
                trans = rectified_transform(self.calib, camera, 1)
                local = points @ trans[:3,:3].T + trans[:3,3]
                k = camera_matrix(self.calib['intrinsics'][f'camRect{camera}']['camera_matrix'])
                uvz = local @ k.T
                projected.append((uvz[:,:2]/np.maximum(uvz[:,2:], 1e-8), local[:,2]))
            (event_xy, z), (image_xy, image_z) = projected
            keep = ((z>0)&(image_z>0)&(event_xy[:,0]>=0)&(event_xy[:,0]<640)&
                    (event_xy[:,1]>=0)&(event_xy[:,1]<480)&(image_xy[:,0]>=0)&
                    (image_xy[:,0]<1440)&(image_xy[:,1]>=0)&(image_xy[:,1]<1080))
            exy, ixy, z = event_xy[keep], image_xy[keep], z[keep]
            pixels = exy.astype(int)
            linear = pixels[:,1]*640+pixels[:,0]
            # Retain the nearest surface per event pixel (z-buffer).
            order = np.argsort(z, kind='stable')
            _, unique = np.unique(linear[order], return_index=True)
            chosen = order[unique]
            exy, ixy, z = exy[chosen], ixy[chosen], z[chosen]
            patch = (exy[:,1]*12/480).astype(int)*16 + (exy[:,0]*16/640).astype(int)
            grid = (ixy + .5)/np.array([1440,1080])*2-1
            mappings.append({'patch':patch, 'grid':grid.astype(np.float32), 'depth':z.astype(np.float32)})
        return np.stack(rgb), mappings


def realm_rotation_grid(cal):
    """Adapt REALM Warper to the DSEC raw-event and rectified RGB calibration."""
    ev=cal['intrinsics']['cam0'];im=cal['intrinsics']['camRect1']
    ke=camera_matrix(ev['camera_matrix']);ki=camera_matrix(im['camera_matrix'])
    y,x=np.indices((480,640));xy=np.stack([x,y],-1).astype(np.float64)
    rays=cv2.undistortPoints(xy.reshape(-1,1,2),ke,np.array(ev['distortion_coeffs'])).reshape(-1,2)
    rotation=np.array(cal['extrinsics']['R_rect1'])@np.array(cal['extrinsics']['T_10'])[:3,:3]
    p=np.column_stack([rays,np.ones(len(rays))])@rotation.T@ki.T
    return (p[:,:2]/p[:,2:]).reshape(480,640,2).astype(np.float32)


class DSECTemporalPairs:
    """Two successive left-camera timestamps at stride 2, using REALM's event count."""

    def __init__(self, root, sequence):
        self.root = Path(root) / sequence
        calib = yaml.safe_load((self.root/'calibration/cam_to_cam.yaml').read_text())
        grid = realm_rotation_grid(calib)
        # Common 4:3 crop chosen inside the calibrated RGB field of view.
        self.x, self.y, self.width, self.height = 40, 10, 560, 420
        self.grid = grid[self.y:self.y+self.height, self.x:self.x+self.width]
        rgb_shape = calib['intrinsics']['camRect1']['resolution']
        if not ((self.grid >= 0) & (self.grid < np.array(rgb_shape)-1)).all():
            raise ValueError(f'{sequence}: training crop extends outside RGB support')
        self.times = np.loadtxt(self.root/'images/timestamps.txt', dtype=np.int64)
        self.event_path = self.root/'events/left/events.h5'
        with h5py.File(self.event_path, 'r') as f:
            timestamps = f['events/t']
            offset = int(f['t_offset'][()])
            first,last = int(timestamps[0]),int(timestamps[-1])
            milliseconds = f['ms_to_idx'][:] if 'ms_to_idx' in f else ()
            self.ends = {}
            for i in range(0, len(self.times), 2):
                end = int(self.times[i])-offset
                if first <= end <= last:
                    pos = indexed_lower_bound(timestamps, milliseconds, end)
                    if pos >= 150000 and self.image(i).is_file():
                        self.ends[i] = pos
        self.pairs = [(i, i+2) for i in self.ends if i+2 in self.ends]

    def image(self, index):
        return self.root/f'images/left/rectified/{index:06}.png'

    def frame(self, index):
        with h5py.File(self.event_path, 'r') as f:
            hi = self.ends[index]
            events = f['events']
            sl = slice(hi-150000, hi)
            x, y, t, p = (events[key][sl] for key in ('x','y','t','p'))
        x, y = x.astype(np.int32)-self.x, y.astype(np.int32)-self.y
        keep = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        packet = np.stack((x[keep]*640/self.width, y[keep]*480/self.height,
                           t[keep].astype(np.float64)-int(t[-1]), p[keep]))
        bgr = cv2.imread(str(self.image(index)))
        if bgr is None:
            raise ValueError(f'Could not decode {self.image(index)}')
        rgb = cv2.cvtColor(cv2.remap(bgr,self.grid,None,cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
        return packet, cv2.resize(rgb, (224,168), interpolation=cv2.INTER_AREA)


def stereo_supervision(sequence, index):
    """Project sparse left RGB disparity into both event views, using a z-buffer."""
    root, calib = sequence.root, sequence.calib
    disparity = cv2.imread(str(root/f'disparity/image/{index:06}.png'), -1)
    if disparity is None:
        raise FileNotFoundError(f'Missing disparity at {index}')
    disparity = disparity.astype(np.float32)/256
    xyz = cv2.reprojectImageTo3D(disparity, np.asarray(calib['disparity_to_depth']['cams_12']))
    valid = (disparity > 0) & np.isfinite(xyz).all(-1) & (xyz[...,2] > 0)
    points = xyz[valid]
    depths, intrinsics, poses = [], [], []
    for camera in (0,3):
        transform = rectified_transform(calib, camera, 1)
        local = points @ transform[:3,:3].T + transform[:3,3]
        k = camera_matrix(calib['intrinsics'][f'camRect{camera}']['camera_matrix'])
        positive = local[:,2] > 0
        local = local[positive]
        pixels = local @ k.T
        xy = np.rint(pixels[:,:2]/pixels[:,2:]).astype(int)
        keep = (xy[:,0]>=0)&(xy[:,0]<640)&(xy[:,1]>=0)&(xy[:,1]<480)
        xy, z = xy[keep], local[keep,2]
        depth = np.full((480,640), np.inf, dtype=np.float32)
        np.minimum.at(depth, (xy[:,1],xy[:,0]), z)
        depth[~np.isfinite(depth)] = 0
        depths.append(depth)
        intrinsics.append(resize_intrinsics(k, (640,480), (224,168)))
        poses.append(rectified_transform(calib, 0, camera))
    right = cv2.imread(str(root/f'images/right/rectified/{index:06}.png'),cv2.IMREAD_GRAYSCALE)
    if right is None:
        raise FileNotFoundError(f'Missing right RGB at {index}')
    k = camera_matrix(calib['intrinsics']['camRect2']['camera_matrix'])
    query = rectified_transform(calib, 0, 2)
    return (np.stack(depths), np.stack(intrinsics), np.stack(poses), query,
            resize_intrinsics(k, (right.shape[1],right.shape[0]), (640,480)),
            cv2.resize(right,(640,480),interpolation=cv2.INTER_AREA))


def resize_intrinsics(k, source_wh, target_wh):
    k = k.copy()
    for axis, scale in enumerate(np.array(target_wh)/source_wh):
        k[axis,axis] *= scale
        k[axis,2] = (k[axis,2]+.5)*scale-.5
    return k
