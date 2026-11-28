"""Register official DSEC LiDAR scans and synchronize the camera poses to held-out clips."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

from kiss_icp.config import load_config
from kiss_icp.kiss_icp import KissICP
from kiss_icp.tools.point_cloud2 import read_points
import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation, Slerp
import yaml

from dsec_data import rectified_transform


MAX_GAP_S = .15


def pose_trajectory(bag):
    config = load_config(None)
    config.registration.max_num_threads = 4
    odometry = KissICP(config)
    timestamps, poses, counts, durations = [], [], [], []
    with AnyReader([bag]) as reader:
        connections = [c for c in reader.connections if c.topic == '/velodyne_points']
        if not connections or any(c.msgtype != 'sensor_msgs/msg/PointCloud2' for c in connections):
            raise ValueError(f'Missing DSEC Velodyne scans in {bag}')
        for connection, _, packet in reader.messages(connections=connections):
            msg = reader.deserialize(packet,connection.msgtype)
            stamp = msg.header.stamp
            time = stamp.sec + 1e-9*stamp.nanosec
            if timestamps and time <= timestamps[-1]:
                raise ValueError(f'Nonincreasing LiDAR scan times in {bag}')
            samples = read_points(msg, field_names=['x','y','z','time'])
            xyz = np.column_stack((samples['x'],samples['y'],samples['z'])).astype(np.float64)
            offsets = samples['time'].astype(np.float64)
            valid = np.isfinite(xyz).all(axis=1) & np.isfinite(offsets)
            xyz, offsets = xyz[valid], offsets[valid]
            if len(xyz) < 100 or np.ptp(offsets) <= 0:
                raise ValueError(f'Insufficient timed LiDAR points in {bag} at {time}')
            # Per-point times are relative within a scan. The scan pose is
            # indexed by the separate PointCloud2 header timestamp.
            odometry.register_frame(xyz,offsets-offsets.min())
            pose = odometry.last_pose
            if not np.isfinite(pose).all() or not np.isclose(np.linalg.det(pose[:3,:3]),1.,atol=1e-5):
                raise ValueError(f'Invalid registration at {time} in {bag}')
            timestamps.append(time)
            poses.append(pose.copy())
            counts.append(len(xyz))
            durations.append(float(np.ptp(offsets)))
    if len(poses) < 2:
        raise ValueError(f'Insufficient scans in {bag}')
    return np.asarray(timestamps),np.asarray(poses),np.asarray(counts),np.asarray(durations)


def align_windows(root, name, windows, drive, times, poses):
    camera_root = root/name
    camera_times = np.loadtxt(camera_root/'images/timestamps.txt',dtype=np.int64)*1e-6
    camera = yaml.safe_load((camera_root/'calibration/cam_to_cam.yaml').read_text())
    lidar = yaml.safe_load((drive/'cam_to_lidar.yaml').read_text())
    lidar_from_rgb = np.asarray(lidar['T_lidar_camRect1'])
    lidar_from_event = lidar_from_rgb @ rectified_transform(camera,1,0)
    rotations = Slerp(times,Rotation.from_matrix(poses[:,:3,:3]))
    aligned = {}
    for indices in windows:
        frame_times = camera_times[indices]
        if frame_times[0] < times[0] or frame_times[-1] > times[-1]:
            continue
        upper = np.searchsorted(times,frame_times,side='right').clip(1,len(times)-1)
        valid = times[upper]-times[upper-1] <= MAX_GAP_S
        if not valid.all():
            continue
        synced = np.repeat(np.eye(4)[None],len(indices),axis=0)
        synced[:,:3,:3] = rotations(frame_times).as_matrix()
        for axis in range(3):
            synced[:,axis,3] = np.interp(frame_times,times,poses[:,axis,3])
        event_cameras = synced @ lidar_from_event
        rgb_cameras = synced @ lidar_from_rgb
        aligned[str(indices[0])] = dict(
            frames=indices,
            event_c2w=np.linalg.solve(event_cameras[0],event_cameras).tolist(),
            rgb_c2w=np.linalg.solve(rgb_cameras[0],rgb_cameras).tolist(),
            max_lidar_bracket_s=float((times[upper]-times[upper-1]).max()),
        )
    return aligned


def prepare(root, clips, odometry):
    trajectories = {}
    aligned = {}
    for name,row in clips['sequences'].items():
        drive = name.rsplit('_',1)[0]
        if drive not in trajectories:
            trajectories[drive] = pose_trajectory(odometry/drive/'lidar_imu.bag')
        times, poses, _, _ = trajectories[drive]
        aligned[name] = align_windows(root,name,row['windows'],odometry/drive,times,poses)
        print(name,len(aligned[name]),'/',len(row['windows']),'synchronized clips',flush=True)
    diagnostics = {}
    for drive,(times,poses,counts,durations) in trajectories.items():
        distances=np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)
        diagnostics[drive] = dict(scans=len(times),timestamp_s=[float(times[0]),float(times[-1])],
                                  median_points=float(np.median(counts)),median_scan_span_s=float(np.median(durations)),
                                  median_scan_gap_s=float(np.median(np.diff(times))),
                                  largest_step_m=float(distances.max()))
    return dict(reference='KISS-ICP pseudo ground truth from official Velodyne scans',
                kiss_icp_version=version('kiss-icp'),scan_time='point cloud header',
                max_interpolation_gap_s=MAX_GAP_S,drives=diagnostics,sequences=aligned)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','clips','odometry','out'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    clips=json.loads(args.clips.read_text())
    report=prepare(args.root,clips,args.odometry)
    if not any(report['sequences'].values()):
        raise ValueError('No DSEC clips synchronized with LiDAR reference')
    temporary=args.out.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,separators=(',',':'),allow_nan=False)+'\n')
    temporary.replace(args.out)
