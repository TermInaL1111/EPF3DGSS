"""Test whether aligned events rank depth perturbations under fixed real appearance and pose."""

import argparse
import json
from contextlib import closing
from io import BytesIO
from pathlib import Path
from statistics import fmean

import cv2
import h5py
import numpy as np
import yaml
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation, Slerp

from audit_event_brightness import log_image, unclipped
from dsec_data import camera_matrix, indexed_lower_bound, rectified_transform
from ev3ds_pose import camera
from event_observation import polarity_counts
from eventscape_data import EventScapeArchive, EventScapeSequence
from mvsec_transfer import lower_bound_after


DEPTH_FACTORS = np.array([.8, .9, 1., 1.1, 1.2])
TRIPLETS = {'10_percent': [1, 2, 3], '20_percent': [0, 2, 4]}
CONDITIONS = ('aligned', 'later_interval', 'flipped', 'zero_change', 'aps_difference')


def normalized_pixels(points, k, distortion):
    if distortion is None:
        return (points - k[:2, 2]) @ np.linalg.inv(k[:2, :2]).T
    model, coefficients = distortion
    undistort = cv2.fisheye.undistortPoints if model == 'equidistant' else cv2.undistortPoints
    return undistort(points[:, None], k, coefficients)[:, 0]


def pixel_rays(shape, k, distortion):
    y, x = np.indices(shape, dtype=np.float64)
    points = np.column_stack((x.ravel(), y.ravel()))
    xy = normalized_pixels(points, k, distortion)
    return np.column_stack((xy, np.ones(len(points)))).reshape(*shape, 3)


def project(points, k, distortion):
    xy = points[:, :2] / points[:, 2:3].clip(min=1e-8)
    if distortion is None:
        return xy @ k[:2, :2].T + k[:2, 2]
    model, coefficients = distortion
    if model == 'equidistant':
        return cv2.fisheye.distortPoints(xy[:, None], k, coefficients)[:, 0]
    return cv2.projectPoints(np.column_stack((xy, np.ones(len(xy)))), np.zeros(3), np.zeros(3),
                             k, coefficients)[0][:, 0]


def transport_depth(depth, rays, k, distortion, image_from_measurement):
    """Move a sparse depth map to its APS timestamp, resolving collisions by z."""
    valid = np.isfinite(depth) & (depth > .1) & (depth < 50.)
    if not valid.any():
        return np.zeros_like(depth)
    points = (rays[valid]*depth[valid, None]) @ image_from_measurement[:3, :3].T
    points += image_from_measurement[:3, 3]
    xy = np.rint(project(points, k, distortion)).astype(np.intp)
    height, width = depth.shape
    keep = ((points[:, 2] > .1) & (xy >= 0).all(axis=-1)
            & (xy < [width, height]).all(axis=-1))
    z = np.full(depth.shape, np.inf)
    np.minimum.at(z, (xy[keep, 1], xy[keep, 0]), points[keep, 2])
    z[~np.isfinite(z)] = 0
    return z


def ev3ds_windows(root, scene_limit, window_limit):
    names = (root/'train_scenes.txt').read_text().split()
    k = np.loadtxt(root/'Intrinsics/0000.txt')[:3, :3]
    rays = pixel_rays((480, 640), k, None)
    for which in np.linspace(0, len(names)-1, min(scene_limit, len(names)), dtype=int):
        name = names[which]
        for index in np.linspace(4, 56, min(27, window_limit), dtype=int):
            images, depths = [], []
            for frame in (index, index+1):
                stem = root/'Scenes'/name/f'{frame:04}'
                images.append(cv2.imread(str(stem.with_suffix('.png')), cv2.IMREAD_GRAYSCALE))
                with np.load(stem.with_suffix('.npz')) as view:
                    depths.append(view['depth_map'])
            packets = [np.load(root/'Event'/name/f'{frame:04}.npy') for frame in (index+1, index+2)]
            pose = np.linalg.solve(camera(root, name, index), camera(root, name, index+1))
            yield name, int(index), images, depths, packets, rays, k, None, pose, [0., 0.]


def tracked_points(first, last, mask, max_corners=256):
    corners = cv2.goodFeaturesToTrack(first, maxCorners=max_corners, qualityLevel=.01, minDistance=5,
                                      mask=mask.astype(np.uint8))
    if corners is None:
        return np.empty((0, 2)), np.empty((0, 2))
    end, forward, _ = cv2.calcOpticalFlowPyrLK(first, last, corners, None)
    if end is None:
        return np.empty((0, 2)), np.empty((0, 2))
    back, backward, _ = cv2.calcOpticalFlowPyrLK(last, first, end, None)
    if back is None:
        return np.empty((0, 2)), np.empty((0, 2))
    consistent = ((forward[:, 0] > 0) & (backward[:, 0] > 0)
                  & (np.linalg.norm((back-corners)[:, 0], axis=-1) <= .3))
    return corners[consistent, 0].astype(np.float64), end[consistent, 0].astype(np.float64)


def eventscape_pose(first, last, depth, rays, k, score_tiles):
    valid = (.1 < depth) & (depth <= 200.) & unclipped(first)

    def tracks(mask):
        u, v = tracked_points(first, last, valid & mask, 768)
        x, y = np.rint(u).astype(np.intp).T
        return rays[y, x]*depth[y, x, None], v

    source, target = tracks(~score_tiles)
    held_out, observations = tracks(score_tiles)
    if len(source) < 30 or len(held_out) < 20:
        return None
    ok, rotation, translation, inliers = cv2.solvePnPRansac(
        source, target, k, None, iterationsCount=150, reprojectionError=1.,
        confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 25:
        return None
    i = inliers[:, 0]
    rotation, translation = cv2.solvePnPRefineLM(source[i], target[i], k, None, rotation, translation)
    matrix = cv2.Rodrigues(rotation)[0]
    projection = project(held_out @ matrix.T + translation[:, 0], k, None)
    residual = np.linalg.norm(projection-observations, axis=-1)
    median = float(np.median(residual))
    if median > 1.:
        return None
    b_from_a = np.eye(4)
    b_from_a[:3, :3], b_from_a[:3, 3] = matrix, translation[:, 0]
    return np.linalg.inv(b_from_a), dict(fit_tracks=len(source), inliers=len(i),
                                        heldout_tracks=len(held_out), median_px=median)


def eventscape_windows(path, scene_limit, window_limit, registration):
    shape = (256, 512)
    k = np.array([[256., 0., 256.], [0., 256., 128.], [0., 0., 1.]])
    rays = pixel_rays(shape, k, None)
    y, x = np.indices(shape)
    score_tiles = ((y//32+x//32) % 2).astype(bool)
    with closing(EventScapeArchive(path)) as archive:
        for town in ('Town01', 'Town02', 'Town03'):
            names = [name for name in archive.sequences if name.startswith(town + '/')]
            if not names:
                raise ValueError('Use the EventScape training archive with Town01, 02 and 03')
            for pos in np.linspace(0, len(names)-1, min(scene_limit, len(names)), dtype=int):
                seq = EventScapeSequence(archive, names[pos])
                vehicle = [np.loadtxt(BytesIO(archive.archive.read(f'{seq.sequence}/vehicle_data/{field}.txt')))
                           for field in ('timestamps', 'position')]
                for index in np.linspace(1, len(seq.ends)-3, min(window_limit, len(seq.ends)-3), dtype=int):
                    images = [cv2.cvtColor(seq.rgb(i), cv2.COLOR_RGB2GRAY) for i in (index,index+1)]
                    depths = [seq.depth(i) for i in (index,index+1)]
                    depths = [np.where((.1 < depth) & (depth <= 200.), depth, 0) for depth in depths]
                    fitted = eventscape_pose(*images, depths[0], rays, k, score_tiles)
                    row = dict(sequence=seq.sequence, frame=int(index), pose_supported=fitted is not None)
                    registration.append(row)
                    if fitted is None:
                        continue
                    pose, quality = fitted
                    xyz = np.stack([np.interp(seq.times[index:index+2], vehicle[0], coordinate)
                                    for coordinate in vehicle[1].T], axis=-1)
                    row.update(quality, camera_shift_m=float(np.linalg.norm(pose[:3, 3])),
                               vehicle_shift_m=float(np.linalg.norm(xyz[1]-xyz[0])))
                    packets = []
                    for i in (index+1,index+2):
                        packet = seq.packet(i)
                        time = packet[2].astype(np.float64)*1e-6
                        use = ((time > seq.times[i-1]) & (time <= seq.times[i])
                               & score_tiles[packet[1], packet[0]])
                        packets.append(packet[:, use])
                    yield seq.sequence, int(index), images, depths, packets, rays, k, None, pose, [0., 0.]


def dsec_trajectory(path):
    from rosbags.highlevel import AnyReader

    times, positions, rotations = [], [], []
    with AnyReader([path]) as reader:
        for connection, _, raw in reader.messages(connections=[c for c in reader.connections if c.topic == '/pose']):
            msg = reader.deserialize(raw, connection.msgtype)
            stamp, p, q = msg.header.stamp, msg.pose.position, msg.pose.orientation
            times.append(stamp.sec+stamp.nanosec*1e-9)
            positions.append((p.x, p.y, p.z))
            rotations.append((q.x, q.y, q.z, q.w))
    times, ids = np.unique(times, return_index=True)
    position = np.asarray(positions)[ids]
    rotation = Slerp(times, Rotation.from_quat(np.asarray(rotations)[ids]))

    def interpolate(at):
        result = np.broadcast_to(np.eye(4), (len(at), 4, 4)).copy()
        result[:, :3, :3] = rotation(at).as_matrix()
        result[:, :3, 3] = np.column_stack([np.interp(at, times, position[:, j]) for j in range(3)])
        return result

    return times[[0, -1]], interpolate


def dsec_rgbd(path, index, q, event_from_rgb, event_shape, event_k, distortion):
    gray = cv2.imread(str(path/f'images/left/rectified/{index:06}.png'), cv2.IMREAD_GRAYSCALE)
    disparity = cv2.imread(str(path/f'disparity/image/{index:06}.png'), cv2.IMREAD_UNCHANGED)
    xyz = cv2.reprojectImageTo3D(disparity.astype(np.float32)/256, q)
    z = xyz[..., 2]
    valid = (disparity > 0) & np.isfinite(z) & (z > .1) & (z <= 50.)
    event_xyz = xyz[valid] @ event_from_rgb[:3, :3].T + event_from_rgb[:3, 3]
    xy = np.rint(project(event_xyz, event_k, distortion)).astype(np.int64)
    h, w = event_shape
    keep = (event_xyz[:, 2] > 0) & ((xy >= 0) & (xy < [w, h])).all(axis=-1)
    pixel = (xy[keep, 1]*w + xy[keep, 0])
    depth = event_xyz[keep, 2]
    order = np.argsort(depth, kind='stable')
    _, nearest = np.unique(pixel[order], return_index=True)
    chosen = order[nearest]
    proxy_gray, proxy_depth = np.zeros(h*w, np.uint8), np.zeros(h*w, np.float32)
    proxy_gray[pixel[chosen]] = gray[valid][keep][chosen]
    proxy_depth[pixel[chosen]] = depth[chosen]
    return gray, np.where(valid, z, 0), proxy_gray.reshape(event_shape), proxy_depth.reshape(event_shape)


def dsec_windows(root, odometry, scene_limit, window_limit, registration):
    scenes = [p for p in sorted(root.iterdir()) if (p/'calibration/cam_to_lidar.yaml').is_file()
              and (odometry/(p.name.rsplit('_', 1)[0]+'-odometry')/'pose.bag').is_file()]
    if not scenes:
        raise ValueError('No DSEC sequences have extracted calibration and an odometry bag')
    for selected in np.linspace(0, len(scenes)-1, min(scene_limit, len(scenes)), dtype=int):
        path = scenes[selected]
        bag = odometry/(path.name.rsplit('_', 1)[0]+'-odometry')/'pose.bag'
        limits, interpolate = dsec_trajectory(bag)
        calib = yaml.safe_load((path/'calibration/cam_to_cam.yaml').read_text())
        hand_eye = yaml.safe_load((path/'calibration/cam_to_lidar.yaml').read_text())
        lidar_from_rgb = np.asarray(hand_eye['T_lidar_camRect1'])
        rgb_k = camera_matrix(calib['intrinsics']['camRect1']['camera_matrix'])
        event = calib['intrinsics']['cam0']
        k = camera_matrix(event['camera_matrix'])
        distortion = (event['distortion_model'], np.asarray(event['distortion_coeffs']))
        shape = tuple(reversed(event['resolution']))
        rays = pixel_rays(shape, k, distortion)
        raw_from_rect = np.eye(4)
        raw_from_rect[:3, :3] = np.asarray(calib['extrinsics']['R_rect0']).T
        event_from_rgb = raw_from_rect @ rectified_transform(calib, 0, 1)
        lidar_from_event = np.linalg.solve(event_from_rgb.T, lidar_from_rgb.T).T
        q = np.asarray(calib['disparity_to_depth']['cams_12'])
        times = np.loadtxt(path/'images/timestamps.txt', dtype=np.int64)
        labels = [int(p.stem) for p in sorted((path/'disparity/image').glob('*.png'))]
        triples = [(a,b,c) for a,b,c in zip(labels, labels[1:], labels[2:])
                   if limits[0] <= times[a]*1e-6 <= times[c]*1e-6 <= limits[1]
                   and max(times[b]-times[a], times[c]-times[b]) <= 130000]
        with h5py.File(path/'events/left/events.h5') as events:
            offset, lookup = int(events['t_offset'][()]), events['ms_to_idx'][:]
            for i in np.linspace(0, len(triples)-1, min(window_limit, len(triples)), dtype=int):
                a, b, c = triples[i]
                first, last = [dsec_rgbd(path, index, q, event_from_rgb, shape, k, distortion)
                               for index in (a,b)]
                lidar = interpolate(times[[a,b]].astype(np.float64)*1e-6)
                rgb_poses, event_poses = lidar @ lidar_from_rgb, lidar @ lidar_from_event
                rgb_pose = np.linalg.solve(*rgb_poses)
                tracks = aps_tracks(first[0], last[0], first[1], rgb_k, None, rgb_pose, first[1] > 0)
                usable = tracks is not None and tracks['tracks'] >= 25 and tracks['epe'][2] <= 2.
                row = dict(sequence=path.name, frame=a, pose_supported=usable,
                           rgb_track_count=tracks['tracks'] if tracks else 0,
                           rgb_epe_px=tracks['epe'][2] if tracks else None)
                registration.append(row)
                if not usable:
                    continue
                timeset = events['events/t']
                bounds = [indexed_lower_bound(timeset, lookup, int(at)-offset+1) for at in times[[a,b,c]]]
                packets = [np.stack([events[f'events/{key}'][lo:hi] for key in ('x','y','t','p')])
                           for lo,hi in zip(bounds, bounds[1:])]
                yield (path.name, a, [first[2],last[2]], [first[3],last[3]], packets, rays, k, distortion,
                       np.linalg.solve(*event_poses), [0., 0.])


def mvsec_windows(root, window_limit):
    name = 'indoor_flying4'
    calibration = yaml.safe_load((root/'calibration/camchain-imucam-indoor_flying.yaml').read_text())['cam0']
    fx, fy, cx, cy = calibration['intrinsics']
    k = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    distortion = ('equidistant', np.asarray(calibration['distortion_coeffs']))
    rays = pixel_rays(tuple(reversed(calibration['resolution'])), k, distortion)
    with h5py.File(root/f'{name}_data.hdf5') as data, h5py.File(root/f'{name}_gt.hdf5') as truth:
        cam, gt = data['davis/left'], truth['davis/left']
        times, depth_times, pose_times = cam['image_raw_ts'][:], gt['depth_image_raw_ts'][:], gt['pose_ts'][:]
        world_from_camera = gt['pose'][:]
        rotation = Slerp(pose_times, Rotation.from_matrix(world_from_camera[:, :3, :3]))
        nearest = np.abs(depth_times[:, None]-times).argmin(axis=0)
        depth_lags = depth_times[nearest]-times
        inside = ((times >= pose_times[0]) & (times <= pose_times[-1])
                  & (depth_times[nearest] >= pose_times[0]) & (depth_times[nearest] <= pose_times[-1])
                  & (np.abs(depth_lags) <= .025))
        eligible = np.flatnonzero(inside[:-2] & inside[1:-1])

        def poses_at(timestamps):
            poses = np.broadcast_to(np.eye(4), (len(timestamps), 4, 4)).copy()
            poses[:, :3, :3] = rotation(timestamps).as_matrix()
            for axis in range(3):
                poses[:, axis, 3] = np.interp(timestamps, pose_times, world_from_camera[:, axis, 3])
            return poses

        for entry in np.linspace(0, len(eligible)-1, min(len(eligible), window_limit), dtype=int):
            index = int(eligible[entry])
            frames = np.array([index, index+1])
            measurements = nearest[frames]
            a, b, da, db = poses_at(np.concatenate((times[frames], depth_times[measurements])))
            depths = [transport_depth(gt['depth_image_raw'][j], rays, k, distortion, np.linalg.solve(t, m))
                      for j, t, m in zip(measurements, (a, b), (da, db), strict=True)]
            events = cam['events']
            lo, mid, hi = [lower_bound_after(events, t) for t in times[index:index+3]]
            packets = [events[lo:mid].T, events[mid:hi].T]
            yield (name, index, cam['image_raw'][index:index+2], depths, packets, rays, k, distortion,
                   np.linalg.solve(a, b), (depth_lags[frames]*1000).tolist())


def surface_normals(depth, rays, y, x):
    """Estimate a shared local plane from four same-surface depth samples."""
    z = depth[y, x]
    inside = (y >= 2) & (y < depth.shape[0]-2) & (x >= 2) & (x < depth.shape[1]-2)
    neighbors = []
    for dy, dx in ((0, -2), (0, 2), (-2, 0), (2, 0)):
        yy = (y+dy).clip(0, depth.shape[0]-1)
        xx = (x+dx).clip(0, depth.shape[1]-1)
        nearby_z = depth[yy, xx]
        inside &= np.abs(nearby_z/z-1) <= .1
        neighbors.append(rays[yy, xx]*nearby_z[:, None])
    left, right, up, down = neighbors
    normal = np.cross(right-left, down-up)
    length = np.linalg.norm(normal, axis=-1)
    usable = inside & (length > 1e-10)
    normal = np.where(usable[:, None], normal/length.clip(min=1e-10)[:, None], [0., 0., 1.])
    normal[np.einsum('ij,ij->i', normal, rays[y, x]) < 0] *= -1
    return normal, usable


def log_differences(first, depths, rays, k, distortion, pose, y, x, local_planes,
                    *, check_ending_depth=True):
    """Predict same-pixel endpoint log changes by sampling one reference image."""
    if not len(x):
        return (np.empty((len(DEPTH_FACTORS), 0)), np.empty(0, dtype=bool), 0,
                np.empty((len(DEPTH_FACTORS), 0, 2)))
    reference_depth, ending_depth = depths
    before = log_image(first, 'published_code')[y, x]
    ray = rays[y, x] @ pose[:3, :3].T
    origin = pose[:3, 3]
    z = reference_depth[y, x]
    normal, estimated = (surface_normals(reference_depth, rays, y, x) if local_planes else
                         (np.broadcast_to([0., 0., 1.], (len(x), 3)), np.zeros(len(x), dtype=bool)))
    plane_depth = np.einsum('ij,ij->i', normal, rays[y, x])*z
    base_depth = normal @ origin
    derivative = np.einsum('ij,ij->i', normal, ray)
    proposals = []
    projections = []
    common = np.ones(len(x), dtype=bool)
    for factor in DEPTH_FACTORS:
        distance = (factor*plane_depth-base_depth) / derivative.clip(min=1e-8)
        points = origin + distance[:, None]*ray
        px, py = project(points, k, distortion).T
        local_depth = map_coordinates(reference_depth, [py, px], order=0, cval=0)
        # Compare to the unperturbed reference surface across all hypothesis projections.
        projected_plane = np.einsum('ij,ij->i', normal, points)
        surface_depth = plane_depth*points[:, 2]/projected_plane.clip(min=1e-8)
        valid = ((derivative > 0) & (distance > 0) & (points[:, 2] > 0)
                 & (np.abs(local_depth/surface_depth.clip(min=1e-8)-1) <= .1))
        brightness = map_coordinates(first.astype(np.float32), [py, px], order=1, cval=0)
        common &= valid & (brightness > 2) & (brightness < 253)
        if factor == 1. and check_ending_depth:
            common &= np.abs(ending_depth[y, x]/distance.clip(min=1e-8)-1) <= .1
        proposals.append(np.log(brightness.clip(min=1)/255)-before)
        projections.append(np.column_stack((px, py)))
    return np.stack(proposals)[:, common], common, int((common & estimated).sum()), np.stack(projections)[:, common]


def huber(error, scale):
    mag = np.abs(error)
    return np.where(mag <= scale, .5*mag**2/scale, mag-.5*scale)


def aps_tracks(first, last, reference_depth, k, distortion, pose, support):
    """Check the reference geometry against independent bidirectional image tracks."""
    start_xy, end_xy = tracked_points(first, last, support)
    if not len(start_xy):
        return None
    x, y = np.rint(start_xy).astype(np.intp).T
    z = reference_depth[y, x]
    normalized = normalized_pixels(start_xy, k, distortion)
    rays = np.column_stack((normalized, np.ones(len(normalized))))
    projections = []
    for factor in DEPTH_FACTORS:
        points = (factor*z[:, None]*rays-pose[:3, 3]) @ pose[:3, :3]
        projections.append(project(points, k, distortion))
    projections = np.stack(projections)
    rotation_only = project(rays @ pose[:3, :3], k, distortion)
    return dict(tracks=len(z), epe=np.linalg.norm(projections-end_xy, axis=-1).mean(-1).tolist(),
                displacement_from_true=np.linalg.norm(projections-projections[2], axis=-1).mean(-1).tolist(),
                rotation_only_epe=float(np.linalg.norm(rotation_only-end_xy, axis=-1).mean()))


def analyze(window, contrast, local_planes):
    name, index, images, depths, packets, rays, k, distortion, pose, depth_dt_ms = window
    first, last = images
    counts, shifted = [polarity_counts(packet, first.shape) for packet in packets]
    active = (counts.sum(0) > 0) & unclipped(first) & unclipped(last)
    valid = active & np.isfinite(depths[0]) & (depths[0] > 0) & (depths[1] > 0)
    y, x = np.where(valid)
    predicted, common, local_support, _ = log_differences(first, depths, rays, k, distortion, pose, y, x, local_planes)
    y, x = y[common], x[common]
    support = np.zeros_like(active)
    support[y, x] = True
    measured = contrast*(counts[0]-counts[1])[y, x]
    later = contrast*(shifted[0]-shifted[1])[y, x]
    aps = (log_image(last, 'published_code')-log_image(first, 'published_code'))[y, x]
    observations = np.stack([measured, later, -measured, np.zeros_like(measured), aps])
    costs = huber(predicted[None]-observations[:, None], contrast)
    patch = (y//16)*((first.shape[1]+15)//16) + x//16
    groups = []
    for block in np.unique(patch):
        ids = patch == block
        if ids.sum() < 16:
            continue
        groups.append(dict(patch=int(block), pixels=int(ids.sum()),
                           separation=float(np.median(np.ptp(predicted[:, ids], axis=0))),
                           costs=costs[:, :, ids].mean(-1).tolist()))
    return dict(sequence=name, frame=index, shift_mm=float(np.linalg.norm(pose[:3, 3])*1000),
                depth_dt_ms=depth_dt_ms, active_pixels=int(active.sum()), joint_depth_pixels=int(valid.sum()),
                common_pixels=len(x), local_plane_pixels=local_support, patches=groups,
                aps_tracks=aps_tracks(first, last, depths[0], k, distortion, pose, support),
                aps_warp_mae=float(np.abs(predicted[2]-aps).mean()) if len(x) else None,
                aps_identity_mae=float(np.abs(aps).mean()) if len(x) else None)


def summarize(rows):
    groups = [patch for row in rows for patch in row['patches']]
    blocks = []
    for min_separation in (0., .02, .05):
        eligible = [patch for patch in groups if patch['separation'] >= min_separation]
        if not eligible:
            continue
        costs = np.asarray([patch['costs'] for patch in eligible])
        report = {}
        for label, ids in TRIPLETS.items():
            trial = costs[:, :, ids]
            winners = np.isclose(trial, trial.min(-1, keepdims=True), rtol=1e-7, atol=1e-9)
            top1 = (winners[:, :, 1]/winners.sum(-1)).mean(0)
            true, near, far = (trial[:, :, j] for j in (1, 0, 2))

            def preference(other):
                ties = np.isclose(true, other, rtol=1e-7, atol=1e-9)
                return np.where(ties, .5, true < other).mean(0)

            report[label] = {
                name: dict(top1=float(win), vs_near=float(n), vs_far=float(f))
                for name, win, n, f in zip(CONDITIONS, top1, preference(near), preference(far), strict=True)}
        blocks.append(dict(min_separation=min_separation, patches=len(eligible),
                           fraction=len(eligible)/len(groups), results=report))
    active_rows = [row for row in rows if row['common_pixels']]
    active_pixels = sum(row['active_pixels'] for row in rows)
    common_pixels = sum(row['common_pixels'] for row in rows)
    tracks = [row['aps_tracks'] for row in rows if row['aps_tracks'] is not None]
    track_summary = None
    if tracks:
        count = sum(track['tracks'] for track in tracks)
        track_summary = dict(tracks=count, windows=len(tracks),
                             epe_px=(sum(track['tracks']*np.asarray(track['epe']) for track in tracks)/count).tolist(),
                             displacement_from_true_px=(sum(track['tracks']*np.asarray(track['displacement_from_true'])
                                                            for track in tracks)/count).tolist(),
                             rotation_only_epe_px=sum(
                                 track['tracks']*track['rotation_only_epe'] for track in tracks)/count)
    return dict(windows=len(rows), windows_with_support=len(active_rows), patches=len(groups),
                median_shift_mm=float(np.median([row['shift_mm'] for row in rows])) if rows else None,
                common_pixel_fraction=common_pixels/active_pixels if active_pixels else None,
                local_plane_fraction=sum(row['local_plane_pixels'] for row in rows)/common_pixels if common_pixels else None,
                aps_warp_mae=fmean(row['aps_warp_mae'] for row in active_rows) if active_rows else None,
                aps_identity_mae=fmean(row['aps_identity_mae'] for row in active_rows) if active_rows else None,
                aps_track_summary=track_summary, blocks=blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('ev3ds', 'mvsec', 'eventscape', 'dsec'))
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--contrast', type=float, required=True)
    parser.add_argument('--sequences', type=int, default=12, help='Sequence limit (per town for EventScape)')
    parser.add_argument('--windows', type=int, default=8)
    parser.add_argument('--odometry', type=Path, help='Root of the public DSEC odometry bags')
    parser.add_argument('--local-planes', action='store_true', help='Use depth-derived local normals where supported')
    args = parser.parse_args()
    if not np.isfinite(args.contrast) or min(args.contrast, args.sequences, args.windows) <= 0:
        parser.error('Contrast, sequence count and window count must be positive')
    if args.dataset == 'dsec' and not args.odometry:
        parser.error('DSEC requires --odometry')
    registration = []
    if args.dataset == 'ev3ds':
        windows = ev3ds_windows(args.root, args.sequences, args.windows)
    elif args.dataset == 'eventscape':
        windows = eventscape_windows(args.root, args.sequences, args.windows, registration)
    elif args.dataset == 'dsec':
        windows = dsec_windows(args.root, args.odometry, args.sequences, args.windows, registration)
    else:
        windows = mvsec_windows(args.root, args.windows)
    cv2.setNumThreads(1)
    rows = [analyze(window, args.contrast, args.local_planes) for window in windows]
    summary = summarize(rows)
    report = dict(dataset=args.dataset, contrast=args.contrast, local_planes=args.local_planes, factors=DEPTH_FACTORS.tolist(),
                  conditions=CONDITIONS, summary=summary, rows=rows, registration=registration)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
