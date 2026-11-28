"""Evaluate the official EvGGS model on EV3DS and the shared MVSEC windows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from image_metrics import PerceptualMetrics, gray_metrics


def camera_batch(intrinsic, query, height, width, far):
    from lib.renderer.rend_utils import focal2fov, getProjectionMatrix

    view = torch.linalg.inv(query).T
    projection = getProjectionMatrix(.01, far, intrinsic, height, width).to(query).T
    return {
        'H': [height], 'W': [width],
        'FovX': [focal2fov(intrinsic[0, 0], width)],
        'FovY': [focal2fov(intrinsic[1, 1], height)],
        'world_view_transform': view[None],
        'full_proj_transform': (view @ projection)[None],
        'camera_center': query[:3, 3][None],
    }


def event_input(events, loader, crop):
    from lib.dataset.utils import events_to_voxel_grid

    events = events.copy()
    xy = events[:2]
    valid = ((xy[0] >= 0) & (xy[0] < loader.W) & (xy[1] >= 0) & (xy[1] < loader.H))
    events = events[:, valid]
    events[3] = np.where(events[3] > 0, 1, -1)
    frame = loader.events_to_frame(events).transpose(2, 0, 1)
    voxel = (events_to_voxel_grid(events.T.copy(), 5, loader.W, loader.H)
             if events.shape[1] else np.zeros((5, loader.H, loader.W), np.float32))
    return tuple(torch.as_tensor(array[:, :, crop].copy(), device='cuda', dtype=torch.float32)[None]
                 for array in (frame, voxel))


def depth_metrics(pred, truth):
    valid = torch.isfinite(truth) & (truth > 0)
    support = (pred > 0) & torch.isfinite(pred) & valid
    result = {'coverage': float(support.sum() / valid.sum()) if valid.any() else None}
    if valid.any():
        p, t = pred[valid], truth[valid]
        result.update(absrel=float(((p - t).abs() / t).mean()),
                      delta1=float((torch.maximum(p/t, t/p.clamp_min(1e-8)) < 1.25).float().mean()))
    return result


def render_window(model, batch, cameras, poses, intrinsic, size=None):
    from lib.renderer import depth2pc, pts2render

    output = model(batch)
    output['target'] = cameras
    for name, pose in zip(('lview', 'rview'), poses):
        output[name]['pts'] = depth2pc(output[name]['depth'], torch.linalg.inv(pose), intrinsic)
    prediction = pts2render(output, [0., 0., 0.])[:, 0].clamp(0, 1)
    if size is not None:
        prediction = F.interpolate(prediction[:, None], size=size, mode='area')[:, 0]
    return prediction, output


def render_metrics(prediction, target, metrics, mask=None):
    psnr, ssim = gray_metrics(prediction, target)
    scores = dict(psnr=float(psnr[0]), ssim=float(ssim[0]),
                  ssim_evggs=float(gray_metrics(prediction, target, padded=True)[1][0]))
    scores.update(metrics(prediction, target))
    if mask is not None and mask.any():
        scores['foreground'] = float(-10 * (prediction-target)[mask].square().mean().clamp_min(1e-12).log10())
    return scores


def ev3ds(args, model, metrics):
    from lib.dataset.Ev3D import ReadEventFromH5

    crop = slice(64, 576)
    names = (args.root / f'{args.split}_scenes.txt').read_text().split()
    if args.limit_scenes:
        names = names[:args.limit_scenes]
    for scene in names:
        loader = ReadEventFromH5(str(args.root), scene)
        starts = range(len(loader)-3) if args.native else range(4, 60, 7)
        for i in list(starts)[:args.count]:
            if args.native:
                item = {k: v.cuda() for k, v in default_collate([loader[i]]).items()}
                cameras = {k: item[k] for k in ('H','W','FovX','FovY','world_view_transform',
                                                 'full_proj_transform','camera_center')}
                sources = (i, i+2)
                target_id = i+1
                poses = (item['lpose'], item['rpose'])
                intrinsic = item['intrinsic']
            else:
                sources = (i, i+4)
                target_id = i+2
                item = {}
                for frame, frame_key, voxel_key in zip(sources, ('leframe', 'reframe'), ('left_voxel', 'right_voxel')):
                    packet = np.load(args.root/'Event'/scene/f'{frame:04}.npy')
                    item[frame_key], item[voxel_key] = event_input(packet, loader, crop)
                poses = tuple(torch.as_tensor(loader.find_pose(loader.npz_files, j), device='cuda', dtype=torch.float32)[None] for j in sources)
                intrinsic = torch.as_tensor(loader.intrinsics.copy(), device='cuda')
                intrinsic[0, 2] = 255.5
                intrinsic[1, 2] = 239.5
                intrinsic = intrinsic[None]
                query = torch.as_tensor(loader.find_pose(loader.npz_files, target_id), device='cuda', dtype=torch.float32)
                cameras = camera_batch(intrinsic[0], query, 480, 512, .99)
            rgb_path = loader.rgb_files[target_id]
            if args.native:
                gray = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2GRAY)
            else:
                with Image.open(rgb_path) as image:
                    gray = np.asarray(image.convert('L'))
            depth = loader.find_depth(loader.npz_files, target_id)
            target = torch.as_tensor(gray[:, crop].copy(), device='cuda')[None].float()/255
            mask = torch.as_tensor(depth[:, crop].copy() > 0, device='cuda')[None]
            report_target = item['cim'][:, 0] if args.native else target
            prediction, output = render_window(model, item, cameras, poses, intrinsic)
            score = render_metrics(prediction, report_target, metrics, mask)
            row = dict(scene=scene, start=i, sources=sources, target=target_id, matched=score)
            if args.native:
                row['raw'] = render_metrics(prediction, target, metrics, mask)
            row['depth'] = [depth_metrics(output[view]['depth'][0, 0],
                            torch.as_tensor(loader.find_depth(loader.npz_files, j)[:, crop].copy(), device='cuda'))
                            for view, j in zip(('lview','rview'), sources)]
            yield row


def mvsec(args, model, metrics):
    import h5py

    from lib.dataset.Ev3D import ReadEventFromH5
    from mvsec_transfer import calibration, flight_windows, write_gray_preview
    from real_geometry import depth_lookup, depth_pair

    # MVSEC's virtual image and pooled grayscale reference are shared with DVLT.
    loader = ReadEventFromH5.__new__(ReadEventFromH5)
    loader.H, loader.W = 480, 640
    k = torch.tensor([[888.8889, 0, 319.5], [0, 888.8889, 239.5], [0, 0, 1]], device='cuda')
    maps = calibration(args.calibration)
    xy, factor = depth_lookup(args.calibration, 480, 640)
    for flight in ('indoor_flying1','indoor_flying2','indoor_flying3'):
        with h5py.File(args.root/f'{flight}_data.hdf5') as data, h5py.File(args.root/f'{flight}_gt.hdf5') as truth:
            image_times = data['davis/left/image_raw_ts'][:]
            gt = truth['davis/left']
            depth_times = gt['depth_image_rect_ts'][:]
            for start, _, (_, target), query, right, raw in flight_windows(
                    args.root, flight, args.count or 24, 4, None, *maps, raw_packets=True,
                    preview=bool(args.preview_dir)):
                item = {}
                item['leframe'], item['left_voxel'] = event_input(raw[0], loader, slice(None))
                item['reframe'], item['right_voxel'] = event_input(raw[1], loader, slice(None))
                camera = camera_batch(k, query, 480, 640, 100.)
                poses = (torch.eye(4, device='cuda')[None], right[None])
                prediction, output = render_window(model, item, camera, poses, k[None], target.shape)
                if args.preview_dir:
                    write_gray_preview(args.preview_dir/f'{flight}_{start}_target.png',target)
                    write_gray_preview(args.preview_dir/f'{flight}_{start}_matched.png',prediction[0])
                scores = render_metrics(prediction, target[None], metrics)
                refs, offsets = depth_pair(gt, image_times, depth_times, start, xy, factor)
                refs[(refs < .1) | (refs > 50)] = torch.nan
                depths = [depth_metrics(output[view]['depth'][0, 0], ref)
                          for view, ref in zip(('lview','rview'), refs)]
                yield dict(flight=flight, start=start, sources=[start,start+4], target=start+2,
                           matched=scores, depth=depths, depth_dt_s=offsets)


@torch.inference_mode()
def main(args):
    if args.preview_dir:
        if args.dataset != 'mvsec':
            raise ValueError('Render previews are available for the shared MVSEC protocol')
        args.preview_dir.mkdir(parents=True,exist_ok=True)
    # Upstream config parses argv at import and resolves its defaults from the checkout.
    sys.path.insert(0, str(args.repository))
    sys.argv = [sys.argv[0]]
    os.chdir(args.repository)
    if 'int' not in np.__dict__:
        np.int = int  # EvGGS voxelization uses the pre-NumPy-1.24 alias.
    from lib.network import EventGaussian

    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    model = EventGaussian().cuda().eval()
    model.load_state_dict(checkpoint['network'], strict=True)
    metrics = PerceptualMetrics('cuda')
    rows = []
    run = ev3ds if args.dataset == 'ev3ds' else mvsec
    for row in run(args, model, metrics):
        rows.append(row)
        if len(rows) == 1 or len(rows) % 20 == 0:
            print(args.dataset, len(rows), row['matched'], flush=True)
    if not rows:
        raise ValueError('No windows evaluated')
    with args.weights.open('rb') as checkpoint_file:
        digest = hashlib.file_digest(checkpoint_file, 'sha256').hexdigest()
    temporary = args.out.with_suffix('.partial')
    temporary.write_text(json.dumps(dict(dataset=args.dataset, native=args.native, split=args.split,
        upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        weights=args.weights.name, sha256=digest,
        weights_step=checkpoint['total_steps'], results=rows), indent=2, allow_nan=False)+'\n')
    temporary.replace(args.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', choices=['ev3ds','mvsec'])
    for name in ('repository','weights','root','out'):
        p.add_argument(name, type=lambda s: Path(s).resolve())
    p.add_argument('--calibration', type=lambda s: Path(s).resolve())
    p.add_argument('--split', choices=['val','test'], default='val')
    p.add_argument('--native', action='store_true', help='Use the official EV3DS two-packet protocol')
    p.add_argument('--count', type=int, help='Windows per scene or flight')
    p.add_argument('--preview-dir', type=lambda s: Path(s).resolve(),
                   help='Score and save the first, middle and last selected window per MVSEC flight')
    p.add_argument('--limit-scenes', type=int)
    args = p.parse_args()
    if args.dataset == 'mvsec' and (args.native or args.calibration is None):
        p.error('MVSEC requires --calibration and uses the shared window protocol')
    main(args)
