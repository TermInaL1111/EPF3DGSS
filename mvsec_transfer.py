"""Zero-shot motion and rendering on real MVSEC indoor flights, with fixed EV3DS field of view."""
import argparse

import json
from math import isqrt
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch
import yaml
from model import EventGaussians
from pilot import bin_events
from event_field import EventField


# OpenCV rectification uses integer pixel centers. Both Gaussian renderers use
# a half-pixel offset when mapping camera coordinates to image array indices.
MVSEC_VIRTUAL_K = ((888.8889, 0, 319.5), (0, 888.8889, 239.5), (0, 0, 1))
MVSEC_RASTER_K = ((888.8889, 0, 320.0), (0, 888.8889, 240.0), (0, 0, 1))


def lower_bound_after(events,t):
    lo, hi = 0,len(events)
    while lo < hi:
        mid = (lo+hi)//2
        if events[mid,2] <= t:
            lo = mid+1
        else:
            hi = mid
    return lo


def calibration(path):
    cam = yaml.safe_load(path.read_text())["cam0"]
    fx,fy,cx,cy = cam["intrinsics"]
    native = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    dist = np.array(cam["distortion_coeffs"])
    virtual = np.array(MVSEC_VIRTUAL_K)
    assert cam["distortion_model"] == "equidistant"
    y,x = np.indices(cam["resolution"][::-1])
    xy = np.stack([x,y],-1).astype(float).reshape(-1,1,2)
    mapping = cv2.fisheye.undistortPoints(xy,native,dist,P=virtual).reshape(*x.shape,2)
    inverse = cv2.fisheye.initUndistortRectifyMap(native,dist,np.eye(3),virtual,(640,480),cv2.CV_32FC1)
    return mapping,inverse


def write_gray_preview(path, image):
    pixels = image.detach().clamp(0,1).mul(255).round().byte().cpu().numpy()
    if not cv2.imwrite(str(path),pixels):
        raise OSError(f'Could not save {path}')


def flight_windows(root,sequence,count,scale,field,mapping,inverse,raw_packets=False,preview=False):
    with h5py.File(root/f"{sequence}_data.hdf5") as data, h5py.File(root/f"{sequence}_gt.hdf5") as truth:
        cam = data["davis/left"]
        gt = truth["davis/left"]
        times = cam["image_raw_ts"][:]
        pose_times = gt["pose_ts"][:]
        poses = gt["pose"][:]
        interp = Slerp(pose_times,Rotation.from_matrix(poses[:,:3,:3]))
        valid = np.nonzero((times[1:-4]>=pose_times[0]) & (times[5:]<=pose_times[-1]))[0]+1
        sampled = valid[np.linspace(0,len(valid)-1,count,dtype=int)]
        if preview:
            sampled = sampled[np.unique([0,(len(sampled)-1)//2,len(sampled)-1])]
        for start in sampled:
            frame_ids = np.array([start,start+2,start+4])
            at = times[frame_ids]
            rigid = np.repeat(np.eye(4)[None],3,axis=0)
            rigid[:,:3,:3] = interp(at).as_matrix()
            rigid[:,:3,3] = np.column_stack([np.interp(at,pose_times,poses[:,j,3]) for j in range(3)])
            query,right = torch.tensor(np.linalg.inv(rigid[0]) @ rigid[1:],device="cuda").float()
            packets = []
            raw_pair = []
            for frame in frame_ids[[0,2]]:
                events = cam["events"]
                block = events[lower_bound_after(events,times[frame-1]):lower_bound_after(events,times[frame])]
                xy = mapping[block[:,1].astype(int),block[:,0].astype(int)]
                raw = np.column_stack([xy,block[:,2:]]).T
                if raw_packets:
                    raw_pair.append(raw)
                counts = bin_events(raw,scale).cuda()
                if field is not None:
                    counts = torch.cat([counts,field(raw).half().float()],dim=0)
                packets.append(counts)
            label = []
            for frame in frame_ids[:2]:
                full = cv2.remap(cam["image_raw"][frame],*inverse,cv2.INTER_LINEAR)
                label.append(torch.tensor(cv2.resize(full,(32*scale,24*scale),interpolation=cv2.INTER_AREA),device="cuda")/255)
            result = int(start),torch.stack(packets),label,query,right
            yield (*result,raw_pair) if raw_packets else result


@torch.inference_mode()
def evaluate(args):
    mapping,inverse = calibration(args.calibration)
    weights = torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    fusion = ("none","geometry","event")[int(weights.get("fusion_id",0))]
    scale = isqrt(len(weights["rays"])//48)
    channels = weights["encoder.0.weight"].shape[1]
    field = None
    if channels == 36:
        if not args.f3_repository or not args.f3_checkpoint:
            raise ValueError("F³ weights and repository are required by this student")
        field = EventField(args.f3_repository,args.f3_checkpoint,scale)
    model = EventGaussians(channels,scale,fusion)
    model.load_state_dict(weights)
    model.cuda().eval()
    rows = []
    for sequence in ("indoor_flying1","indoor_flying2","indoor_flying3"):
        for start,packet,label,query,right in flight_windows(args.root,sequence,24,scale,field,mapping,inverse):
            def psnr(image):
                return float(-10*(image-label[1]).square().mean().log10())
            packet = packet[None]
            row = dict(sequence=sequence,start=int(start),source_copy_psnr=psnr(label[0]))
            for name,input_packet in [("matched",packet),("zero",torch.zeros_like(packet))]:
                scene = model(input_packet)
                row[name+"_psnr"] = psnr(model.render(scene,query[None]))
                rot = scene["rotation"][0].T @ right[:3,:3]
                row[name+"_rotation_deg"] = float(((rot.trace()-1)/2).clamp(-1,1).acos().rad2deg())
            row["identity_rotation_deg"] = float(((right[:3,:3].trace()-1)/2).clamp(-1,1).acos().rad2deg())
            rows.append(row)
    args.out.write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root","calibration","checkpoint","out"):
        p.add_argument(name,type=Path)
    p.add_argument("--f3-repository",type=Path)
    p.add_argument("--f3-checkpoint",type=Path)
    evaluate(p.parse_args())
