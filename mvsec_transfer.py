"""Zero-shot motion and rendering on real MVSEC indoor flights, with fixed EV3DS field of view."""
import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch
import yaml
from model import EventGaussians
from pilot import bin_events


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
    virtual = np.array([[888.8889,0,319.5],[0,888.8889,239.5],[0,0,1.]])
    assert cam["distortion_model"] == "equidistant"
    y,x = np.indices(cam["resolution"][::-1])
    xy = np.stack([x,y],-1).astype(float).reshape(-1,1,2)
    mapping = cv2.fisheye.undistortPoints(xy,native,dist,P=virtual).reshape(*x.shape,2)
    inverse = cv2.fisheye.initUndistortRectifyMap(native,dist,np.eye(3),virtual,(640,480),cv2.CV_32FC1)
    return mapping,inverse


@torch.inference_mode()
def evaluate(args):
    mapping,inverse = calibration(args.calibration)
    weights = torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    model = EventGaussians()
    model.load_state_dict(weights)
    model.cuda().eval()
    rows = []
    for sequence in ("indoor_flying1","indoor_flying2","indoor_flying3"):
        with h5py.File(args.root/f"{sequence}_data.hdf5") as data, h5py.File(args.root/f"{sequence}_gt.hdf5") as truth:
            cam = data["davis/left"]
            gt = truth["davis/left"]
            times = cam["image_raw_ts"][:]
            pose_times = gt["pose_ts"][:]
            poses = gt["pose"][:]
            interp = Slerp(pose_times,Rotation.from_matrix(poses[:,:3,:3]))
            valid = np.nonzero((times[1:-4]>=pose_times[0]) & (times[5:]<=pose_times[-1]))[0]+1
            for start in valid[np.linspace(0,len(valid)-1,24,dtype=int)]:
                frame_ids = np.array([start,start+2,start+4])
                at = times[frame_ids]
                rigid = np.repeat(np.eye(4)[None],3,axis=0)
                rigid[:,:3,:3] = interp(at).as_matrix()
                rigid[:,:3,3] = np.column_stack([np.interp(at,pose_times,poses[:,j,3]) for j in range(3)])
                query,right = torch.tensor(np.linalg.inv(rigid[0]) @ rigid[1:],device="cuda").float()
                packets = []
                for frame in frame_ids[[0,2]]:
                    events = cam["events"]
                    block = events[lower_bound_after(events,times[frame-1]):lower_bound_after(events,times[frame])]
                    xy = mapping[block[:,1].astype(int),block[:,0].astype(int)]
                    packets.append(bin_events(np.column_stack([xy,block[:,2:]]).T))
                label = []
                for frame in frame_ids[:2]:
                    full = cv2.remap(cam["image_raw"][frame],*inverse,cv2.INTER_LINEAR)
                    label.append(torch.tensor(cv2.resize(full,(32,24),interpolation=cv2.INTER_AREA),device="cuda")/255)
                def psnr(image):
                    return float(-10*(image-label[1]).square().mean().log10())
                packet = torch.stack(packets)[None].cuda()
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
    evaluate(p.parse_args())
