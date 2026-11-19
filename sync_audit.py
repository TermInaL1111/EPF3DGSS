"""Audit EV3DS source alignment using v2e timestamps and signed image changes."""
import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def signed_image(x, y, polarity):
    image = np.zeros((480, 640), np.float32)
    np.add.at(image, (y.astype(int), x.astype(int)), np.where(polarity > 0, 1., -1.))
    return image


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a*b)/denom) if denom else None


def audit(root, out):
    rows = []
    scenes = (root / "train_scenes.txt").read_text().split()[:12] + (root / "val_scenes.txt").read_text().split()
    for scene in scenes:
        cap = cv2.VideoCapture(str(root / "RGB_Videos" / f"{scene}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert fps > 0 and count >= 62
        with h5py.File(root / "H5_Files" / scene / "event.h5") as f:
            events = f["events"][:]
        timestamps = events[:, 0]
        assert np.all(timestamps[1:] >= timestamps[:-1])
        for i in sorted({j for start in range(4, 60, 7) for j in (start,start+4)}):
            pngs = []
            for j in range(i-2, i+2):
                png = cv2.imread(str(root / "Scenes" / scene / f"{j:04}.png"))
                pngs.append(cv2.cvtColor(png, cv2.COLOR_BGR2GRAY).astype(np.float32))
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, video = cap.read()
            assert ok
            video_gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY).astype(np.float32)
            video_mae = [float(np.abs(video_gray-img).mean()) for img in pngs[1:]]
            boundaries = np.rint(np.array([i-1,i])*1e6/fps).astype(int)
            lo, hi = timestamps.searchsorted(boundaries, side="right")
            _, x, y, p = events[lo:hi].T
            h5_image = signed_image(x, y, p)
            x, y, t, p = np.load(root / "Event" / scene / f"{i:04}.npy")
            npy_image = signed_image(x, y, p)
            diffs = np.diff(np.log1p(pngs), axis=0)
            rows.append(dict(scene=scene, frame=i, fps=fps, video_frames=count,
                             video_png_mae_prev_same_next=video_mae, h5_bounds_us=boundaries.tolist(),
                             npy_time_range=[float(t.min()),float(t.max())],
                             npy_signed_cosine_prev_same_next=[cosine(npy_image,d) for d in diffs],
                             h5_signed_cosine_prev_same_next=[cosine(h5_image,d) for d in diffs]))
        cap.release()
    out.write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("out",type=Path)
    args = p.parse_args()
    audit(args.root,args.out)
