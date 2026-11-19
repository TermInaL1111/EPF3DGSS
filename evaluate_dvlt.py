"""Evaluate event conditioning across the complete EV3DS validation split."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from event_dvlt import EventDVLT, render, voxelize
from event_field import EventField
from train_event_dvlt import labels


@torch.inference_mode()
def evaluate(args):
    torch.manual_seed(10)
    models = {}
    for name, path in (("distilled", args.distilled), ("undistilled", args.undistilled)):
        model = EventDVLT(args.assets).cuda().eval()
        delta = torch.load(path, weights_only=True)
        assert set(delta) == {key for key, p in model.named_parameters() if p.requires_grad}
        model.load_state_dict(delta, strict=False)
        models[name] = model
    field = EventField(args.f3_repository, args.assets / "f3.pth", 4)
    scenes = (args.root / "val_scenes.txt").read_text().split()
    figure, axes = plt.subplots(6, 5, figsize=(9, 7), layout="constrained")
    results = []
    for start in range(4, 60, 7):
        inputs = []
        for scene in scenes:
            packets = [np.load(args.root / "Event" / scene / f"{i:04}.npy") for i in (start, start+4)]
            volume = torch.stack([voxelize(raw) for raw in packets])[None].cuda()
            features = torch.stack([field(raw).half().float() for raw in packets])[None]
            inputs.append((volume, features))
        for index, scene in enumerate(scenes):
            target, mask, pose = labels(args.root, scene, start)
            voxel, feat = inputs[index]
            row = dict(scene=scene, start=start, donor=scenes[(index+1) % len(scenes)])
            images = {}
            for name, model in models.items():
                row[name] = {}
                for condition, (v, f) in (
                    ("matched", (voxel, feat)),
                    ("shuffled", inputs[(index+1) % len(scenes)]),
                    ("zero", (torch.zeros_like(voxel), torch.zeros_like(feat))),
                ):
                    torch.manual_seed(10)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        cloud = model(v, f)
                    prediction = render(cloud, pose[:1])
                    err = (prediction - target).square()
                    row[name][condition] = dict(psnr=float(-10*err.mean().log10()),
                                                 foreground=float(-10*err[mask].mean().log10()))
                    if index < 6 and start == 4:
                        images[name, condition] = prediction[0]
            if index < 6 and start == 4:
                y, x = torch.where(mask[0])
                region = (slice(max(0,int(y.min())-8),int(y.max())+9),
                          slice(max(0,int(x.min())-8),int(x.max())+9))
                for axis, img in zip(axes[index], [target[0], images["distilled","matched"],
                                                   images["undistilled","matched"], images["distilled","shuffled"],
                                                   images["distilled","zero"]]):
                    axis.imshow(img[region].cpu(), cmap="gray", vmin=0, vmax=1)
                    axis.set_xticks([])
                    axis.set_yticks([])
                axes[index,0].set_ylabel(scene)
            results.append(row)
    for axis, title in zip(axes[0], ["Target", "Distilled", "No distillation", "Shuffled events", "Empty events"]):
        axis.set_title(title, fontsize=10)
    figure.savefig(args.figure, dpi=160)
    plt.close(figure)
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("root", "assets", "f3_repository", "distilled", "undistilled", "out", "figure"):
        p.add_argument(arg, type=Path)
    evaluate(p.parse_args())
