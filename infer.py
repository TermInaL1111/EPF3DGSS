"""Reconstruct a grayscale Gaussian scene from two calibrated EV3DS event packets."""
import argparse

from math import isqrt
from pathlib import Path
import numpy as np
import torch
from pilot import packet
from model import EventGaussians
from event_field import EventField


@torch.inference_mode()
def reconstruct(args):
    weights = torch.load(args.weights,map_location="cpu",weights_only=True)
    channels = weights["encoder.0.weight"].shape[1]
    fusion = ("none","geometry","event")[int(weights.get("fusion_id",0))]
    scale = isqrt(len(weights["rays"])//48)
    net = EventGaussians(input_channels=channels,scale=scale,fusion_mode=fusion)
    net.load_state_dict(weights)
    net.cuda().eval()
    paths = (args.first,args.second)
    events = torch.stack([packet(path,scale) for path in paths]).cuda()
    if channels == 36:
        if not (args.f3_repository and args.f3_checkpoint):
            raise ValueError("This checkpoint requires the F³ repository and weights")
        encode = EventField(args.f3_repository,args.f3_checkpoint,scale)
        fields = torch.stack([encode(np.load(path)).half().float() for path in paths])
        events = torch.cat([events,fields],1)
    scene = net(events[None])
    np.savez_compressed(args.output,**{key:value[0].cpu().numpy() for key,value in scene.items()
                                       if key != "feature"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("weights","first","second","output"):
        p.add_argument(name,type=Path)
    p.add_argument("--f3-repository",type=Path)
    p.add_argument("--f3-checkpoint",type=Path)
    reconstruct(p.parse_args())
