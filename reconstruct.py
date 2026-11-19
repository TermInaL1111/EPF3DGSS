"""Primary inference entry point: two raw event packets -> event DVLT -> Gaussian scene."""
import argparse
from pathlib import Path
import numpy as np
import torch

from event_field import EventField
from event_dvlt import EventDVLT,voxelize


@torch.inference_mode()
def reconstruct(args):
    model=EventDVLT(args.assets).cuda().eval()
    delta=torch.load(args.student,map_location="cpu",weights_only=True)
    expected={name for name,p in model.named_parameters() if p.requires_grad}
    if set(delta)!=expected:
        raise ValueError("Student checkpoint does not contain the complete event adapters and Gaussian head")
    model.load_state_dict(delta,strict=False)
    encoder=EventField(args.f3_repository,args.f3_checkpoint,4)
    packets=[np.load(path) for path in (args.first,args.second)]
    volume=torch.stack([voxelize(packet) for packet in packets])[None].cuda()
    field=torch.stack([encoder(packet).half().float() for packet in packets])[None]
    with torch.autocast("cuda",dtype=torch.bfloat16):
        cloud=model(volume,field)
    attributes={key:cloud[key][0].float().cpu().numpy() for key in
                ("centers","scales","quats","opacity","gray","rotation","translation","source_depth")}
    if not all(np.isfinite(arr).all() for arr in attributes.values()):
        raise ValueError("Nonfinite scene parameters")
    np.savez_compressed(args.output,**attributes)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("assets","student","f3_repository","f3_checkpoint","first","second","output"):
        parser.add_argument(arg,type=Path)
    reconstruct(parser.parse_args())
