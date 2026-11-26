"""Primary inference entry point: two raw event packets -> event DVLT -> Gaussian scene."""
import argparse
from pathlib import Path
import numpy as np
import torch

from event_field import EventField
from event_dvlt import EventDVLT,load_scene_weights,voxelize


@torch.inference_mode()
def reconstruct(args):
    model=EventDVLT(args.assets).cuda().eval()
    delta=torch.load(args.student,map_location="cpu",weights_only=True)
    steps,use_f3=load_scene_weights(model,delta)
    packets=[np.load(path,allow_pickle=False) for path in (args.first,args.second)]
    for packet in packets:
        if packet.ndim!=2 or packet.shape[0]!=4 or not np.isfinite(packet).all():
            raise ValueError("Each event packet must be a finite [4,N] array of x, y, timestamp and polarity")
    volume=torch.stack([voxelize(packet) for packet in packets])[None].cuda()
    if use_f3:
        encoder=EventField(args.f3_repository,args.f3_checkpoint,4)
        field=torch.stack([encoder(packet).half().float() for packet in packets])[None]
    else:
        field=volume.new_zeros((1,2,32,96,128))
    with torch.autocast("cuda",dtype=torch.bfloat16):
        cloud=model(volume,field,steps=steps)
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
