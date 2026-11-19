"""Extract the published frozen F³ field from native EV3DS packets."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


class EventField:
    def __init__(self, repository, checkpoint):
        self.encoder = torch.hub.load(str(repository), "f3", source="local",
                                      name="1280x720x20_patchff_ds1_small", compile=False,
                                      return_feat=True, return_logits=False, single_batch_mode=True)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        self.encoder.load_state_dict({k.removeprefix("_orig_mod."):v for k,v in weights.items()}, strict=True)
        self.encoder.cuda().eval()
        # Use a fixed corner reference for channel centering. Empty packets map to zero.
        self.reference = self._encode(torch.zeros(1,4,device="cuda"))

    @torch.inference_mode()
    def _encode(self, events):
        # Upstream returns (batch, x, y, channels). Undo center padding before pooling.
        field = self.encoder(events).permute(0,3,2,1)
        return F.avg_pool2d(field[:,:,120:600,320:960],20)[0]

    @torch.inference_mode()
    def __call__(self, raw):
        x,y,t,p = raw
        keep = (x>=0)&(x<640)&(y>=0)&(y<480)
        if not np.any(keep):
            return torch.zeros_like(self.reference)
        t = t[keep]
        age = (t.max()-t)/max(float(t.max()-t.min()),1e-12)
        packet = np.stack([(x[keep]+320)/1280,(y[keep]+120)/720,
                           np.minimum(19,np.floor(age*20))/20,p[keep]>0],axis=1)
        field = self._encode(torch.from_numpy(packet).float().cuda())
        return field-self.reference


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("root","repository","checkpoint","out"):
        p.add_argument(arg,type=Path)
    args = p.parse_args()
    extract = EventField(args.repository,args.checkpoint)
    bank = {}
    for split,count in (("train",12),("val",6)):
        names = (args.root/f"{split}_scenes.txt").read_text().split()[:count]
        for scene in names:
            for i in range(4,60,7):
                for frame in (i,i+4):
                    raw = np.load(args.root/"Event"/scene/f"{frame:04}.npy")
                    bank[(scene,frame)] = extract(raw).half().cpu()
            print(split,scene,flush=True)
    torch.save(bank,args.out)
