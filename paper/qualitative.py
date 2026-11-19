"""Display the first validation triplet in each of the first four held-out scenes."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import torch
from pilot import dataset,foreground
from model import EventGaussians


@torch.inference_mode()
def plot(args):
    names = (args.root/"val_scenes.txt").read_text().split()[:4]
    fields = torch.load(args.fields,weights_only=True)
    ev,gt,target,_,_ = dataset(args.root,names,None,fields=fields,scale=4)
    mask = foreground(args.root,names,scale=4)
    model = EventGaussians(36,4,"event").cuda().eval()
    model.load_state_dict(torch.load(args.weights,weights_only=True))
    fig,axes = plt.subplots(len(names),4,figsize=(6.5,5.7),layout="constrained")
    for n,name in enumerate(names):
        idx = 8*n
        inp = ev[idx:idx+1]
        query = target[idx:idx+1]
        correct = model.render(model(inp),query)[0]
        empty = model.render(model(torch.zeros_like(inp)),query)[0]
        y,x = torch.where(mask[idx]>.1)
        x0,x1 = max(0,int(x.min())-4),min(128,int(x.max())+5)
        y0,y1 = max(0,int(y.min())-4),min(96,int(y.max())+5)
        count = inp[0,0,:4].expm1()
        signed = (count[1]+count[3]-count[0]-count[2])
        activity = .5+.5*signed.sign()*signed.abs().log1p()/signed.abs().max().clamp_min(1).log1p()
        for axis,image in zip(axes[n],(activity,gt[idx],correct,empty)):
            axis.imshow(image[y0:y1,x0:x1].cpu(),cmap="gray",vmin=0,vmax=1,interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
        axes[n,0].set_ylabel(name)
    for axis,title in zip(axes[0],("Source events","Target gray","Matched events","Zero input")):
        axis.set_title(title,fontsize=10)
    fig.savefig(args.out,dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root","fields","weights","out"):
        p.add_argument(name,type=Path)
    plot(p.parse_args())
