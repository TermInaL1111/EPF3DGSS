"""Paired native-resolution EV3DS reconstruction and resource measurements."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ev3ds_pose import camera
from model import EventGaussians
from pixel_gaussians import PixelGaussians
from pilot import packet


def example(root,scene,i,fields):
    ids = (i,i+4)
    coarse = torch.stack([torch.cat([packet(root/"Event"/scene/f"{j:04}.npy",4),
                                    fields[scene,j].float()]) for j in ids])[None].cuda()
    native = torch.stack([packet(root/"Event"/scene/f"{j:04}.npy",20) for j in ids])[None].cuda()
    with Image.open(root/"Scenes"/scene/f"{i+2:04}.png") as im:
        image = torch.tensor(np.array(im.convert("L")),device="cuda")[None]/255
    with np.load(root/"Scenes"/scene/f"{i+2:04}.npz") as data:
        mask = torch.tensor(data["depth_map"]>0,device="cuda")[None]
    poses = np.linalg.solve(camera(root,scene,i),np.stack([camera(root,scene,j) for j in (i+2,i+4)]))
    query,right = torch.tensor(poses,device="cuda",dtype=torch.float32)
    return coarse,native,image,mask,query[None],right[None]


def main(args):
    torch.manual_seed(10)
    fields = torch.load(args.fields,weights_only=True)
    bank = torch.load(args.teacher,weights_only=True)
    train_names,val_names = [(args.root/f"{s}_scenes.txt").read_text().split()[:n]
                             for s,n in [("train",12),("val",6)]]
    assert bank["keys"]==[(s,i) for s in train_names for i in range(4,60,7)]
    weight = torch.load(args.warmstart,weights_only=True)
    coarse_model = EventGaussians(36,4).cuda().eval()
    coarse_model.load_state_dict(weight)
    model = PixelGaussians().cuda()
    model.load_state_dict({**model.state_dict(),**weight},strict=True)
    if args.coarse_control:
        model = EventGaussians(36,4).cuda()
        model.load_state_dict(weight)
    opt = torch.optim.AdamW(model.parameters(),lr=.0002)
    args.out.mkdir(parents=True,exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    sampler = torch.Generator().manual_seed(10)
    for step in range(args.steps):
        idx = int(torch.randint(len(bank["keys"]),(1,),generator=sampler))
        scene,i = bank["keys"][idx]
        events,native,target,mask,q,right = example(args.root,scene,i,fields)
        cloud = model(events) if args.coarse_control else model(events,native)
        pred = model.render(cloud,q,(480,640))
        error = (pred-target).abs()
        feature = F.avg_pool2d(cloud["feature"],4)
        teacher = bank["features"][idx:idx+1].cuda()
        loss = error.mean()+error[mask].mean()+.1*(1-F.cosine_similarity(model.distill(feature),teacher,dim=1).mean())
        loss += F.l1_loss(cloud["rotation"],right[:,:3,:3])+F.l1_loss(cloud["translation"],right[:,:3,3])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step%50==0:
            print(step,float(loss.detach()),flush=True)
        del cloud,pred,loss
    peak = torch.cuda.max_memory_allocated()/2**30
    torch.save(model.state_dict(),args.out/("coarse_student.pt" if args.coarse_control else "pixel_student.pt"))
    rows = []
    with torch.inference_mode():
        for scene in val_names:
            for i in range(4,60,7):
                events,native,target,mask,q,_ = example(args.root,scene,i,fields)
                row = dict(scene=scene,start=i)
                for name,net in [("coarse",coarse_model),("coarse_tuned" if args.coarse_control else "pixel",model)]:
                    cloud = net(events) if name.startswith("coarse") else net(events,native)
                    pred = net.render(cloud,q,(480,640))
                    err = (pred-target).square()
                    row[name] = dict(count=cloud["centers"].shape[1],psnr=float(-10*err.mean().log10()),
                                     foreground=float(-10*err[mask].mean().log10()))
                rows.append(row)
    (args.out/"scores.json").write_text(json.dumps(dict(steps=args.steps,train=train_names,validation=val_names,
             peak_allocated_gib=peak,batch=1,seed=10,rows=rows),indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root","fields","teacher","warmstart","out"):
        p.add_argument(name,type=Path)
    p.add_argument("--steps",type=int,default=200)
    p.add_argument("--coarse-control",action="store_true")
    main(p.parse_args())
