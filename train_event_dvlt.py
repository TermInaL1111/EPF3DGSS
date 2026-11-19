"""Train the event DVLT student, then evaluate its Gaussian head on held-out EV3DS objects."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from ev3ds_pose import camera
from event_dvlt import EventDVLT,load_rgb,render,voxelize


def get_inputs(root,fields,scene,i):
    frames = [i,i+4]
    vox = [voxelize(np.load(root/"Event"/scene/f"{frame:04}.npy")) for frame in frames]
    return torch.stack(vox)[None].cuda(),torch.stack([fields[scene,j] for j in frames])[None].float().cuda()


def labels(root,scene,i):
    with Image.open(root/"Scenes"/scene/f"{i+2:04}.png") as im:
        gray = torch.tensor(np.array(im.convert("L")),device="cuda")[None]/255
    with np.load(root/"Scenes"/scene/f"{i+2:04}.npz") as data:
        valid = torch.tensor(data["depth_map"]>0,device="cuda")[None]
    poses = np.linalg.solve(camera(root,scene,i),np.stack([camera(root,scene,j) for j in (i+2,i+4)]))
    return gray,valid,torch.tensor(poses,device="cuda",dtype=torch.float32)


@torch.no_grad()
def rgb_targets(root,keys,assets,geometry=False):
    teacher = load_rgb(assets)
    targets = {}
    features = []
    hook = teacher.depth_decoder.register_forward_pre_hook(lambda module,args: features.append(args[0].detach()))
    for scene,i in keys:
        frames = []
        for j in [i,i+4]:
            with Image.open(root/"Scenes"/scene/f"{j:04}.png") as im:
                frames.append(torch.from_numpy(np.array(im.convert("RGB").resize((224,168),Image.Resampling.BOX))))
        with torch.autocast("cuda",dtype=torch.bfloat16):
            raw = teacher.forward_inference(torch.stack(frames).permute(0,3,1,2)[None].cuda().float()/255)
        targets[scene,i] = {"tokens": features.pop().half().cpu()}
        if geometry:
            targets[scene,i].update({k:v.half().cpu() for k,v in geometry_maps(raw).items()})
    hook.remove()
    return targets


def geometry_maps(raw):
    # Pool at the patch stride to supervise geometry at the transformer's spatial resolution.
    depth = raw["depth"].float().flatten(0,1).permute(0,3,1,2).clamp_min(1e-6).log()
    rays = raw["rays"].float().flatten(0,1).permute(0,3,1,2)
    return {"log_depth": F.avg_pool2d(depth,14), "rays": F.avg_pool2d(rays,14)}


def alignment(tokens,reference,hidden,extra):
    tokens,reference = tokens.float(),reference.float()
    importance = reference.norm(dim=-1)
    importance = importance/importance.amax(1,keepdim=True).clamp_min(1e-6)
    error = 1-F.cosine_similarity(tokens,reference,dim=-1)+importance*(tokens-reference).abs().mean(-1)
    masked = F.pad(hidden.squeeze(-1),(extra,0))
    return error.mean() + .25*(error*masked).sum()/masked.sum().clamp_min(1)


@torch.no_grad()
def evaluate(model,root,fields,keys):
    rows=[]
    for scene,i in keys:
        voxel,field = get_inputs(root,fields,scene,i)
        gray,mask,poses = labels(root,scene,i)
        row=dict(scene=scene,start=i)
        for name,v,f in [("matched",voxel,field),("zero",torch.zeros_like(voxel),torch.zeros_like(field))]:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                cloud=model(v,f)
            pred=render(cloud,poses[:1])
            err=(pred-gray).square()
            rot=cloud["rotation"][0].T @ poses[1,:3,:3]
            row[name]=dict(psnr=float(-10*err.mean().log10()),foreground=float(-10*err[mask].mean().log10()),
                           rotation_deg=float(((rot.trace()-1)/2).clamp(-1,1).acos().rad2deg()),
                           gaussians=int(cloud["centers"].shape[1]))
        rows.append(row)
    return rows


def main(args):
    torch.manual_seed(args.seed)
    fields=torch.load(args.fields,weights_only=True)
    keys=[]
    for split,count in [("train",12),("val",6)]:
        scenes=(args.root/f"{split}_scenes.txt").read_text().split()[:count]
        keys.append([(s,i) for s in scenes for i in range(4,60,7)])
    train,val=keys
    targets=rgb_targets(args.root,train,args.assets,args.geometry_weight>0)
    model=EventDVLT(args.assets).cuda()
    trainables={n:p for n,p in model.named_parameters() if p.requires_grad}
    opt=torch.optim.AdamW([{"params":[p for n,p in trainables.items() if n.startswith("backbone.")],"lr":3e-5},
                           {"params":[p for n,p in trainables.items() if not n.startswith("backbone.")],"lr":3e-4}])
    args.out.mkdir(parents=True,exist_ok=True)
    log=[]
    for step in range(args.steps):
        scene,i=train[int(torch.randint(len(train),(1,)))]
        voxel,field=get_inputs(args.root,fields,scene,i)
        fraction=args.mask_fraction*min(1,(step+1)/max(1,args.warmup))
        with torch.autocast("cuda",dtype=torch.bfloat16):
            if step<args.warmup:
                tokens,pos,hidden=model.geometry(voxel,field,mask_fraction=fraction)
                if args.geometry_weight:
                    raw=model.backbone._decode(tokens,168,224,1,2,pos)
            else:
                cloud=model(voxel,field,mask_fraction=fraction,return_geometry=bool(args.geometry_weight))
                tokens,hidden=cloud["tokens"],cloud["masked"]
        target=targets[scene,i]
        align=alignment(tokens,target["tokens"].cuda(),hidden,model.backbone.patch_start_idx)
        loss=align
        if args.geometry_weight:
            maps=geometry_maps(raw if step<args.warmup else cloud["raw_geometry"])
            loss=loss+args.geometry_weight*sum(F.l1_loss(value,target[key].cuda().float()) for key,value in maps.items())
        if step>=args.warmup:
            gray,mask,poses=labels(args.root,scene,i)
            error=(render(cloud,poses[:1])-gray).abs()
            loss=loss+error.mean()+error[mask].mean()
        opt.zero_grad()
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(trainables.values(),1.,error_if_nonfinite=True)
        opt.step()
        log.append([step,float(align.detach()),float(loss.detach()),float(norm)])
        if step%10==0: print(step,log[-1][1:],flush=True)
    torch.save({n:p.detach().cpu() for n,p in trainables.items()},args.out/"event_dvlt_gs.pt")
    rows=evaluate(model,args.root,fields,val)
    report=dict(seed=args.seed,geometry_weight=args.geometry_weight,mask_fraction=args.mask_fraction,steps=args.steps,warmup=args.warmup,input_hw=[168,224],gs_hw=[480,640],
                loops=model.backbone.inference_steps,trainable_params=sum(p.numel() for p in trainables.values()),
                train=train,validation=val,trace=log,rows=rows)
    (args.out/"scores.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","fields","assets","out"): p.add_argument(name,type=Path)
    p.add_argument("--steps",type=int,default=96)
    p.add_argument("--warmup",type=int,default=32)
    p.add_argument("--seed",type=int,default=10)
    p.add_argument("--geometry-weight",type=float,default=0.)
    p.add_argument("--mask-fraction",type=float,default=.25)
    main(p.parse_args())
