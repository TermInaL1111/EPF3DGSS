"""Paired, scene-disjoint EV3DS pilot. Run both arms with identical batches."""
import argparse
import json
from pathlib import Path
import numpy as np
from ev3ds_pose import camera
from PIL import Image
import torch
import torch.nn.functional as F
from safetensors import safe_open
import timm
from model import EventGaussians


def teacher_from_checkpoint(path):
    teacher = timm.create_model("vit_base_patch14_reg4_dinov2", pretrained=False,
                                num_classes=0, dynamic_img_size=True)
    teacher.norm = torch.nn.LayerNorm(768, elementwise_affine=False)
    with safe_open(str(path), framework="pt") as weights:
        prefix = "patch_embed_encoder."
        state = {k.removeprefix(prefix): weights.get_tensor(k) for k in weights.keys()
                 if k.startswith(prefix) and not k.endswith("mask_token")}
    position = state["pos_embed"]
    state["cls_token"] = state["cls_token"] + position[:, :1]
    state["pos_embed"] = position[:, 1:]
    state["reg_token"] = state.pop("register_tokens")
    teacher.load_state_dict(state, strict=True)
    return teacher.cuda().eval()


def packet(path, scale=1):
    return bin_events(np.load(path),scale)


def bin_events(raw,scale=1):
    x,y,t,p = raw
    cell = 20//scale
    out = np.zeros((4,24*scale,32*scale),np.float32)
    if len(t):
        valid = (x>=0)&(x<640)&(y>=0)&(y<480)
        channel = 2*(t>(t.min()+t.max())/2).astype(int)+(p>0)
        np.add.at(out,(channel[valid],y[valid].astype(int)//cell,x[valid].astype(int)//cell),1)
    return torch.from_numpy(np.log1p(out))


def windows(scene_index, mixed=False):
    # Eight disjoint protocol cells per scene. The cyclic offset balances frame IDs.
    for index, start in enumerate(range(4,60,7)):
        cell = (index + scene_index) % 8
        gap = (2,4,8,16)[cell//2] if mixed else 4
        first, second = start, start+gap
        if mixed and cell%2:
            first, second = second, first
        yield first, start+gap//2, second


@torch.no_grad()
def dataset(root, scenes, teacher, mixed=False, fields=None, scale=1):
    rows = []
    for scene_index, scene in enumerate(scenes):
        for first, mid, second in windows(scene_index,mixed):
            ids = [first, second]
            events = torch.stack([packet(root/"Event"/scene/f"{j:04}.npy",scale) for j in ids])
            if fields is not None:
                dense = torch.stack([fields[(scene,j)].float() for j in ids])
                events = torch.cat([events,dense],dim=1)
            def image(j, mode, size):
                with Image.open(root/"Scenes"/scene/f"{j:04}.png") as im:
                    return torch.tensor(np.array(im.convert(mode).resize(size,Image.Resampling.BOX)))/255
            labels = image(mid,"L",(32*scale,24*scale))
            poses = [camera(root,scene,j) for j in [first,mid,second]]
            target, right = torch.tensor(np.linalg.inv(poses[0]) @ np.stack(poses[1:])).float()
            feat = torch.empty(0)
            if teacher is not None:
                rgb = image(first,"RGB",(112,84)).permute(2,0,1).cuda()[None]
                rgb = (rgb-rgb.new_tensor([.485,.456,.406])[None,:,None,None])/rgb.new_tensor([.229,.224,.225])[None,:,None,None]
                tokens = teacher.forward_features(rgb)[:,teacher.num_prefix_tokens:]
                feat = tokens.transpose(1,2).reshape(768,6,8).cpu()
            rows.append((events,labels,target,right,feat))
    return [torch.stack(x).cuda() for x in zip(*rows)]


def foreground(root, scenes, mixed=False, scale=1):
    masks = []
    for scene_index,scene in enumerate(scenes):
        for _,mid,_ in windows(scene_index,mixed):
            with np.load(root/"Scenes"/scene/f"{mid:04}.npz") as archive:
                valid = archive["depth_map"] > 0
            masks.append(torch.tensor(valid.reshape(24*scale,20//scale,32*scale,20//scale).mean((1,3)), dtype=torch.float32))
    return torch.stack(masks).cuda()


def relative_depth_loss(student_depth, target_depth, events, weighting):
    residual = student_depth.log() - target_depth.log()
    confidence = torch.ones_like(residual)
    if weighting == "event":
        counts = events[:,:,:4].expm1().sum(2)
        mass = F.avg_pool2d(counts,4)
        # A log support factor retains supervision in quiet regions.
        confidence = 1 + mass.log1p()
    confidence = confidence / confidence.sum((1,2,3),keepdim=True)
    offset = (confidence*residual).sum((1,2,3),keepdim=True)
    return (confidence*(residual-offset).abs()).sum((1,2,3)).mean()


def run(train,val,weight,steps,seed,mask=None,depth_targets=None,weighting="uniform",fusion_mode="none"):
    torch.manual_seed(seed)
    net = EventGaussians(input_channels=train[0].shape[2], scale=train[0].shape[-1]//32,fusion_mode=fusion_mode).cuda()
    opt = torch.optim.AdamW(net.parameters(),lr=.001)
    for _ in range(steps):
        ids = torch.randint(len(train[0]),(8,),device="cuda")
        events, gray, target, right, teacher = [x[ids] for x in train]
        scene = net(events)
        pred = net.render(scene,target)
        loss = F.l1_loss(pred,gray) + F.l1_loss(scene["rotation"],right[:,:3,:3]) + F.l1_loss(scene["translation"],right[:,:3,3])
        if mask is not None:
            region = mask[ids]
            loss += ((pred-gray).abs()*region).sum((1,2)).div(region.sum((1,2)).clamp_min(1)).mean()
        if weight:
            feature = F.adaptive_avg_pool2d(scene["feature"],teacher.shape[-2:])
            loss += weight*(1-F.cosine_similarity(net.distill(feature),teacher,dim=1).mean())
        if depth_targets is not None:
            loss += .1*relative_depth_loss(scene["source_depth"],depth_targets[ids],events,weighting)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = torch.cat([net.render(net(e),q) for e,q in zip(val[0].split(8),val[2].split(8))])
        psnr = -10*(pred-val[1]).square().mean((1,2)).log10()
        return psnr.view(-1,8).mean(1).tolist(), net


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("teacher",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--steps",type=int,default=400)
    p.add_argument("--seed",type=int,default=10)
    p.add_argument("--foreground",action="store_true")
    p.add_argument("--cached-features",type=Path,help="Training features cached by geometry_teacher.py")
    p.add_argument("--permute-teacher",action="store_true",
                   help="Rotate target features across training scenes at the same frame index")
    p.add_argument("--mixed-spacing",action="store_true",help="Balance gaps and source order in training")
    p.add_argument("--event-fields",type=Path,help="Frozen F³ features from event_field.py")
    p.add_argument("--depth-targets",type=Path)
    p.add_argument("--depth-weighting",choices=("uniform","event"),default="uniform")
    p.add_argument("--scale",type=int,choices=(1,4),default=1)
    p.add_argument("--fusion",choices=("none","geometry","event"),default="none")
    args = p.parse_args()
    fields = torch.load(args.event_fields,weights_only=True) if args.event_fields else None
    if args.scale != 1 and args.depth_targets:
        p.error("The released depth bank samples the coarse Gaussian grid")
    if args.mixed_spacing and args.cached_features:
        p.error("Cached DVLT bank contains only four-frame forward pairs")
    train_names, val_names = [(args.root/f"{s}_scenes.txt").read_text().split()[:n]
                              for s,n in [("train",12),("val",6)]]
    assert set(train_names).isdisjoint(val_names)
    if args.cached_features:
        bank = torch.load(args.cached_features,weights_only=True,map_location="cpu")
        keys = [(name,i) for name in train_names for i in range(4,60,7)]
        if bank["keys"] != keys or bank["features"].shape != (len(keys),768,6,8):
            raise ValueError("Cached multiview features do not match training windows")
        train = dataset(args.root,train_names,None,fields=fields,scale=args.scale)
        train[-1] = bank["features"].cuda()
    else:
        teacher = teacher_from_checkpoint(args.teacher)
        train = dataset(args.root,train_names,teacher,args.mixed_spacing,fields,args.scale)
        del teacher
    if args.permute_teacher:
        train[-1] = train[-1].roll(8,0)
    val = dataset(args.root,val_names,None,fields=fields,scale=args.scale)
    result = dict(train=train_names,validation=val_names,steps=args.steps,seed=args.seed,arms={})
    args.out.mkdir(parents=True,exist_ok=True)
    mask = foreground(args.root,train_names,args.mixed_spacing,args.scale) if args.foreground else None
    result["fusion"] = args.fusion
    result["pose_source"] = "scene_npz"
    result["scale"] = args.scale
    result["foreground_loss"] = args.foreground
    result["event_fields"] = bool(args.event_fields)
    result["mixed_spacing"] = args.mixed_spacing
    result["teacher"] = bank["kind"] if args.cached_features else "image_encoder"
    result["teacher_permuted"] = args.permute_teacher
    depths = None
    if args.depth_targets:
        depth_bank = torch.load(args.depth_targets,weights_only=True)
        expected = [(name,i) for name in train_names for i in range(4,60,7)]
        if args.mixed_spacing or depth_bank["keys"] != expected:
            raise ValueError("Depth targets must match the sampled training pairs")
        depths = depth_bank["depth"].cuda()
    result["depth_weighting"] = args.depth_weighting if depths is not None else None
    for weight in [0.,.1]:
        scores, net = run(train,val,weight,args.steps,args.seed,mask,depths,args.depth_weighting,args.fusion)
        result["arms"][str(weight)] = dict(scene_psnr=scores,mean_psnr=np.mean(scores))
        torch.save(net.state_dict(),args.out/f"student_{weight}.pt")
        print(result["arms"][str(weight)],flush=True)
    (args.out/"scores.json").write_text(json.dumps(result,indent=2)+"\n")
