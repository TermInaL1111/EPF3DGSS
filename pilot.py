"""Paired, scene-disjoint EV3DS pilot. Run both arms with identical batches."""
import argparse
import json
from pathlib import Path
import numpy as np
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


def packet(path):
    x,y,t,p = np.load(path)
    out = np.zeros((4,24,32),np.float32)
    if len(t):
        valid = (x>=0)&(x<640)&(y>=0)&(y<480)
        channel = 2*(t>(t.min()+t.max())/2).astype(int)+(p>0)
        np.add.at(out,(channel[valid],y[valid].astype(int)//20,x[valid].astype(int)//20),1)
    return torch.from_numpy(np.log1p(out))


@torch.no_grad()
def dataset(root, scenes, teacher):
    rows = []
    cv = np.diag([1,-1,-1,1])
    for scene in scenes:
        for i in range(4, 60, 7):
            ids = [i,i+4]
            events = torch.stack([packet(root/"Event"/scene/f"{j:04}.npy") for j in ids])
            def image(j, mode, size):
                with Image.open(root/"Scenes"/scene/f"{j:04}.png") as im:
                    return torch.tensor(np.array(im.convert(mode).resize(size,Image.Resampling.BOX)))/255
            labels = image(i+2,"L",(32,24))
            poses = [np.loadtxt(root/"Poses"/f"r_{j+1:04}.txt") @ cv for j in [i,i+2,i+4]]
            target, right = torch.tensor(np.linalg.inv(poses[0]) @ np.stack(poses[1:])).float()
            feat = torch.empty(0)
            if teacher is not None:
                rgb = image(i,"RGB",(112,84)).permute(2,0,1).cuda()[None]
                rgb = (rgb-rgb.new_tensor([.485,.456,.406])[None,:,None,None])/rgb.new_tensor([.229,.224,.225])[None,:,None,None]
                tokens = teacher.forward_features(rgb)[:,teacher.num_prefix_tokens:]
                feat = tokens.transpose(1,2).reshape(768,6,8).cpu()
            rows.append((events,labels,target,right,feat))
    return [torch.stack(x).cuda() for x in zip(*rows)]


def run(train,val,weight,steps,seed):
    torch.manual_seed(seed)
    net = EventGaussians().cuda()
    opt = torch.optim.AdamW(net.parameters(),lr=.001)
    for _ in range(steps):
        ids = torch.randint(len(train[0]),(8,),device="cuda")
        events, gray, target, right, teacher = [x[ids] for x in train]
        scene = net(events)
        pred = net.render(scene,target)
        loss = F.l1_loss(pred,gray) + F.l1_loss(scene["rotation"],right[:,:3,:3]) + F.l1_loss(scene["translation"],right[:,:3,3])
        if weight:
            loss += weight*(1-F.cosine_similarity(net.distill(scene["feature"]),teacher,dim=1).mean())
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
    args = p.parse_args()
    train_names, val_names = [(args.root/f"{s}_scenes.txt").read_text().split()[:n]
                              for s,n in [("train",12),("val",6)]]
    assert set(train_names).isdisjoint(val_names)
    teacher = teacher_from_checkpoint(args.teacher)
    train = dataset(args.root,train_names,teacher)
    del teacher
    val = dataset(args.root,val_names,None)
    result = dict(train=train_names,validation=val_names,steps=args.steps,seed=args.seed,arms={})
    args.out.mkdir(parents=True,exist_ok=True)
    for weight in [0.,.1]:
        scores, net = run(train,val,weight,args.steps,args.seed)
        result["arms"][str(weight)] = dict(scene_psnr=scores,mean_psnr=np.mean(scores))
        torch.save(net.state_dict(),args.out/f"student_{weight}.pt")
        print(result["arms"][str(weight)],flush=True)
    (args.out/"scores.json").write_text(json.dumps(result,indent=2)+"\n")
