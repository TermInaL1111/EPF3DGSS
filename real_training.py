"""Paired DVLT supervision on real DAVIS packets, holding out an entire flight."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from event_field import EventField
from mvsec_transfer import calibration, flight_windows
from pilot import run, teacher_from_checkpoint


@torch.no_grad()
def collect(args, flights, count, field, maps, teacher=None):
    rows = []
    frame_ids = []
    for flight in flights:
        for start,packet,images,query,right in flight_windows(args.root,flight,count,4,field,*maps):
            token = torch.empty(0,device="cuda")
            if teacher is not None:
                # DAVIS supplies grayscale APS frames. Replicate the left source into three channels.
                image = F.interpolate(images[0][None,None],(84,112),mode="area").expand(-1,3,-1,-1)
                mean = image.new_tensor([.485,.456,.406])[None,:,None,None]
                std = image.new_tensor([.229,.224,.225])[None,:,None,None]
                feature = teacher.forward_features((image-mean)/std)[:,teacher.num_prefix_tokens:]
                token = feature.transpose(1,2).reshape(768,6,8)
            rows.append((packet,images[1],query,right,token))
            frame_ids.append([flight,start])
    return [torch.stack(col) for col in zip(*rows)],frame_ids


@torch.no_grad()
def evaluate(net,validation):
    events,gray,query,right,_ = validation
    scores = {}
    for name,input_events in (("matched",events),("zero",torch.zeros_like(events)),
                              ("wrong_time",events.roll(len(events)//2,0))):
        rows = []
        for inp,ref,pose,gt in zip(input_events.split(8),gray.split(8),query.split(8),right.split(8)):
            scene = net(inp)
            prediction = net.render(scene,pose)
            psnr = -10*(prediction-ref).square().mean((1,2)).log10()
            rotation = scene["rotation"].transpose(1,2) @ gt[:,:3,:3]
            angle = ((rotation.diagonal(dim1=1,dim2=2).sum(1)-1)/2).clamp(-1,1).acos().rad2deg()
            rows.extend(torch.stack([psnr,angle],1).tolist())
        scores[name] = rows
    return scores


def main(args):
    maps = calibration(args.calibration)
    field = EventField(args.f3_repository,args.f3_checkpoint,4)
    teacher = teacher_from_checkpoint(args.teacher)
    train_flights = [f"indoor_flying{i}" for i in (1,2,3) if i != args.held_out]
    train,train_ids = collect(args,train_flights,64,field,maps,teacher)
    del teacher
    val,val_ids = collect(args,[f"indoor_flying{args.held_out}"],32,field,maps)
    del field
    original = train[-1]
    report = dict(train=train_ids,validation=val_ids,steps=args.steps,columns=["psnr","rotation_deg"],arms={})
    args.out.mkdir(parents=True,exist_ok=True)
    for seed in (10,11,12):
        for name,weight in (("none",0.),("aligned",.1),("wrong_training_time",.1)):
            # Shuffle within each training flight to preserve its marginal feature distribution.
            train[-1] = original if name!="wrong_training_time" else original.reshape(2,64,768,6,8).roll(32,1).flatten(0,1)
            _,net = run(train,val,weight,args.steps,seed)
            key = f"{name}_{seed}"
            report["arms"][key] = evaluate(net,val)
            torch.save(net.state_dict(),args.out/f"{key}.pt")
            (args.out/"scores.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
            print(key,flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root","calibration","f3_repository","f3_checkpoint","teacher","out"):
        p.add_argument(name,type=Path)
    p.add_argument("--steps",type=int,default=400)
    p.add_argument("--held-out",type=int,choices=(1,2,3),default=3)
    main(p.parse_args())
