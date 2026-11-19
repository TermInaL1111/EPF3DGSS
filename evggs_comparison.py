"""Match EV3DS scene, source/target IDs, native crop and foreground mask to an archived EvGGS run."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from ev3ds_pose import camera
import torch
from event_field import EventField
from model import EventGaussians
from pilot import packet


@torch.inference_mode()
def compare(args):
    archive = args.evggs_results
    metadata = json.loads((archive/"manifest.json").read_text())
    if metadata["dataset"]!="ev3ds" or metadata["native_resolution"] is not True:
        raise ValueError("Expected native EV3DS baseline outputs")
    rows = [json.loads(line) for line in (archive/"per_sample.jsonl").read_text().splitlines()]
    index = {(r["scene"],r["loader_index"]):r for r in rows if r["nonwrapping"]}
    experiment = json.loads((args.run/"scores.json").read_text())
    names = sorted(set(experiment["validation"]) & {r["scene"] for r in rows})
    assert names and set(names).isdisjoint(experiment["train"])
    net = EventGaussians(36,4,experiment["fusion"]).cuda().eval()
    net.load_state_dict(torch.load(args.run/"student_0.1.pt",weights_only=True))
    field = EventField(args.f3_repository,args.f3_checkpoint,4)
    out = []
    for scene in names:
        for i in range(4,60,7):
            base = index[scene,i]
            assert base["sources"]==[i,i+2] and base["target"]==i+1
            blocks = []
            for j in base["sources"]:
                path = args.root/"Event"/scene/f"{j:04}.npy"
                blocks.append(torch.cat([packet(path,4).cuda(),field(np.load(path)).half().float()]))
            cloud = net(torch.stack(blocks)[None])
            poses = [camera(args.root,scene,j)
                     for j in (i,i+1)]
            target = torch.tensor(np.linalg.solve(*poses),device="cuda").float()[None]
            prediction = net.render(cloud,target,(480,640))[0,:,64:576]
            gray = cv2.imread(str(args.root/"Scenes"/scene/f"{i+1:04}.png"))
            gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)[:,64:576]
            with np.load(args.root/"Scenes"/scene/f"{i+1:04}.npz") as f:
                mask = torch.tensor(f["depth_map"][:,64:576]>0,device="cuda")
            if abs(float(mask.float().mean())-base["mask_fraction"])>1e-6:
                raise ValueError(f"Foreground mask differs at {scene}/{i+1}")
            error = (prediction-torch.tensor(gray,device="cuda")/255).square()
            out.append(dict(scene=scene,sources=base["sources"],target=i+1,
                            epf_raw=float(-10*error.mean().log10()),
                            epf_foreground=float(-10*error[mask].mean().log10()),
                            evggs_raw=base["raw_psnr"],evggs_foreground=base["mask_inside_psnr"]))
    report = dict(baseline_commit=metadata["upstream_commit"],baseline_weight_sha256=metadata["checkpoint_sha256"],
                  archive_sha256=hashlib.sha256((archive/"per_sample.jsonl").read_bytes()).hexdigest(),
                  student=experiment,results=out)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("root","evggs_results","run","f3_repository","f3_checkpoint","out"):
        parser.add_argument(arg,type=Path)
    compare(parser.parse_args())
