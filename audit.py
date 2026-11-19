"""Measure dependence on event content under EV3DS's shared camera trajectory."""
import argparse
import json
from pathlib import Path
import torch
from model import EventGaussians
from pilot import dataset, foreground


@torch.no_grad()
def main(root, run):
    report = json.loads((run/"scores.json").read_text())
    events, gray, cameras, right, _ = dataset(root,report["validation"],None)
    _, train_gray, *_ = dataset(root,report["train"],None)
    # Every scene uses the same eight sampled time indices.
    template = train_gray.reshape(-1,8,24,32).mean(0).repeat(len(report["validation"]),1,1)
    def score(images):
        return float((-10*(images-gray).square().mean((-1,-2)).log10()).mean())
    records = {"training_scene_mean_image_psnr":score(template),"arms":{}}
    region = foreground(root,report["validation"])
    for weight in report["arms"]:
        net = EventGaussians().cuda().eval()
        net.load_state_dict(torch.load(run/f"student_{weight}.pt",weights_only=True))
        measurements = {}
        for name, packets in [("matched",events),("other_scene_same_time",events.roll(8,0)),
                              ("zeros",torch.zeros_like(events))]:
            scene = net(packets)
            images = net.render(scene,cameras)
            measurements[name] = score(images)
            mse = ((images-gray).square()*region).sum((1,2))/region.sum((1,2)).clamp_min(1)
            measurements[name+"_foreground_psnr"] = float((-10*mse.log10()).mean())
            if name == "matched":
                rotation = scene["rotation"].transpose(1,2) @ right[:,:3,:3]
                cosine = (rotation.diagonal(dim1=1,dim2=2).sum(1)-1)/2
                measurements["relative_rotation_deg"] = float(cosine.clamp(-1,1).acos().rad2deg().mean())
                measurements["relative_translation_l2"] = float((scene["translation"]-right[:,:3,3]).norm(dim=1).mean())
        records["arms"][weight] = measurements
    (run/"audit.json").write_text(json.dumps(records,indent=2)+"\n")
    print(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    main(args.root,args.run)
