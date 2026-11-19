"""Audit frozen DVLT on EV3DS and cache its multiview training features."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch
from torch.nn import functional as F
from dvlt.model.dvlt.model import DVLTModel
from dvlt.common.rays import rays_to_pose


def angle(cosine):
    return cosine.clamp(-1, 1).acos().rad2deg().item()


def reference(root, scene, frame, shape):
    with np.load(root / "Scenes" / scene / f"{frame:04}.npz") as data:
        depth = torch.tensor(data["depth_map"], device="cuda")
    return F.interpolate(depth[None,None], shape, mode="nearest-exact")[0,0]


@torch.inference_mode()
def extract(args):
    torch.manual_seed(0)
    np.random.seed(0)
    config = json.loads((args.assets / "config.json").read_text())
    teacher = DVLTModel(**config)
    teacher.load_state_dict(load_file(args.assets / "dvlt.safetensors"), strict=True)
    teacher.cuda().eval()
    # Both decoders consume the final shared-loop tokens.
    feature = []
    def capture(module, inputs):
        feature.append(inputs[0].detach())
    hook = teacher.depth_decoder.register_forward_pre_hook(capture)
    image_features = []
    def capture_encoder(module, inputs, output):
        image_features.append(output["x_norm_patchtokens"].detach())
    encoder_hook = teacher.patch_embed_encoder.register_forward_hook(capture_encoder)
    rows = []
    cache = []
    encoder_cache = []
    source_depths = []
    for split, count in (("train",12),("val",6)):
        names = (args.root / f"{split}_scenes.txt").read_text().split()[:count]
        for scene in names:
            for i in range(4,60,7):
                frames = (i, i+4)
                images = []
                for frame in frames:
                    path = args.root/"Scenes"/scene/f"{frame:04}.png"
                    with Image.open(path) as image:
                        rgb = image.convert("RGB").resize((448,336),Image.Resampling.BOX)
                        images.append(torch.tensor(np.array(rgb)).permute(2,0,1))
                images = torch.stack(images)[None].cuda().float()/255
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = teacher.forward_inference(images)
                tokens = torch.cat(feature)[:,teacher.patch_start_idx:].float()
                feature.clear()
                if split == "train":
                    grid = tokens[0].T.reshape(1,768,24,32)
                    cache.append(F.avg_pool2d(grid,4)[0].cpu())
                    image_grid = image_features[0][0].T.reshape(1,768,24,32).float()
                    encoder_cache.append(F.avg_pool2d(image_grid,4)[0].cpu())
                    # Source Gaussians are centered on the 8 by 6 grid, in each source frame.
                    depth_map = out["depth"][0,...,0].float()
                    source_depths.append(F.interpolate(depth_map[:,None],(6,8),mode="bilinear",
                                                       align_corners=False)[:,0].cpu())
                image_features.clear()
                rays = out["rays"].float()
                pose, _ = rays_to_pose(rays, torch.ones_like(rays[...,0]),336,448,14)
                pred = torch.linalg.inv(pose[0,0]) @ pose[0,1]
                cv = np.diag([1,-1,-1,1])
                poses = [np.loadtxt(args.root/"Poses"/f"r_{j+1:04}.txt") @ cv for j in frames]
                gt = torch.tensor(np.linalg.inv(poses[0]) @ poses[1],device="cuda",dtype=torch.float32)
                rot = pred[:3,:3].T @ gt[:3,:3]
                row = dict(split=split,scene=scene,frame=i,rotation_error_deg=angle((rot.trace()-1)/2),
                           identity_rotation_error_deg=angle((gt[:3,:3].trace()-1)/2),
                           translation_angle_deg=angle(F.cosine_similarity(pred[:3,3],gt[:3,3],dim=0)))
                ref = torch.stack([reference(args.root,scene,j,(336,448)) for j in frames])
                valid = ref>0
                depth = out["depth"][0,...,0].float()[valid]
                ref = ref[valid]
                scaled = depth*(ref/depth).median()
                row["aligned_abs_rel"] = ((scaled-ref).abs()/ref).mean().item()
                row["median_plane_abs_rel"] = ((ref.median()-ref).abs()/ref).mean().item()
                rows.append(row)
            print(split,scene,flush=True)
    hook.remove()
    encoder_hook.remove()
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"teacher_geometry.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")
    keys = [(r["scene"],r["frame"]) for r in rows if r["split"]=="train"]
    torch.save(dict(keys=keys,depth=torch.stack(source_depths)),args.out/"depth_targets.pt")
    for name, features in [("loop",cache),("encoder",encoder_cache)]:
        torch.save(dict(keys=keys,kind=name,features=torch.stack(features)),args.out/f"{name}_features.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("root","assets","out"):
        parser.add_argument(arg,type=Path)
    extract(parser.parse_args())
