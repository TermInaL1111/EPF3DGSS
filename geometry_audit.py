"""Compare held-out EV3DS geometry for the frozen teacher and trained event student."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from scipy.stats import spearmanr

from dvlt.common.rays import rays_to_pose
from event_dvlt import EventDVLT, load_rgb
from ev3ds_pose import camera
from train_event_dvlt import get_inputs


@torch.inference_mode()
def audit(args):
    teacher = load_rgb(args.assets)
    student = EventDVLT(args.assets).eval().cuda()
    delta = torch.load(args.student, weights_only=True)
    assert set(delta) == {name for name, p in student.named_parameters() if p.requires_grad}
    student.load_state_dict(delta, strict=False)
    fields = torch.load(args.fields, weights_only=True)
    rows = []
    for scene, start in json.loads(args.report.read_text())["validation"]:
        reference = args.root / "Scenes" / scene
        frames, depths = [], []
        for i in (start, start + 4):
            with Image.open(reference / f"{i:04}.png") as image:
                frames.append(np.asarray(image.convert("RGB").resize((224, 168), Image.Resampling.BOX)))
            with np.load(reference / f"{i:04}.npz") as archive:
                depths.append(archive["depth_map"].squeeze())
        rgb = torch.tensor(np.stack(frames), device="cuda").permute(0, 3, 1, 2)[None].float() / 255
        voxels, field = get_inputs(args.root, fields, scene, start)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            teacher_raw = teacher.forward_inference(rgb)
            tokens, pos, _ = student.geometry(voxels, field)
            student_raw = student.backbone._decode(tokens, 168, 224, 1, 2, pos)
        truth = np.linalg.solve(camera(args.root, scene, start), camera(args.root, scene, start + 4))
        row = dict(scene=scene, start=start)
        for name, raw in (("rgb_teacher", teacher_raw), ("event_student", student_raw)):
            rays = raw["rays"].float()
            torch.manual_seed(101)
            poses, _ = rays_to_pose(rays, torch.ones_like(rays[..., 0]), 168, 224, 14)
            relative = torch.linalg.solve(poses[0, 0], poses[0, 1]).cpu().numpy()
            cos = (np.trace(truth[:3, :3].T @ relative[:3, :3]) - 1) / 2
            pred_depth = raw["depth"][0, ..., 0].float().cpu().numpy()
            correlations = []
            for gt, pred in zip(depths, pred_depth):
                gt = np.asarray(Image.fromarray(gt.astype(np.float32)).resize((224, 168), Image.Resampling.NEAREST))
                valid = np.isfinite(gt) & (gt > 0)
                rho = spearmanr(gt[valid], pred[valid]).statistic
                correlations.append(float(rho) if np.isfinite(rho) else None)
            row[name] = dict(rotation_deg=float(np.rad2deg(np.arccos(np.clip(cos, -1, 1)))),
                             foreground_depth_spearman=correlations)
        rows.append(row)
    args.out.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("root", "assets", "fields", "student", "report", "out"):
        parser.add_argument(arg, type=Path)
    audit(parser.parse_args())
