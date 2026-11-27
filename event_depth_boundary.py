"""Recover event-supported depth discontinuities before lifting points to 3D."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def event_bins(packet, shape, bins=5):
    """Keep event polarities separate and bin the raw timestamps within the packet."""
    height, width = shape
    x, y, t, p = packet
    mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t, p = x[mask], y[mask], t[mask], p[mask]
    counts = np.zeros((2, bins, height, width), dtype=np.float32)
    if t.size:
        time = ((t - t.min()) / max(float(t.max() - t.min()), 1e-12) * bins).astype(int)
        np.add.at(counts, ((p > 0).astype(int), time.clip(max=bins-1), y.astype(int), x.astype(int)), 1)
    return torch.from_numpy(counts)


class EventDepthBoundary(nn.Module):
    """Predict local sample offsets, a log-depth residual and two directed edge logits."""

    def __init__(self, token_width=768):
        super().__init__()
        self.project = nn.Conv2d(token_width, 16, 1)
        self.refine = nn.Sequential(nn.Conv2d(28, 32, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(32, 5, 3, padding=1))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, tokens, depth, confidence, evidence):
        height, width = evidence.shape[-2:]
        size = (height, width)
        coarse = depth.clamp_min(1e-6).log()
        dense = F.interpolate(coarse, size, mode='bilinear', align_corners=False)
        conf = F.interpolate(confidence.clamp_min(0).log1p(), size, mode='bilinear', align_corners=False)
        features = F.interpolate(self.project(tokens), size, mode='bilinear', align_corners=False)
        offsets, residual, edges = self.refine(torch.cat((features, dense, conf, evidence), dim=1)).split((2, 1, 2), 1)
        x = (torch.arange(width, device=depth.device, dtype=depth.dtype) + .5) * (2 / width) - 1
        y = (torch.arange(height, device=depth.device, dtype=depth.dtype) + .5) * (2 / height) - 1
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1)[None]
        # Bound the displacements to one coarse-pixel step along each axis.
        step = depth.new_tensor([2 / depth.shape[-1], 2 / depth.shape[-2]])
        grid = grid + offsets.float().tanh().permute(0, 2, 3, 1) * step
        sampled = F.grid_sample(coarse, grid, padding_mode='border', align_corners=False)
        return sampled + residual.float(), edges.float()


def adjacent(values):
    return (values[..., :, 1:] - values[..., :, :-1],
            values[..., 1:, :] - values[..., :-1, :])


def depth_loss(prediction, edges, reference):
    target = reference.clamp_min(1e-6).log()
    valid = torch.isfinite(reference) & (reference >= .1) & (reference <= 200)
    reference = torch.where(valid, target, 0)
    error = prediction - reference
    centered = error - error[valid].mean().detach()
    depth = centered[valid].abs().mean()
    pair_masks = (valid[..., :, 1:] & valid[..., :, :-1],
                  valid[..., 1:, :] & valid[..., :-1, :])
    classification, discontinuity = [], []
    for pred, true, mask, logits in zip(adjacent(prediction), adjacent(reference), pair_masks,
                                      (edges[:, :1, :, :-1], edges[:, 1:, :-1, :]), strict=True):
        difference = true.abs()
        positive = mask & (difference >= math.log1p(.10))
        negative = mask & (difference <= math.log1p(.02))
        if positive.any():
            discontinuity.append(F.smooth_l1_loss(pred[positive], true[positive], beta=.1))
            classification.append(F.softplus(-logits[positive]).mean())
        if negative.any():
            classification.append(F.softplus(logits[negative]).mean())
    return depth + .5 * sum(discontinuity) / max(1, len(discontinuity)) + .1 * sum(classification) / max(1, len(classification))
