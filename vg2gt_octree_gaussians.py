"""VG^2GT-style self-splitting readout on the multi-scale octree leaves.

The octree head already selects leaves by point and view support, which is
important for sparse event depth fields.  This variant keeps that selection and
adds bounded centre residuals and several bounded-scale Gaussians per leaf.
"""

import torch
from torch import nn
from torch.nn import functional as F

from voxel_gaussians import VoxelGaussianHead


class SelfSplittingOctreeHead(VoxelGaussianHead):
    def __init__(self, splits=2, max_scale_leaf=2., prune=0.):
        super().__init__(octree=True)
        self.splits = splits
        self.max_scale_leaf = max_scale_leaf
        self.prune = prune
        self.attributes = nn.Sequential(
            nn.Linear(24, 48), nn.GELU(), nn.Linear(48, splits * 12))
        self.split_out = nn.Parameter(torch.randn(splits, 12) * .02)

    def decode_attributes(self, centers, fused, voxel_size):
        hidden = self.attributes(fused).view(len(centers), self.splits, 12)
        attributes = (hidden + self.split_out[None]).float()
        width = voxel_size.reshape(-1)[:, None, None]
        centers = centers[:, None] + torch.tanh(attributes[..., :3]) * width
        scales = torch.sigmoid(attributes[..., 3:6]) * (self.max_scale_leaf * width)
        quats = F.normalize(attributes[..., 6:10] +
                            attributes.new_tensor([1., 0., 0., 0.]), dim=-1)
        gray = torch.sigmoid(attributes[..., 10:11])
        opacity = torch.sigmoid(attributes[..., 11:12] - 1)
        centers = centers.reshape(-1, 3)
        scales = scales.reshape(-1, 3)
        quats = quats.reshape(-1, 4)
        gray = gray.reshape(-1, 1)
        opacity = opacity.reshape(-1)
        if self.prune > 0:
            keep = opacity >= self.prune
            centers, scales, quats, gray, opacity = (
                x[keep] for x in (centers, scales, quats, gray, opacity))
        return dict(centers=centers, scales=scales, quats=quats, gray=gray, opacity=opacity)
