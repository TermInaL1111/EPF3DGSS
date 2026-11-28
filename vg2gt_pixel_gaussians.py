"""Self-splitting voxel Gaussian head with a per-pixel colour branch.

The geometry and shape attributes follow the VG^2GT-style voxel head.  Colour
is predicted at full image resolution and averaged over the points that form a
voxel, so sub-voxel intensity detail is not lost before Gaussian decoding.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class SelfSplittingPixelColourHead(nn.Module):
    voxel_size_m = .05

    def __init__(self, splits=2, width=24, hidden=64, max_scale_voxels=2., min_points=1, min_views=1, prune=0.):
        super().__init__()
        self.splits = splits
        self.max_scale_voxels = max_scale_voxels
        self.min_points = min_points
        self.min_views = min_views
        self.prune = prune
        self.projections = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(4))
        self.refine = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                          nn.Conv2d(width, width, 3, padding=1))
            for _ in range(4))
        self.appearance = nn.Conv2d(17, width, 3, padding=1)
        self.color = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.color.weight)
        nn.init.zeros_(self.color.bias)
        self.voxel_mlp = nn.Sequential(
            nn.Linear(width + 3, width), nn.GELU(), nn.Linear(width, width))
        self.attributes = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        self.split_head = nn.Linear(hidden, 12)
        self.split_bias = nn.Parameter(torch.randn(splits, hidden) * .02)
        self.split_out = nn.Parameter(torch.randn(splits, 12) * .02)
        nn.init.zeros_(self.split_head.weight)
        nn.init.zeros_(self.split_head.bias)

    def forward(self, pyramid, appearance, points, valid, confidence=None, boundary=None,
                event_leaf_ranking=False, matched_leaf_budget=False):
        fused = None
        for patch, project, refine, size in zip(
                reversed(pyramid), self.projections, self.refine,
                ((6, 8), (12, 16), (24, 32), (48, 64)), strict=True):
            level = F.interpolate(project(patch), size, mode='bilinear', align_corners=False)
            if fused is not None:
                level = level + F.interpolate(fused, size, mode='bilinear', align_corners=False)
            fused = level + refine(level)
        dense = F.gelu(self.appearance(appearance) + F.interpolate(
            fused, appearance.shape[-2:], mode='bilinear', align_corners=False))
        colour = torch.sigmoid(self.color(dense)).reshape(points.shape[0], -1).float()
        dense = dense.permute(0, 2, 3, 1).reshape(points.shape[0], -1, dense.shape[1])
        views = torch.arange(appearance.shape[0] // points.shape[0], device=points.device)
        views = views.repeat_interleave(appearance.shape[-2] * appearance.shape[-1])
        scenes = [self.decode(xyz[mask], features[mask], colour[index, mask], views[mask])
                  for index, (xyz, features, mask) in enumerate(zip(points, dense, valid, strict=True))]
        cloud = {key: pad_sequence([scene[key] for scene in scenes], batch_first=True)
                 for key in scenes[0]}
        counts = torch.tensor([len(scene['opacity']) for scene in scenes], device=points.device)
        padding = torch.arange(cloud['opacity'].shape[1], device=points.device)[None] >= counts[:, None]
        identity = cloud['quats'].new_tensor([1., 0., 0., 0.])
        cloud['quats'] = torch.where(padding[..., None], identity, cloud['quats'])
        cloud['scales'] = cloud['scales'].masked_fill(padding[..., None], self.voxel_size_m / 2)
        cloud['gray'] = cloud['gray'].masked_fill(padding[..., None], .5)
        cloud['opacity'] = cloud['opacity'].masked_fill(padding, 0.)
        cloud['gaussian_count'] = counts
        return cloud

    def decode(self, points, features, colours, views):
        if not len(points):
            raise ValueError('No valid candidates in the DVLT pointmap')
        points, features = points.float(), features.float()
        cells = (points / self.voxel_size_m).floor().long()
        grid, membership, counts = torch.unique(
            cells, dim=0, return_inverse=True, return_counts=True)
        from vggt.utils.octdensify import max_view_counts
        view_counts = max_view_counts(membership, views, len(counts))
        keep = (counts >= self.min_points) & (view_counts >= self.min_views)
        if not keep.all():
            grid, counts = grid[keep], counts[keep]
        base = (grid.float() + .5) * self.voxel_size_m
        fused = features.new_zeros(len(keep), features.shape[-1]).index_add_(
            0, membership, features)
        fused = fused[keep] if not keep.all() else fused
        fused = fused / counts[:, None]
        base_colour = colours.new_zeros(len(keep)).index_add_(0, membership, colours)
        base_colour = base_colour[keep] if not keep.all() else base_colour
        base_colour = base_colour / counts
        fused = fused.float()
        fused = fused + self.voxel_mlp(torch.cat((fused, base.float()), -1))
        hidden = self.attributes(fused)
        attributes = (self.split_head(hidden[:, None] + self.split_bias[None]) +
                      self.split_out[None]).float()
        centers = base[:, None] + torch.tanh(attributes[..., :3]) * self.voxel_size_m
        scales = torch.sigmoid(attributes[..., 3:6]) * (self.max_scale_voxels * self.voxel_size_m)
        quats = F.normalize(attributes[..., 6:10] + attributes.new_tensor([1., 0., 0., 0.]), dim=-1)
        gray = (base_colour[:, None, None] + .25 * torch.tanh(attributes[..., 10:11])).clamp(0, 1)
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
