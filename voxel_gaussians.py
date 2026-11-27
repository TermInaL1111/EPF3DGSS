"""A Gaussian readout that decodes occupied 3D voxels."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class VoxelGaussianHead(nn.Module):
    """Reassemble recurrent features, fuse them spatially, then fuse in 3D."""

    voxel_size_m = .05

    def __init__(self, octree=False):
        super().__init__()
        self.octree = octree
        width = 24
        self.projections = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(4))
        self.refine = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                          nn.Conv2d(width, width, 3, padding=1))
            for _ in range(4))
        self.appearance = nn.Conv2d(17, width, 3, padding=1)
        self.attributes = nn.Sequential(nn.Linear(width, 48), nn.GELU(), nn.Linear(48, 9))
        nn.init.zeros_(self.attributes[-1].weight)
        nn.init.zeros_(self.attributes[-1].bias)

    def forward(self, pyramid, appearance, points, valid, confidence=None):
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
        dense = dense.permute(0, 2, 3, 1).reshape(points.shape[0], -1, dense.shape[1])
        if self.octree:
            views = torch.arange(appearance.shape[0] // points.shape[0], device=points.device)
            views = views.repeat_interleave(appearance.shape[-2] * appearance.shape[-1])
            scenes = [self.decode_octree(xyz[mask], features[mask], views[mask], conf[mask])
                      for xyz, features, mask, conf in zip(points, dense, valid, confidence, strict=True)]
        else:
            scenes = [self.decode_scene(xyz[mask], features[mask])
                      for xyz, features, mask in zip(points, dense, valid, strict=True)]
        cloud = {key: pad_sequence([scene[key] for scene in scenes], batch_first=True)
                 for key in scenes[0]}
        counts = torch.tensor([len(scene['opacity']) for scene in scenes], device=points.device)
        padding = torch.arange(cloud['opacity'].shape[1], device=points.device)[None] >= counts[:, None]
        identity = cloud['quats'].new_tensor([1., 0., 0., 0.])
        cloud['quats'] = torch.where(padding[..., None], identity, cloud['quats'])
        cloud['scales'] = cloud['scales'].masked_fill(padding[..., None], self.voxel_size_m / 2)
        cloud['gaussian_count'] = counts
        return cloud

    def decode_scene(self, points, features):
        if not len(points):
            raise ValueError('No valid candidates in the DVLT pointmap')
        points, features = points.float(), features.float()
        cells = (points / self.voxel_size_m).floor().long()
        _, membership, counts = torch.unique(cells, dim=0, return_inverse=True, return_counts=True)
        centers = points.new_zeros(len(counts), 3).index_add_(0, membership, points) / counts[:, None]
        fused = features.new_zeros(len(counts), features.shape[-1]).index_add_(
            0, membership, features) / counts[:, None]
        return self.decode_attributes(centers, fused, self.voxel_size_m)

    def decode_octree(self, points, features, views, confidence):
        from vggt.utils.octdensify import max_view_counts, select_leaves

        if not len(points):
            raise ValueError('No valid candidates in the DVLT pointmap')
        points, features = points.float(), features.float()
        confidence = confidence.float().nan_to_num(posinf=0, neginf=0).clamp_min(0)
        fine = ((points - points.amin(0)) / self.voxel_size_m).floor().long()
        coords, memberships, counts, view_counts = [], [], [], []
        for factor in (4, 2, 1):
            grid, ids, support = torch.unique(fine // factor, dim=0, return_inverse=True, return_counts=True)
            coords.append(grid)
            memberships.append(ids)
            counts.append(support)
            view_counts.append(max_view_counts(ids, views, len(grid)))
        leaves = select_leaves(coords, view_counts, ('point_count',) * 2, (.3, .3), None)
        centers, fused, widths = [], [], []
        for factor, ids, support, selected in zip((4, 2, 1), memberships, counts, leaves, strict=True):
            count = len(support)
            sums = points.new_zeros(count).index_add_(0, ids, confidence)
            weighted = points.new_zeros(count, 3).index_add_(0, ids, points * confidence[:, None])
            mean = points.new_zeros(count, 3).index_add_(0, ids, points) / support[:, None]
            center = torch.where(sums[:, None] > 0, weighted / sums[:, None].clamp_min(1e-12), mean)
            features_by_cell = features.new_zeros(count, features.shape[-1]).index_add_(
                0, ids, features) / support[:, None]
            centers.append(center[selected])
            fused.append(features_by_cell[selected])
            widths.append(points.new_full((len(selected), 1), self.voxel_size_m * factor))
        return self.decode_attributes(torch.cat(centers), torch.cat(fused), torch.cat(widths))

    def decode_attributes(self, centers, fused, voxel_size):
        sizes, rotation, gray, opacity = self.attributes(fused).float().split((3, 4, 1, 1), dim=-1)
        quaternion = F.normalize(rotation + rotation.new_tensor([1., 0., 0., 0.]), dim=-1)
        return dict(centers=centers, scales=sizes.clamp(-3, 3).exp() * (voxel_size / 2),
                    quats=quaternion, gray=gray.sigmoid(), opacity=(opacity[:, 0] - 1).sigmoid())
