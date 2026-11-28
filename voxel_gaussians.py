"""A Gaussian readout that decodes occupied 3D voxels."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class VoxelGaussianHead(nn.Module):
    """Reassemble recurrent features, fuse them spatially, then fuse in 3D."""

    voxel_size_m = .05

    def __init__(self, octree=False, per_pixel_color=False, center_color=False, voxel_size=None,
                 color_width=None, color_layers=1):
        super().__init__()
        self.octree = octree
        self.per_pixel_color = per_pixel_color
        self.center_color = center_color
        if voxel_size is not None:
            self.voxel_size_m = float(voxel_size)
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
        if per_pixel_color:
            hidden = color_width or width
            layers = [nn.Conv2d(width, hidden, 3, padding=1), nn.GELU()]
            for _ in range(color_layers - 1):
                layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU()]
            layers.append(nn.Conv2d(hidden, 1, 1))
            self.color_head = nn.Sequential(*layers)
            nn.init.zeros_(self.color_head[-1].weight)
            nn.init.zeros_(self.color_head[-1].bias)

    def forward(self, pyramid, appearance, points, valid, confidence=None, boundary=None,
                event_leaf_ranking=False, matched_leaf_budget=False, source_intrinsics=None):
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
        color_map, color0, source_k = None, None, None
        if self.per_pixel_color:
            color_map = torch.sigmoid(self.color_head(dense)).reshape(points.shape[0], -1).float()
            if self.center_color and source_intrinsics is not None:
                views = appearance.shape[0] // points.shape[0]
                color0 = color_map.view(points.shape[0], views, appearance.shape[-2],
                                        appearance.shape[-1])[:, 0]
                base = source_intrinsics.view(points.shape[0], views, 3, 3)[:, 0].float()
                sx, sy = 640 / 224, 480 / 168
                source_k = base.clone()
                source_k[:, 0, 0] *= sx
                source_k[:, 0, 2] = (base[:, 0, 2] + .5) * sx - .5
                source_k[:, 1, 1] *= sy
                source_k[:, 1, 2] = (base[:, 1, 2] + .5) * sy - .5
        dense = dense.permute(0, 2, 3, 1).reshape(points.shape[0], -1, dense.shape[1])
        if self.octree:
            views = torch.arange(appearance.shape[0] // points.shape[0], device=points.device)
            views = views.repeat_interleave(appearance.shape[-2] * appearance.shape[-1])
            boundaries = [None] * len(points) if boundary is None else boundary
            scenes = [
                self.decode_octree(xyz[mask], features[mask], views[mask], conf[mask],
                                   None if score is None else score[mask],
                                   event_leaf_ranking, matched_leaf_budget,
                                   None if color_map is None else color_map[index][mask],
                                   None if color0 is None else color0[index],
                                   None if source_k is None else source_k[index])
                for index, (xyz, features, mask, conf, score) in enumerate(zip(
                    points, dense, valid, confidence, boundaries, strict=True))
            ]
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

    def decode_octree(self, points, features, views, confidence, boundary=None,
                      event_leaf_ranking=True, matched_leaf_budget=False, colors=None,
                      color0=None, source_k=None):
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
        geometric_leaves = select_leaves(coords, view_counts, ('point_count',) * 2, (.3, .3), None)
        leaves, budget = geometric_leaves, None
        if boundary is not None:
            priorities = []
            for ids, support, views_by_cell in zip(memberships, counts, view_counts, strict=True):
                mass = points.new_zeros(len(support)).index_add_(0, ids, confidence)
                edge_mass = points.new_zeros(len(support)).index_add_(0, ids, boundary.float() * confidence)
                geometric = views_by_cell.float().log1p()
                priorities.append(geometric / geometric.amax().clamp_min(1e-8)
                                  + edge_mass / mass.clamp_min(1e-8))

            def score(selected, level):
                return priorities[level][selected[level]]

            event_leaves = select_leaves(coords, view_counts, ('learned_score',) * 2, (.3, .3), score)
            if event_leaf_ranking:
                leaves = event_leaves
            if matched_leaf_budget:
                budget = min(sum(len(ids) for ids in selection) for selection in (geometric_leaves, event_leaves))
        centers, fused, widths, base_colors = [], [], [], []
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
            if colors is not None:
                color_by_cell = colors.new_zeros(count).index_add_(0, ids, colors) / support
                base = color_by_cell[selected]
                if color0 is not None and source_k is not None:
                    uvz = center[selected] @ source_k.transpose(0, 1)
                    z = uvz[:, 2]
                    u = uvz[:, 0] / z.clamp_min(1e-6)
                    v = uvz[:, 1] / z.clamp_min(1e-6)
                    valid = (z > 0) & (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
                    grid = torch.stack(((u + .5) / 640 * 2 - 1, (v + .5) / 480 * 2 - 1), -1)
                    sampled = F.grid_sample(color0[None, None], grid[None, None],
                                            align_corners=False)[0, 0, 0]
                    base = torch.where(valid, sampled, base)
                base_colors.append(base)
        cloud = self.decode_attributes(torch.cat(centers), torch.cat(fused), torch.cat(widths),
                                       None if colors is None else torch.cat(base_colors))
        if budget is not None and budget < len(cloud['opacity']):
            keep = torch.argsort(cloud['opacity'].detach(), descending=True, stable=True)[:budget].sort().values
            cloud = {key: value[keep] for key, value in cloud.items()}
        return cloud

    def decode_attributes(self, centers, fused, voxel_size, base_color=None):
        sizes, rotation, gray, opacity = self.attributes(fused).float().split((3, 4, 1, 1), dim=-1)
        quaternion = F.normalize(rotation + rotation.new_tensor([1., 0., 0., 0.]), dim=-1)
        if base_color is not None:
            gray = (base_color[:, None] + .25 * torch.tanh(gray)).clamp(0, 1)
        else:
            gray = gray.sigmoid()
        return dict(centers=centers, scales=sizes.clamp(-3, 3).exp() * (voxel_size / 2),
                    quats=quaternion, gray=gray, opacity=(opacity[:, 0] - 1).sigmoid())
