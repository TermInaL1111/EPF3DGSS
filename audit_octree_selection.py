"""Run the official OctreeSplat point-count selector on frozen MVSEC DVLT point maps."""

import argparse
import json
from contextlib import closing
from pathlib import Path

import numpy as np
import torch

from event_dvlt import EventDVLT, load_scene_weights, voxelize
from mvsec_gaussian_data import MVSECTemporalGaussianSequence
from vggt.utils.octdensify import max_view_counts, select_leaves


@torch.inference_mode()
def inspect(args):
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = EventDVLT(args.assets, device='cpu').eval()
    steps, use_field = load_scene_weights(model, checkpoint)
    if use_field:
        raise ValueError('Select a no-F³ fixed-center checkpoint for this diagnostic')
    field = torch.zeros((1, 2, 32, 96, 128))
    fractions = (.3, .3)
    sizes = [args.voxel_side / 2**level for level in range(3)]
    rows = []
    calib = args.root / 'calibration/camchain-imucam-indoor_flying.yaml'
    with closing(MVSECTemporalGaussianSequence(args.root, 'indoor_flying4', calib)) as source:
        for row in np.linspace(0, len(source.indices)-1, args.windows, dtype=int):
            index = source.indices[row]
            voxel = torch.stack([voxelize(source.packet(v, index)) for v in range(2)])[None]
            points = model(voxel, field, steps=steps)['centers'][0].float()
            views = torch.arange(2).repeat_interleave(points.shape[0] // 2)
            valid = torch.isfinite(points).all(dim=-1)
            points, views = points[valid], views[valid]
            if not len(points):
                raise ValueError(f'No valid DVLT points at frame {index}')
            fine = ((points - points.amin(dim=0)) / sizes[-1]).floor().long()
            grids, memberships, counts = [], [], []
            for level in range(3):
                coords, ids = torch.unique(fine // 2**(2-level), dim=0, return_inverse=True)
                grids.append(coords)
                memberships.append(ids)
                counts.append(max_view_counts(ids, views, len(coords)))
            leaves = select_leaves(grids, counts, ['point_count']*2, fractions, None)
            coverage = torch.zeros_like(views)
            for grid, ids, selected in zip(grids, memberships, leaves, strict=True):
                mask = torch.zeros(len(grid), dtype=torch.bool)
                mask[selected] = True
                coverage += mask[ids]
            if not (coverage == 1).all():
                raise RuntimeError(f'Octree leaves do not partition the source points at frame {index}')
            rows.append(dict(frame=index, valid_points=len(points),
                             uniform_voxels=[len(g) for g in grids],
                             selected_leaves=[len(ids) for ids in leaves],
                             total_leaves=sum(len(ids) for ids in leaves)))
            print(rows[-1], flush=True)
    return dict(checkpoint=str(args.checkpoint), checkpoint_step=checkpoint['step'],
                sequence='indoor_flying4', voxel_sizes_m=sizes, split_fractions=fractions, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('root', 'assets', 'checkpoint', 'output'):
        parser.add_argument(field, type=Path)
    parser.add_argument('--windows', type=int, default=8)
    parser.add_argument('--voxel-side', type=float, default=.20)
    args = parser.parse_args()
    if args.windows < 1 or not np.isfinite(args.voxel_side) or args.voxel_side <= 0:
        parser.error('Provide a positive window count and finite voxel side')
    args.output.write_text(json.dumps(inspect(args), indent=2, allow_nan=False) + '\n')
