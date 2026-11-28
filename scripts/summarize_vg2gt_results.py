"""Build the VG^2GT-style head comparison from frozen-head evaluation runs."""

import json
from pathlib import Path
from statistics import fmean

ROOT = Path('/data/userwj/epf-runs')
OUT = Path('results')


def means(path):
    rows = json.loads(path.read_text())
    return dict(
        windows=len(rows),
        psnr=fmean(row['matched_psnr'] for row in rows),
        ssim=fmean(row['matched_ssim'] for row in rows),
        lpips_vgg=fmean(row['matched_lpips_vgg'] for row in rows),
        lpips_alex=fmean(row['matched_lpips_alex'] for row in rows),
        zero_psnr=fmean(row['zero_psnr'] for row in rows),
        gaussians=fmean(row.get('matched_gaussian_count', 0) for row in rows),
    )


def eval_run(name, path):
    return {
        'name': name,
        'dsec': means(ROOT / path / 'val_test_dsec.json'),
        'eventscape': means(ROOT / path / 'val_test_eventscape.json'),
    }


references = json.loads((OUT / 'gaussian_camera_components.json').read_text())
target8 = json.loads((OUT / 'octree_target_baseline.json').read_text())

runs = [
    eval_run('octree_target4_12k', 'octree_target4_eval_12k'),
    eval_run('vg2gt_fine_target4_4k6', 'vg2gt_target4_eval_46k'),
    eval_run('vg2gt_octree_target4_12k', 'vg2gt_octree_target4_12k'),
]

report = {
    'protocol': {
        'camera': 'known sensor intrinsics + event-predicted relative source poses',
        'target_offset': 4,
        'train_windows': 4034,
        'train_steps': {'octree': 12000, 'vg2gt_fine': 4600, 'vg2gt_octree': 12000},
        'note': 'Fine-voxel VG2GT run was stopped at 4.6k after DSEC geometry collapsed.',
    },
    'target2_known_intrinsics': {
        dataset: references[dataset]['components']['intrinsics']['scores']
        for dataset in ('dsec', 'eventscape')
    },
    'target8_octree_reference_head': {
        dataset: {
            'windows': target8[dataset]['windows'],
            'target2': target8[dataset]['runs']['t2']['matched'],
            'target8': target8[dataset]['runs']['t8']['matched'],
        }
        for dataset in ('dsec', 'eventscape')
    },
    'runs': runs,
}
(OUT / 'vg2gt_head_comparison.json').write_text(json.dumps(report, indent=2) + '\n')

lines = ['dataset,method,target_offset,windows,psnr,ssim,lpips_vgg,lpips_alex,zero_psnr,gaussians']
for dataset in ('dsec', 'eventscape'):
    for label, key in (('octree', 'target2_known_intrinsics'),
                       ('octree', 'octree_target4_12k'),
                       ('vg2gt_fine', 'vg2gt_fine_target4_4k6'),
                       ('vg2gt_octree', 'vg2gt_octree_target4_12k')):
        if key == 'target2_known_intrinsics':
            scores = report[key][dataset]
            lines.append(f'{dataset},{label},2,,{scores["psnr"]:.4f},{scores["ssim"]:.4f},'
                         f'{scores["lpips_vgg"]:.4f},{scores["lpips_alex"]:.4f},,')
        else:
            run = next(item for item in runs if item['name'] == key)
            scores = run[dataset]
            offset = 4
            lines.append(f'{dataset},{label},{offset},{scores["windows"]},{scores["psnr"]:.4f},'
                         f'{scores["ssim"]:.4f},{scores["lpips_vgg"]:.4f},{scores["lpips_alex"]:.4f},'
                         f'{scores["zero_psnr"]:.4f},{scores["gaussians"]:.0f}')
(OUT / 'vg2gt_head_comparison.csv').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
