"""Stratify single-camera rendering comparisons by reference query motion."""

import argparse
import json
from pathlib import Path
from statistics import fmean

import numpy as np


METRICS = dict(psnr=1, ssim=1, lpips_vgg=-1, lpips_alex=-1)


def stratify(arms, motion):
    by_arm = {}
    for arm, rows in arms.items():
        keys = [(r.get('sequence',r.get('flight')),r.get('frame',r.get('start'))) for r in rows]
        if len(set(keys)) != len(rows) or set(keys) != motion.keys():
            raise ValueError(f'Window manifest differs for {arm}')
        by_arm[arm] = dict(zip(keys,rows,strict=True))

    def scores(row, condition, metric):
        return row[condition][metric] if condition in row else row[f'{condition}_{metric}']

    groups = np.array_split(np.argsort([motion[key] for key in motion],kind='stable'),3)
    keys = list(motion)
    report = []
    for ids in groups:
        selected = [keys[i] for i in ids]
        segment = [motion[key] for key in selected]
        reference = next(iter(by_arm.values()))
        copy = fmean(reference[key]['source_copy_psnr'] for key in selected)
        result = dict(windows=len(selected),query_motion_m=[min(segment),max(segment)],source_copy_psnr=copy)
        result['heads'] = {}
        for arm, rows in by_arm.items():
            metrics = {}
            for metric, sign in METRICS.items():
                pairs = [(scores(rows[key],'matched',metric),scores(rows[key],'zero',metric)) for key in selected]
                matched, zero = (fmean(items) for items in zip(*pairs,strict=True))
                metrics[metric] = dict(matched=matched,zero=zero,
                    event_improvement=sign*(matched-zero),
                    event_improvement_pct=100*sign*(matched-zero)/abs(zero) if zero else None)
            result['heads'][arm] = metrics
        report.append(result)
    return report


def summary(root):
    def read(path):
        return json.loads((root/path).read_text())

    reports = {}
    for dataset,folder,cameras in (
        ('dsec','dsec_temporal_gaussians','dsec_temporal_cameras'),
        ('eventscape','eventscape_realm_gaussians','eventscape_rgbd_cameras'),
    ):
        sequences = read(f'{cameras}/test.json')['sequences']
        if dataset == 'dsec':
            sequences = {name:row['cameras'] for name,row in sequences.items()}
        # First source to query in the same physical camera. DSEC's offset
        # between the RGB and event sensors is excluded from temporal motion.
        motion = {(name,int(index)):float(np.linalg.norm(np.asarray(poses[0]['c2w'])[:3,3]))
                  for name,rows in sequences.items() for index,poses in rows.items()}
        arms = {head:read(f'{folder}/{head}/val_test.json') for head in ('fixed','octree')}
        reports[dataset] = stratify(arms,motion)

    octree = read('mvsec_octree_no_f3/shared_128x96.json')
    motion = {(r['flight'],r['start']):r['query_baseline_m'] for r in octree}
    reports['mvsec'] = stratify({'octree':octree},motion)
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summary(args.runs),indent=2,allow_nan=False)+'\n')
