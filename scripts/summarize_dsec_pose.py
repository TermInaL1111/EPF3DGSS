"""Diagnose DSEC pose AUC and compare final F3 arms on matched windows."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean, median

import numpy as np


def pose_summary(rows):
    direction = [r['translation_deg'] for r in rows if r['translation_deg'] is not None]
    return dict(windows=len(rows), signed_direction_windows=len(direction),
                rotation_median_deg=median(r['rotation_deg'] for r in rows),
                rotation_below_3_deg=sum(r['rotation_deg'] < 3 for r in rows),
                translation_median_deg=median(direction) if direction else None,
                translation_at_least_20_deg=sum(t >= 20 for t in direction),
                translation_reversed=sum(t > 90 for t in direction),
                pose_auc20=fmean(r['pose_auc20'] for r in rows))


def compare(no_f3, f3):
    if (no_f3['step'],no_f3['depth_cap_m']) != (f3['step'],f3['depth_cap_m']):
        raise ValueError('Expected checkpoints with the same training length and depth cap')
    arms = {}
    for name,data in (('no_f3',no_f3),('f3',f3)):
        rows = data['rows']
        by_key = {(r['sequence'],r['frame']):r for r in rows}
        if not rows or len(rows) != len(by_key):
            raise ValueError('DSEC evaluation requires nonempty, unique windows')
        arms[name] = by_key
    keys = arms['no_f3'].keys()
    if keys != arms['f3'].keys():
        raise ValueError('F3 arms must use the same DSEC windows')
    per_sequence = defaultdict(list)
    for key in keys:
        per_sequence[key[0]].append(key)
    sequence_names = list(per_sequence)
    rng = np.random.default_rng(20261127)
    samples = rng.integers(len(sequence_names),size=(20000,len(sequence_names)))
    counts = np.asarray([len(per_sequence[name]) for name in sequence_names])
    denominator = counts[samples].sum(axis=1)
    changes = {}
    for metric in ('absrel','delta1','dvlt_point_rel_l2','sim3_point_inlier25','pose_auc20'):
        totals = np.asarray([sum(arms['f3'][key]['matched'][metric]-arms['no_f3'][key]['matched'][metric]
                                 for key in per_sequence[sequence]) for sequence in sequence_names])
        low,high = np.quantile(totals[samples].sum(axis=1)/denominator,[.025,.975])
        changes[metric] = dict(f3_minus_no_f3=float(totals.sum()/len(keys)),
                               sequence_bootstrap_95_interval=[float(low),float(high)])
    result = {}
    for name,rows in arms.items():
        result[name] = {condition:pose_summary([rows[key][condition] for key in keys])
                        for condition in ('matched','zero')}
        result[name]['per_sequence'] = {sequence:pose_summary([rows[key]['matched'] for key in sequence_keys])
                                        for sequence,sequence_keys in per_sequence.items()}
    return dict(steps=no_f3['step'],windows=len(keys),sequences=len(per_sequence),
                depth_cap_m=no_f3['depth_cap_m'],arms=result,f3_changes=changes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('no_f3','f3','out'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    report = compare(json.loads(args.no_f3.read_text()),json.loads(args.f3.read_text()))
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
