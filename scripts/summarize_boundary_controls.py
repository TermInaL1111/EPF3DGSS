"""Summarize paired event controls using sequence-level resampling."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


CONTROLS = ('time_reversed', 'polarity_swapped', 'time_uniform')
METRICS = ('boundary_f1_1px', 'boundary_absrel')
DIRECTIONS = (1, -1)


def group_stats(rows, arm):
    values = [row[arm] for row in rows]
    errors = [row['boundary_absrel'] for row in values if row['boundary_absrel'] is not None]
    counts = ('edge_predicted_near_reference', 'edge_reference_near_predicted',
              'edge_predicted', 'edge_reference')
    return [*(sum(row[k] for row in values) for k in counts), sum(errors), len(errors)]


def scores(totals):
    near_pred, near_ref, pred, ref, error, count = totals.T
    precision = near_pred / np.maximum(pred, 1)
    recall = near_ref / np.maximum(ref, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return np.stack([f1, error / np.maximum(count, 1)], axis=-1)


def compare(report, baseline):
    rows = report['rows']
    keys = [(r['sequence'], r['index']) for r in rows]
    if (not rows or len(set(keys)) != len(keys)
            or keys != [(r['sequence'], r['index']) for r in baseline['rows']]):
        raise ValueError('Controls and boundary baseline must use the same windows')
    groups = defaultdict(list)
    for row in rows:
        groups[row['sequence']].append(row)
    rng = np.random.default_rng(20261127)
    selections = rng.integers(len(groups),size=(20000,len(groups)))
    reference = np.asarray([group_stats(group,'temporal') for group in groups.values()])
    original_score = scores(reference.sum(axis=0)[None])[0]
    original_samples = scores(reference[selections].sum(axis=1))
    result = {}
    for arm in CONTROLS:
        control = np.asarray([group_stats(group,arm) for group in groups.values()])
        control_score = scores(control.sum(axis=0)[None])[0]
        control_samples = scores(control[selections].sum(axis=1))
        changes = (control_samples-original_samples)*DIRECTIONS
        bounds = np.quantile(changes,[.025,.975],axis=0).T
        result[arm] = {}
        for i,metric in enumerate(METRICS):
            change = DIRECTIONS[i]*(control_score[i]-original_score[i])
            result[arm][metric] = dict(original=float(original_score[i]),control=float(control_score[i]),
                improvement=float(change),improvement_pct=float(100*change/abs(original_score[i]))
                if original_score[i] else None,sequence_bootstrap_95_interval=bounds[i].tolist())
    return dict(dataset=report['dataset'],windows=len(rows),sequences=len(groups),controls=result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('report','baseline','out'):
        parser.add_argument(argument,type=Path)
    args = parser.parse_args()
    report, baseline = (json.loads(path.read_text()) for path in (args.report,args.baseline))
    result = compare(report,baseline)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
