"""Compare zero-shot and adapted DVLT geometry on identical MVSEC windows."""

import argparse
import json
from pathlib import Path
from statistics import fmean


def compare(reports):
    metrics = dict(absrel=-1, delta1=1, ate_m=-1, pose_auc3=1,
                   pose_auc20=1, pose_auc30=1, point_absrel=-1,
                   point_inlier3=1, point_inlier25=1)
    scores = {}
    rows_by_arm = {}
    expected = None
    for name, report in reports.items():
        if (report['clip_views'], report['max_depth_m'], report['camera_grid']) != (
                18, 80, 'full rectified DAVIS'):
            raise ValueError('Expected the same MVSEC outdoor geometry protocol')
        rows = report['rows']
        windows = {(r['flight'], tuple(r['frames'])) for r in rows}
        if not windows or len(windows) != len(rows) or (expected is not None and windows != expected):
            raise ValueError('The MVSEC reports must use the same unique windows')
        expected = windows
        scores[name] = {metric: fmean(r[metric] for r in rows) for metric in metrics}
        rows_by_arm[name] = {(row['flight'],tuple(row['frames'])):row for row in rows}
    normal_windows = 0
    if all('cloud_accuracy_m' in row for rows in rows_by_arm.values() for row in rows.values()):
        nc = 'cloud_normal_consistency'
        normal_keys = {key for key in expected if all(rows[key][nc] is not None for rows in rows_by_arm.values())}
        normal_windows = len(normal_keys)
        cloud_metrics = dict(cloud_accuracy_m=-1,cloud_completeness_m=-1)
        if normal_keys:
            cloud_metrics[nc] = 1
        metrics.update(cloud_metrics)
        for arm, rows in rows_by_arm.items():
            scores[arm].update({metric:fmean(rows[key][metric] for key in
                                            (normal_keys if metric == nc else expected))
                                for metric in cloud_metrics})
    relative = {}
    for metric, sign in metrics.items():
        base, adapted = (scores[arm][metric] for arm in ('base','adapted'))
        relative[metric] = 100*sign*(adapted-base)/abs(base) if base else None
    return dict(windows=len(expected), scores=scores, adapted_vs_base_improvement_pct=relative,
                steps={name: report['step'] for name, report in reports.items()},
                cloud_normal_windows=normal_windows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    reports = {arm: json.loads((args.out/f'{arm}.json').read_text())
               for arm in ('teacher','base','adapted')}
    report = compare(reports)
    temporary = args.out/'comparison.json.tmp'
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.out/'comparison.json')
