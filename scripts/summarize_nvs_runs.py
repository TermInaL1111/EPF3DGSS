"""Summarize several frozen-head evaluations on their common rendering windows."""

import argparse
import json
from pathlib import Path
from statistics import fmean


METRICS = ('psnr', 'ssim', 'lpips_vgg', 'lpips_alex')


def read_run(path):
    runs = {}
    for dataset, name in (('dsec', 'val_test_dsec.json'), ('eventscape', 'val_test_eventscape.json')):
        file = path / name
        if file.is_file():
            rows = json.loads(file.read_text())
            runs[dataset] = {(row['sequence'], row['frame']): row for row in rows}
    return runs


def summarize(specs):
    report = {}
    for dataset in ('dsec', 'eventscape'):
        runs = {label: read_run(path).get(dataset, {}) for label, path in specs}
        runs = {label: rows for label, rows in runs.items() if rows}
        if not runs:
            continue
        common = set.intersection(*(set(rows) for rows in runs.values()))
        order = sorted(common)
        entry = dict(windows=len(order), runs={})
        for label, rows in runs.items():
            scores = {}
            for arm in ('matched', 'zero', 'matched_calibrated', 'matched_intrinsics', 'matched_poses',
                        'matched_reference_depth'):
                keys = [f'{arm}_{metric}' for metric in METRICS]
                if all(key in rows[order[0]] for key in keys):
                    scores[arm] = {metric: fmean(rows[key][f'{arm}_{metric}'] for key in order)
                                   for metric in METRICS}
                    for extra in ('gaussian_count', 'alpha_covered_fraction', 'alpha_mean'):
                        field = f'{arm}_{extra}'
                        if field in rows[order[0]]:
                            scores[arm][extra] = fmean(rows[key][field] for key in order)
            entry['runs'][label] = scores
        report[dataset] = entry
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', help='label=path entries')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    specs = []
    for item in args.runs:
        label, path = item.split('=', 1)
        specs.append((label, Path(path)))
    report = summarize(specs)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(text + '\n')
    print(text)
