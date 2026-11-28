"""Measure how matched-budget MVSEC deltas vary across flights and windows."""

import argparse
import json
from pathlib import Path
from statistics import fmean, median


METRICS = dict(psnr=1, ssim=1, lpips_vgg=-1, lpips_alex=-1)
FLIGHTS = tuple(f'indoor_flying{i}' for i in (1, 2, 3))


def compare(geometry, events):
    keys = [(row['flight'], row['start']) for row in geometry]
    if (len(keys) != 72 or len(set(keys)) != 72
            or keys != [(row['flight'], row['start']) for row in events]
            or any(sum(flight == name for flight, _ in keys) != 24 for name in FLIGHTS)):
        raise ValueError('Expected the same 24 windows per flight, in both arms')
    output = {}
    for condition in ('matched', 'zero'):
        groups = {name: [] for name in FLIGHTS}
        for a, b in zip(geometry, events, strict=True):
            first, second = a[condition], b[condition]
            count = first['gaussian_count']
            if not count or count != second['gaussian_count']:
                raise ValueError(f"Unmatched Gaussian budget in {a['flight']}/{a['start']}")
            groups[a['flight']].append((count, {metric: sign*(second[metric]-first[metric])
                                               for metric, sign in METRICS.items()}))
        all_rows = [row for group in groups.values() for row in group]
        summary = {}
        for metric in METRICS:
            values = [scores[metric] for _, scores in all_rows]
            summary[metric] = dict(
                mean_improvement=fmean(values), median_improvement=median(values),
                improved_windows=sum(value > 0 for value in values),
                tied_windows=sum(value == 0 for value in values),
                per_flight={name: fmean(scores[metric] for _, scores in rows)
                            for name, rows in groups.items()},
                leave_one_flight_out={name: fmean(scores[metric] for other, rows in groups.items()
                                                 if other != name for _, scores in rows)
                                      for name in groups})
        counts = [count for count, _ in all_rows]
        output[condition] = dict(gaussians=dict(minimum=min(counts), median=median(counts),
                                               maximum=max(counts), mean=fmean(counts)),
                                 metrics=summary)
    return dict(windows=len(keys), diagnostics=output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('prefix')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    reports = [json.loads((args.run/f'{args.prefix}_{arm}/shared_128x96.json').read_text())
               for arm in ('geometry', 'event')]
    result = compare(*reports)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.output)
