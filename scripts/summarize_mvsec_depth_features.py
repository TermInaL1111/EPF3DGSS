"""Summarize a same-camera intervention on DVLT geometry and recurrent features."""
import argparse
import json
from pathlib import Path
from statistics import fmean, median


ARMS = {
    'real_depth_real_features': 'zero_appearance_calibrated',
    'real_depth_zero_features': 'real_depth_zero_features_calibrated',
    'zero_depth_real_features': 'zero_depth_real_features_calibrated',
    'zero_depth_zero_features': 'zero_calibrated',
}
METRICS = {'psnr': 1, 'ssim': 1, 'lpips_vgg': -1, 'lpips_alex': -1}


def compare(rows, baseline):
    keys = [(row['flight'], row['start']) for row in rows]
    if len(set(keys)) != 72 or keys != [(row['flight'], row['start']) for row in baseline]:
        raise ValueError('Expected the same 72 MVSEC windows')
    scores = {}
    for name, key in ARMS.items():
        scores[name] = {metric: fmean(row[key][metric] for row in rows) for metric in METRICS}
        scores[name]['gaussian_count_median'] = median(row[key]['gaussian_count'] for row in rows)
    for depth in ('real', 'zero'):
        keys = [ARMS[f'{depth}_depth_{features}_features'] for features in ('real', 'zero')]
        if any(row[keys[0]]['gaussian_count'] != row[keys[1]]['gaussian_count'] for row in rows):
            raise ValueError(f'Feature intervention changed Gaussian count at {depth} depth')
    reference = scores['real_depth_real_features']
    changes = {arm: {metric: dict(delta=values[metric]-reference[metric],
                                  improvement_pct=100*sign*(values[metric]-reference[metric])/abs(reference[metric])
                                  if reference[metric] else None)
                     for metric, sign in METRICS.items()}
               for arm, values in scores.items() if arm != 'real_depth_real_features'}
    return dict(scores=scores, relative_to_real=changes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    heads = {}
    for head in ('fixed', 'octree'):
        rows = json.loads((args.folder/f'{head}_depth_features.json').read_text())
        baseline = json.loads((args.folder/f'mixed_{head}_mvsec.json').read_text())
        heads[head] = compare(rows, baseline)
    args.out.write_text(json.dumps(dict(windows=72, heads=heads), indent=2, allow_nan=False)+'\n')
