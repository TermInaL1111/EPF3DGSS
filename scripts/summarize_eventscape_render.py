"""Compare Gaussian heads on all accepted temporal test cameras."""
import argparse
import json
from pathlib import Path
from statistics import fmean, median

from summarize_render import summarize


def compare(manifest, arms, image_size=(512,256)):
    expected = {(name, int(index)) for name, rows in manifest['sequences'].items() for index in rows}
    for rows in arms.values():
        if len(rows) != len(expected) or {(r['sequence'], r['frame']) for r in rows} != expected:
            raise ValueError('Every arm must render exactly the accepted test windows')
    heads = {arm: summarize(rows)['all'] for arm, rows in arms.items()}
    for arm, rows in arms.items():
        heads[arm]['gaussian_count_median'] = {
            condition: median(r[f'{condition}_gaussian_count'] for r in rows)
            for condition in ('matched', 'zero')
        }
    metrics = dict(psnr=1, ssim=1, lpips_vgg=-1, lpips_alex=-1)
    relative = {}
    for condition in ('matched', 'zero'):
        relative[condition] = {}
        for metric, sign in metrics.items():
            baseline, estimate = (heads[arm][metric][condition] for arm in ('fixed','octree'))
            relative[condition][metric] = dict(
                delta=estimate-baseline,
                improvement_pct=100*sign*(estimate-baseline)/abs(baseline) if baseline else None)
    return dict(windows=len(expected), image_size=list(image_size), heads=heads,
                octree_vs_fixed=relative)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('--dsec', action='store_true', help='Use a DSEC camera manifest and 640x480 scores')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.dsec:
        manifest['sequences'] = {name:row['cameras'] for name,row in manifest['sequences'].items()}
    arms = {arm: json.loads((args.run/arm/'val_test.json').read_text()) for arm in ('fixed', 'octree')}
    report = compare(manifest, arms, (640,480) if args.dsec else (512,256))
    output = args.run/('dsec_comparison.json' if args.dsec else 'town05_comparison.json')
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(output)
