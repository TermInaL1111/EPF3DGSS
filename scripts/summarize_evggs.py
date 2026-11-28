"""Summarize EvGGS metrics and verify the EPF comparison uses the same windows."""
import argparse
import json
from pathlib import Path
from statistics import mean


def aggregate(rows):
    fields = ('psnr', 'ssim', 'ssim_evggs', 'lpips_vgg', 'lpips_alex', 'foreground')
    return {k: mean(row[k] for row in rows) for k in fields if all(k in row for row in rows)}


def pair_with_epf(rows, epf, key, baseline):
    windows = [(row[key], row['start']) for row in rows]
    if not windows or len(set(windows)) != len(windows):
        raise ValueError('Expected nonempty unique EvGGS evaluation windows')
    if windows != [(row[key], row['start']) for row in epf]:
        raise ValueError('EPF and EvGGS windows differ')
    scores = aggregate([row['matched'] for row in epf])
    if scores.keys() != baseline.keys():
        raise ValueError('EPF and EvGGS metrics differ')
    deltas = {k:value-baseline[k] for k,value in scores.items()}
    return {'epf': scores, 'epf_minus_evggs': deltas,
            'relative_improvement_pct': {
                k:100*(-1 if k.startswith('lpips_') else 1)*delta/abs(baseline[k])
                if baseline[k] else None for k,delta in deltas.items() if k != 'foreground'}}


def main(args):
    baseline = json.loads(args.baseline.read_text())
    rows = baseline.pop('results')
    key = 'scene' if baseline['dataset'] == 'ev3ds' else 'flight'
    windows = [(row[key], row['start']) for row in rows]
    if len(set(windows)) != len(windows):
        raise ValueError('Duplicate evaluation windows')
    report = dict(**baseline, windows=len(rows), matched=aggregate([row['matched'] for row in rows]))
    if baseline['native']:
        report['raw'] = aggregate([row['raw'] for row in rows])
    if key == 'flight':
        report['per_flight'] = {
            name: aggregate([row['matched'] for row in rows if row[key] == name])
            for name in sorted({row[key] for row in rows})
        }
    depths = [view for row in rows for view in row.get('depth', []) if 'absrel' in view]
    if depths:
        report['depth'] = dict(views=len(depths), **{
            k: mean(view[k] for view in depths) for k in ('absrel', 'delta1', 'coverage')
        })
    if args.epf:
        if baseline['native']:
            raise ValueError('A paired comparison requires the shared protocol')
        report.update(pair_with_epf(rows, json.loads(args.epf.read_text()), key, report['matched']))
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--epf', type=Path)
    main(parser.parse_args())
