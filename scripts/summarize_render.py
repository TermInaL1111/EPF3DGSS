"""Summarize matched/zero render results and their paired differences by scene."""
import argparse
import json
from pathlib import Path
from statistics import fmean


def summarize(rows):
    def read(row, condition, metric):
        return row.get(condition,{}).get(metric,row.get(f'{condition}_{metric}'))

    groups = {'all': rows}
    for row in rows:
        key = next(row[k] for k in ('sequence','scene','flight') if k in row)
        groups.setdefault(key,[]).append(row)
    summary = {}
    for name,group in groups.items():
        report = dict(windows=len(group))
        for metric in ('psnr','ssim','overlap_psnr','foreground','rotation_deg','lpips_vgg','lpips_alex','ssim_evggs'):
            pairs = [(read(r,'matched',metric),read(r,'zero',metric)) for r in group]
            pairs = [(m,z) for m,z in pairs if m is not None and z is not None]
            if pairs:
                matched,zero = zip(*pairs)
                report[metric] = dict(matched=fmean(matched),zero=fmean(zero),
                                      matched_minus_zero=fmean(m-z for m,z in pairs))
        for key in ('source_baseline_m','query_baseline_m','identity_rotation_deg',
                    'source_copy_psnr','source_copy_ssim'):
            values = [row[key] for row in group if key in row]
            if values:
                report[key] = fmean(values)
        summary[name] = report
    return summary


def compare_dsec(calibrated, predicted):
    def index(rows):
        lookup = {(row['sequence'],row['frame']):row for row in rows}
        if not lookup or len(lookup) != len(rows):
            raise ValueError('Each arm must contain unique sequence/frame rows')
        return lookup

    reference, estimate = index(calibrated), index(predicted)
    if reference.keys() != estimate.keys():
        raise ValueError('DSEC Gaussian arms use different validation windows')
    rows = []
    for key, baseline in reference.items():
        row = estimate[key]
        if abs(baseline['overlap_fraction']-row['overlap_fraction']) > 1e-6:
            raise ValueError(f'Camera overlap differs in {key}')
        rows.append(dict(sequence=key[0],frame=key[1],
                         **{metric:row[metric]-baseline[metric]
                            for metric in ('matched_psnr','matched_ssim','matched_overlap_psnr')}))
    return dict(windows=len(rows),calibrated=summarize(calibrated),predicted=summarize(predicted),
                matched_predicted_minus_calibrated={
                    metric:fmean(row[metric] for row in rows)
                    for metric in ('matched_psnr','matched_ssim','matched_overlap_psnr')},
                paired_rows=rows)


def compare_boundary(rows):
    keys = [(r['flight'], r['start']) for r in rows]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('Expected unique MVSEC flight windows')
    arms = ('matched', 'boundary_geometry', 'boundary_accumulated', 'boundary_fixed_edge', 'boundary_temporal')
    metrics = {'psnr': 1, 'ssim': 1, 'lpips_vgg': -1, 'lpips_alex': -1}
    summary = {}
    for arm in arms:
        values = [row[arm] for row in rows]
        summary[arm] = {metric: fmean(r[metric] for r in values) for metric in metrics}
        if all('gaussian_count' in r for r in values):
            summary[arm]['mean_gaussian_count'] = fmean(r['gaussian_count'] for r in values)
    relative = {}
    for reference in ('matched', 'boundary_geometry'):
        relative[reference] = {
            arm: {m: 100 * sign * (summary[arm][m] - summary[reference][m]) / abs(summary[reference][m])
                  if summary[reference][m] else None for m, sign in metrics.items()}
            for arm in arms if arm != reference
        }
    return dict(windows=len(rows), summary=summary, relative_improvement_pct=relative)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--predicted',type=Path,help='Compare a calibrated input against a DSEC predicted arm')
    args=p.parse_args()
    rows = json.loads(args.input.read_text())
    if args.boundary:
        report = compare_boundary(rows)
    elif args.predicted:
        report = compare_dsec(rows, json.loads(args.predicted.read_text()))
    else:
        report = summarize(rows)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
