"""Compare fixed-head source-camera interventions on the identical rendering windows."""

import argparse
import json
from pathlib import Path
from statistics import fmean

from summarize_attribute_supervision import manifest_windows, metric_change, read_json
from summarize_multiscene_gaussians import METRICS


GEOMETRY_DIAGNOSTICS = (
    'source_depth_fraction','metric_depth_absrel','aligned_depth_absrel','reference_depth_scale',
    'focal_x_ratio','focal_y_ratio','principal_point_error_px','local_ray_angle_deg',
    'rotation_error_deg','translation_error_m','depth_reprojection_px',
)


def summarize(root, reference):
    config = read_json(root/'config.json')
    if not config['source_camera_control'] or config['steps']:
        raise ValueError('Expected a frozen two-camera evaluation')
    metrics = METRICS | {'alpha_mean':1,'alpha_covered_fraction':1}
    report = {}
    for dataset in ('dsec','eventscape'):
        expected = manifest_windows(config,dataset,'test')
        filename = f'val_test_{dataset}.json'
        rows = read_json(root/filename)
        original = {(row['sequence'],row['frame']):row for row in read_json(reference/filename)}
        keys = {(row['sequence'],row['frame']) for row in rows}
        if len(rows) != len(keys) or keys != expected or keys != original.keys():
            raise ValueError(f'Unexpected {dataset} test windows')
        max_drift = max(abs(row[f'{arm}_psnr']-original[row['sequence'],row['frame']][f'{arm}_psnr'])
                        for row in rows for arm in ('matched','zero'))
        if max_drift > .05:
            raise ValueError(f'Predicted-camera results changed by {max_drift:.3f} dB on {dataset}')
        scores = {}
        changes = {}
        for arm in ('matched','zero'):
            predicted = {m:fmean(row[f'{arm}_{m}'] for row in rows) for m in metrics}
            calibrated = {m:fmean(row[f'{arm}_calibrated_{m}'] for row in rows) for m in metrics}
            scores[arm] = dict(predicted=predicted,calibrated=calibrated)
            changes[arm] = {m:metric_change(predicted[m],calibrated[m],sign) for m,sign in metrics.items()}
        report[dataset] = dict(windows=len(rows),scores=scores,calibrated_vs_predicted=changes,
                               maximum_predicted_psnr_drift_db=max_drift)
        if config.get('source_camera_components'):
            predicted = scores['matched']['predicted']
            components = {}
            for component in ('intrinsics','poses'):
                estimate = {m:fmean(row[f'matched_{component}_{m}'] for row in rows) for m in metrics}
                components[component] = dict(scores=estimate,
                    vs_predicted={m:metric_change(predicted[m],estimate[m],sign) for m,sign in metrics.items()})
            report[dataset]['components'] = components
        if config.get('source_depth_control'):
            predicted_depth = scores['matched']['calibrated']
            measured_depth = {m:fmean(row[f'matched_reference_depth_{m}'] for row in rows) for m in metrics}
            diagnostics = {}
            for key in GEOMETRY_DIAGNOSTICS:
                values = [row[f'matched_calibrated_{key}'] for row in rows
                          if row[f'matched_calibrated_{key}'] is not None]
                diagnostics[key] = dict(mean=fmean(values) if values else None,windows=len(values))
            report[dataset]['depth_control'] = dict(
                measured_depth_scores=measured_depth,
                vs_predicted_depth={m:metric_change(predicted_depth[m],measured_depth[m],sign)
                                    for m,sign in metrics.items()},
                geometry=diagnostics,
            )
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','reference','out'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    report = summarize(args.root,args.reference)
    temporary = args.out.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.out)
