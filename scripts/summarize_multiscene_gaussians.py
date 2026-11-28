"""Validate and compare the mixed-training Gaussian heads across all test sets."""

import argparse
import json
from pathlib import Path
from statistics import fmean

from summarize_eventscape_render import compare
from summarize_evggs import aggregate, pair_with_epf
from summarize_render import summarize


METRICS = dict(psnr=1,ssim=1,lpips_vgg=-1,lpips_alex=-1)


def training_windows(folder, config):
    windows = set()
    for key in ('dsec_temporal_cameras','eventscape_cameras'):
        manifest = json.loads((Path(config[key])/'train.json').read_text())
        for name, rows in manifest['sequences'].items():
            indices = rows['cameras'] if key == 'dsec_temporal_cameras' else rows
            windows.update((name,int(index)) for index in indices)
    for arm in ('fixed','octree'):
        with (folder/arm/'train.jsonl').open() as stream:
            samples = [(r['step'],r['sequence'],r['frame']) for r in map(json.loads,stream)]
        if len(samples) != config['steps'] or any(row[0] != i for i,row in enumerate(samples,1)):
            raise ValueError(f'Incomplete training record for {arm}')
        width = len(windows)
        for start in range(0,len(samples),width):
            items = [row[1:] for row in samples[start:start+width]]
            if len(set(items)) != len(items) or set(items) - windows:
                raise ValueError(f'Unexpected windows or repeated sampling within a pass for {arm}')
        if arm == 'fixed':
            reference = samples
        elif samples != reference:
            raise ValueError('Fixed and octree used different training windows')
    return len(windows)


def camera_control(rows):
    report = {}
    for name,suffix in (('predicted',''),('calibrated','_calibrated')):
        report[name] = {}
        for metric,sign in METRICS.items():
            real,zero,swapped = (fmean(row[f'{condition}{suffix}'][metric] for row in rows)
                                 for condition in ('matched','zero','swapped'))
            report[name][metric] = dict(real=real,zero=zero,
                real_improvement=sign*(real-zero),
                improvement_pct=100*sign*(real-zero)/abs(zero) if zero else None,
                swapped=swapped,real_vs_swapped=sign*(real-swapped),
                real_vs_swapped_pct=100*sign*(real-swapped)/abs(swapped) if swapped else None)
    return report


def compare_training_domains(arms):
    pairs = (('dsec_temporal','mixed_octree'), ('eventscape_realm','mixed_octree'),
             ('mixed_fixed','mixed_octree'))
    result = {}
    for baseline, method in pairs:
        a = arms[baseline]['source_camera_control']
        b = arms[method]['source_camera_control']
        result[f'{method}_vs_{baseline}'] = {}
        for camera in ('predicted','calibrated'):
            comparison = {}
            for metric, sign in METRICS.items():
                reference = a[camera][metric]['real']
                estimate = b[camera][metric]['real']
                change = estimate-reference
                comparison[metric] = dict(baseline=reference,method=estimate,delta=change,
                    improvement_pct=100*sign*change/abs(reference) if reference else None)
            result[f'{method}_vs_{baseline}'][camera] = comparison
    return result


def validation_progress(folder, steps):
    """Compare each validation pass with the same head before optimization."""
    def read(step):
        rows = json.loads((folder/f'val_{step:06}.json').read_text())
        index = {(row['sequence'], row['frame']):row for row in rows}
        if not rows or len(index) != len(rows):
            raise ValueError(f'Empty or duplicate validation windows at step {step}')
        return index

    first = read(0)
    groups = {name:[key for key in first if key[0].startswith('Town05/') == eventscape]
              for name,eventscape in (('dsec',False),('eventscape',True))}
    if not all(groups.values()):
        raise ValueError('Validation must include both datasets')
    report = {}
    for step in steps:
        current = read(step)
        if current.keys() != first.keys():
            raise ValueError(f'Validation windows differ at step {step}')
        report[step] = {}
        for name,keys in groups.items():
            result = dict(windows=len(keys),metrics={})
            for metric,sign in METRICS.items():
                initial,real,zero = (fmean(rows[key][f'{arm}_{metric}'] for key in keys)
                                     for rows,arm in ((first,'matched'),(current,'matched'),(current,'zero')))
                result['metrics'][metric] = dict(initial=initial,real=real,zero=zero,
                    change_since_initial=real-initial, real_improvement=sign*(real-zero),
                    improvement_pct=100*sign*(real-zero)/abs(zero) if zero else None)
            report[step][name] = result
    return report


def summarize_runs(root):
    folder = root/'multiscene_gaussians'

    def read(path):
        return json.loads(path.read_text())

    configs = [read(folder/arm/'config.json') for arm in ('fixed','octree')]
    shared = [{key:value for key,value in spec.items() if key not in ('out','head')}
              for spec in configs]
    if shared[0] != shared[1]:
        raise ValueError('The Gaussian heads must use the same geometry and training settings')
    config = configs[0]
    if config['steps'] != 12000 or len(config['train']) != 561:
        raise ValueError('Unexpected mixed training budget or scene count')

    dsec = read(Path(config['dsec_temporal_cameras'])/'test.json')
    dsec['sequences'] = {name:row['cameras'] for name,row in dsec['sequences'].items()}
    eventscape = read(Path(config['eventscape_cameras'])/'test.json')
    report = {'head_steps':config['steps'],'train_sequences':len(config['train']),
              'train_windows':training_windows(folder,config)}
    milestones = list(range(report['train_windows'],config['steps'],report['train_windows']))
    report['validation_progress'] = {head:validation_progress(folder/head,[*milestones,config['steps']])
                                     for head in ('fixed','octree')}
    for dataset, manifest, size in (
        ('dsec',dsec,(640,480)),('eventscape',eventscape,(512,256)),
    ):
        rows = {head: read(folder/head/f'val_test_{dataset}.json') for head in ('fixed','octree')}
        report[dataset] = compare(manifest,rows,size)

    baseline = read(root/'evggs'/'mvsec_shared.json')
    rows = baseline['results']
    if baseline['dataset'] != 'mvsec' or baseline['native'] or len(rows) != 72:
        raise ValueError('Expected the 72-window shared MVSEC EvGGS reference')
    reference = aggregate([row['matched'] for row in rows])
    arms = {}
    for arm in ('dsec_temporal','eventscape_realm','mixed_fixed','mixed_octree'):
        predictions = read(folder/f'{arm}_mvsec.json')
        arms[arm] = pair_with_epf(rows,predictions,'flight',reference)
        calibrated = [row | {'matched':row['matched_calibrated']} for row in predictions]
        arms[arm]['calibrated_vs_evggs'] = pair_with_epf(rows,calibrated,'flight',reference)
        arms[arm]['event_and_zero'] = summarize(predictions)['all']
        arms[arm]['source_camera_control'] = camera_control(predictions)
    report['mvsec'] = dict(windows=len(rows),image_size=[128,96],evggs=reference,arms=arms,
                          training_domain_comparisons=compare_training_domains(arms))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    report = summarize_runs(args.root)
    temporary = args.out.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.out)
