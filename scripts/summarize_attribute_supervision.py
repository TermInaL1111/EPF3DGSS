"""Compare matched photometric and LPIPS continuation on the three test sets."""
import argparse
import json
from pathlib import Path
from statistics import median

from summarize_multiscene_gaussians import METRICS, camera_control, validation_progress
from summarize_render import summarize as summarize_renders


ARMS = ('photometric', 'perceptual')


def read_json(path):
    return json.loads(path.read_text())


def metric_change(base, estimate, sign):
    delta = estimate-base
    return dict(delta=delta, improvement_pct=100*sign*delta/abs(base) if base else None)


def manifest_windows(config, dataset, split):
    key = 'dsec_temporal_cameras' if dataset == 'dsec' else 'eventscape_cameras'
    manifest = read_json(Path(config[key])/f'{split}.json')
    sequences = manifest['sequences']
    if dataset == 'dsec':
        sequences = {name:row['cameras'] for name,row in sequences.items()}
    return {(name,int(index)) for name,indices in sequences.items() for index in indices}


def temporal_test(root, initial, config, dataset, image_size):
    expected = manifest_windows(config,dataset,'test')
    folders = {arm:root/arm for arm in ARMS}
    scores = {}
    for arm,folder in dict(initial=initial, **folders).items():
        rows = read_json(folder/f'val_test_{dataset}.json')
        if not expected or len(rows) != len(expected) or {(r['sequence'],r['frame']) for r in rows} != expected:
            raise ValueError(f'Unexpected {dataset} test windows for {arm}')
        summary = summarize_renders(rows)['all']
        scores[arm] = {
            metric: {
                **summary[metric],
                'real_vs_zero':metric_change(summary[metric]['zero'],summary[metric]['matched'],sign),
            }
            for metric,sign in METRICS.items()
        }
        scores[arm]['gaussian_count_median'] = {
            condition:median(r[f'{condition}_gaussian_count'] for r in rows)
            for condition in ('matched','zero')
        }
    compared = {
        condition: {
            metric:metric_change(scores['photometric'][metric][condition],
                                 scores['perceptual'][metric][condition],sign)
            for metric,sign in METRICS.items()
        }
        for condition in ('matched','zero')
    }
    return dict(windows=len(expected), image_size=list(image_size), scores=scores,
                perceptual_vs_photometric=compared)


def summarize(root, reference_path):
    reference = read_json(reference_path)
    configs = [read_json(root/arm/'config.json') for arm in ARMS]
    shared = [{key:value for key,value in config.items() if key not in ('out','lpips_weight')}
              for config in configs]
    if shared[0] != shared[1] or [c.get('lpips_weight',0) for c in configs] != [0,.5]:
        raise ValueError('Expected a matched continuation differing only in LPIPS supervision')
    config = configs[0]
    datasets = dict(dsec=(640,480), eventscape=(512,256))
    training = {dataset:manifest_windows(config,dataset,'train') for dataset in datasets}
    expected = set.union(*training.values())
    steps = config['steps']
    if steps != len(expected) or steps != sum(map(len,training.values())):
        raise ValueError('Expected one full pass over the two disjoint training manifests')
    schedules = []
    rows = {}
    reference_keys = [(row['flight'],row['start']) for row in reference]
    for arm in ARMS:
        with (root/arm/'train.jsonl').open() as stream:
            schedule = [(row['step'],row['sequence'],row['frame']) for row in map(json.loads,stream)]
        if len(schedule) != steps or any(row[0] != step for step,row in enumerate(schedule,1)):
            raise ValueError(f'Incomplete continuation for {arm}')
        if {row[1:] for row in schedule} != expected:
            raise ValueError(f'Training manifest coverage differs for {arm}')
        schedules.append(schedule)
        rows[arm] = read_json(root/arm/'mvsec.json')
        if len(set(reference_keys)) != 72 or reference_keys != [(r['flight'],r['start']) for r in rows[arm]]:
            raise ValueError(f'Incompatible MVSEC windows for {arm}')
    if schedules[0] != schedules[1]:
        raise ValueError('The continuation arms used different sample orders')
    scores = {'initial':camera_control(reference), **{arm:camera_control(rows[arm]) for arm in ARMS}}
    compared = {}
    for camera in ('predicted','calibrated'):
        compared[camera] = {}
        for metric,sign in METRICS.items():
            base,new = (scores[arm][camera][metric]['real'] for arm in ARMS)
            compared[camera][metric] = metric_change(base,new,sign)
    return dict(
        steps=steps,
        train_windows={dataset:len(windows) for dataset,windows in training.items()},
        validation_progress={arm:validation_progress(root/arm,[steps]) for arm in ARMS},
        mvsec=dict(windows=72, scores=scores, perceptual_vs_photometric=compared),
        **{dataset:temporal_test(root,Path(config['init_head']).parent,config,dataset,size)
           for dataset,size in datasets.items()},
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ('root','reference','out'):
        parser.add_argument(path,type=Path)
    args = parser.parse_args()
    result = summarize(args.root,args.reference)
    temporary = args.out.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.out)
