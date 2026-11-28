"""Compare matched source-view training arms on the three rendering benchmarks."""

import argparse
import json
from math import dist, isfinite
from pathlib import Path
from statistics import fmean

from summarize_attribute_supervision import manifest_windows, metric_change, read_json
from summarize_multiscene_gaussians import METRICS, camera_control, validation_progress
from summarize_render import summarize as summarize_renders


def index(rows, sequence_key, frame_key):
    keys = [(row[sequence_key],row[frame_key]) for row in rows]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('Expected nonempty unique evaluation windows')
    return set(keys)


def motion_groups(arms, motion, metrics):
    if any(rows.keys() != motion.keys() for rows in arms.values()):
        raise ValueError('Motion records and render results use different windows')
    ordered = sorted(motion,key=lambda key:(motion[key],key))
    if len(ordered) < 3 or any(not isfinite(motion[key]) or motion[key] < 0 for key in ordered):
        raise ValueError('Expected at least three windows with finite camera displacement')
    width, remainder = divmod(len(ordered),3)
    groups = {}
    for i,name in enumerate(('short','medium','long')):
        start = i*width+min(i,remainder)
        group = ordered[start:start+width+(i<remainder)]
        scores = {
            arm:{condition:{metric:fmean(metrics(rows[key],condition,metric) for key in group)
                            for metric in METRICS}
                 for condition in ('matched','zero')}
            for arm,rows in arms.items()
        }
        changes = {metric:metric_change(scores['target']['matched'][metric],
                                        scores['source']['matched'][metric],sign)
                   for metric,sign in METRICS.items()}
        groups[name] = dict(windows=len(group),displacement_m=[motion[group[0]],motion[group[-1]]],
                            scores=scores,source_vs_target=changes)
    return groups


def temporal_motion(config,dataset):
    camera_key = 'dsec_temporal_cameras' if dataset=='dsec' else 'eventscape_cameras'
    sequences = read_json(Path(config[camera_key])/'test.json')['sequences']
    offset = config['target_offset']
    motion = {}
    for name,rows in sequences.items():
        if dataset=='dsec':
            rows = rows['cameras']
        for frame,records in rows.items():
            query = records[str(offset)] if isinstance(records,dict) else records[(offset-2)//2]
            motion[name,int(frame)] = dist([row[3] for row in query['c2w'][:3]],(0,0,0))
    return motion


def summarize(root, steps, mvsec_reference):
    folders = {arm:root/f'{arm}_{steps}' for arm in ('target','source')}
    configs = {arm:read_json(folder/'config.json') for arm,folder in folders.items()}
    shared = [{key:value for key,value in config.items() if key not in ('out','source_view_weight')}
              for config in configs.values()]
    config = shared[0]
    if (shared[0] != shared[1] or config['steps'] != steps
            or configs['target'].get('source_view_weight',0) != 0
            or configs['source'].get('source_view_weight') != .5):
        raise ValueError('Training arms differ outside the source-view loss')
    schedules = {}
    for arm,folder in folders.items():
        with (folder/'train.jsonl').open() as stream:
            rows = list(map(json.loads,stream))
        if len(rows) != steps or any(r['step'] != i for i,r in enumerate(rows,1)):
            raise ValueError(f'Incomplete training log for {arm}')
        if any(('source_view_loss' in row) != (arm=='source') for row in rows):
            raise ValueError(f'Unexpected source-view loss logs for {arm}')
        schedules[arm] = [(r['sequence'],r['frame']) for r in rows]
    if schedules['target'] != schedules['source']:
        raise ValueError('Training window order differs')
    expected_train = set.union(*(manifest_windows(config,dataset,'train')
                                 for dataset in ('dsec','eventscape')))
    width = len(expected_train)
    for start in range(0,steps,width):
        epoch = schedules['target'][start:start+width]
        if len(set(epoch)) != len(epoch) or set(epoch)-expected_train:
            raise ValueError('Training windows differ from the expected manifest')

    report = dict(steps=steps,unique_training_windows=width,
                  validation_progress={arm:validation_progress(folder,[*range(width,steps,width),steps])
                                       for arm,folder in folders.items()})
    for dataset in ('dsec','eventscape'):
        filename = f'val_test_{dataset}.json'
        expected = manifest_windows(config,dataset,'test')
        scores = {}
        by_window = {}
        for arm,folder in dict(initial=Path(config['init_head']).parent,**folders).items():
            rows = read_json(folder/filename)
            if index(rows,'sequence','frame') != expected:
                raise ValueError(f'Unexpected {dataset} evaluation window for {arm}')
            scores[arm] = summarize_renders(rows)['all']
            by_window[arm] = {(r['sequence'],r['frame']):r for r in rows}
        changes = {metric:metric_change(scores['target'][metric]['matched'],
                                        scores['source'][metric]['matched'],sign)
                   for metric,sign in METRICS.items()}
        groups = motion_groups(by_window,temporal_motion(config,dataset),
                               lambda row,condition,metric:row[f'{condition}_{metric}'])
        report[dataset] = dict(windows=len(expected),scores=scores,source_vs_target=changes,
                               motion_groups=groups)

    expected = index(read_json(mvsec_reference)['results'],'flight','start')
    scores = {}
    by_window = {}
    for arm,folder in folders.items():
        rows = read_json(folder/'mvsec.json')
        if index(rows,'flight','start') != expected:
            raise ValueError(f'Unexpected MVSEC evaluation window for {arm}')
        scores[arm] = camera_control(rows)
        by_window[arm] = {(r['flight'],r['start']):r for r in rows}
    motion = {key:row['query_baseline_m'] for key,row in by_window['target'].items()}
    if any(abs(by_window['source'][key]['query_baseline_m']-value) > 1e-6
           for key,value in motion.items()):
        raise ValueError('MVSEC query displacement differs between arms')
    changes = {camera:{metric:metric_change(scores['target'][camera][metric]['real'],
                                            scores['source'][camera][metric]['real'],sign)
                       for metric,sign in METRICS.items()} for camera in ('predicted','calibrated')}
    report['mvsec'] = dict(windows=len(expected),scores=scores,source_vs_target=changes,
                           predicted_camera_protocol='known intrinsics and event-predicted relative pose',
                           motion_groups={
                               camera:motion_groups(by_window,motion,
                                   lambda row,condition,metric:row[f'{condition}{suffix}'][metric])
                               for camera,suffix in (('predicted',''),('calibrated','_calibrated'))
                           })
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('mvsec_reference',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--steps',type=int,default=12000)
    args = parser.parse_args()
    report = summarize(args.root,args.steps,args.mvsec_reference)
    temporary = args.out.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.out)
