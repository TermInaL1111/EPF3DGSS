"""Compare matched source-view training arms on the three rendering benchmarks."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from math import dist, isfinite
from pathlib import Path
from random import Random
from statistics import fmean, quantiles
from zipfile import ZipFile

from summarize_attribute_supervision import manifest_windows, metric_change, read_json
from summarize_evggs import aggregate, pair_with_epf
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
    reference, candidate = (arm for arm in arms if arm != 'initial')
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
        changes = {metric:metric_change(scores[reference]['matched'][metric],
                                        scores[candidate]['matched'][metric],sign)
                   for metric,sign in METRICS.items()}
        groups[name] = dict(windows=len(group),displacement_m=[motion[group[0]],motion[group[-1]]],
                            scores=scores)
        groups[name][f'{candidate}_vs_{reference}'] = changes
    return groups


def scene_comparison(arms, score):
    target,source = (windows for arm,windows in arms.items() if arm != 'initial')
    if target.keys() != source.keys():
        raise ValueError('Scene comparison requires paired windows')
    grouped = defaultdict(list)
    for key in sorted(target):
        grouped[key[0]].append(key)
    counts = [len(keys) for keys in grouped.values()]
    results = {}
    for metric,sign in METRICS.items():
        totals = [sum(sign*(score(source[key],metric)-score(target[key],metric)) for key in keys)
                  for keys in grouped.values()]
        per_scene = {name:total/count for name,total,count in zip(grouped,totals,counts,strict=True)}
        interval = None
        if len(counts) >= 10:
            rng = Random(20261128)
            samples = []
            for _ in range(10000):
                selected = rng.choices(range(len(counts)),k=len(counts))
                samples.append(sum(totals[i] for i in selected)/sum(counts[i] for i in selected))
            cuts = quantiles(samples,n=40,method='inclusive')
            interval = [cuts[0],cuts[-1]]
        results[metric] = dict(improvement=sum(totals)/sum(counts),per_scene=per_scene,
                               scene_bootstrap_95_interval=interval)
    return dict(scenes=len(counts),metrics=results)


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


def evggs_comparison(reference, arms):
    rows = reference['results']
    if reference['dataset'] != 'mvsec' or reference['native'] or len(rows) != 72:
        raise ValueError('Expected the official checkpoint on the 72 shared MVSEC windows')
    baseline = aggregate([row['matched'] for row in rows])
    comparisons = {}
    for arm,indexed in arms.items():
        aligned = []
        for row in rows:
            candidate = indexed[row['flight'],row['start']]
            if any(candidate[key] != row[key] for key in ('sources','target')):
                raise ValueError(f'EvGGS source or target frames differ for {arm}')
            if ('query_raster_intrinsics' not in row or
                    candidate.get('query_raster_intrinsics') != row['query_raster_intrinsics']):
                raise ValueError(f'EvGGS query raster intrinsics differ or are missing for {arm}')
            aligned.append(candidate | {'matched':candidate['matched_calibrated']})
        comparisons[arm] = pair_with_epf(rows,aligned,'flight',baseline)
    return dict(protocol='shared virtual cameras, calibrated source poses, 128x96 full grayscale',
                query_raster_intrinsics=rows[0]['query_raster_intrinsics'],
                weights=reference['weights'], weights_sha256=reference['sha256'],
                upstream_commit=reference['upstream_commit'], baseline=baseline, arms=comparisons)


def checkpoint_record(path):
    digest = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            digest.update(block)
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=digest.hexdigest())


def checkpoint_members(path):
    with ZipFile(path) as archive:
        return {name.split('/',1)[1]:sha256(archive.read(name)).hexdigest()
                for name in archive.namelist()}


def summarize_validation(root, steps, milestone):
    folders = {arm:root/f'{arm}_{steps}' for arm in ('target','source')}
    reports = {arm:validation_progress(folder,[milestone])[milestone]
               for arm,folder in folders.items()}
    keys = {}
    for arm,folder in folders.items():
        rows = read_json(folder/f'val_{milestone:06}.json')
        keys[arm] = index(rows,'sequence','frame')
    if keys['target'] != keys['source']:
        raise ValueError('Validation windows differ between training arms')
    datasets = {}
    for dataset in reports['target']:
        baseline = reports['target'][dataset]['metrics']
        estimate = reports['source'][dataset]['metrics']
        datasets[dataset] = dict(
            windows=reports['target'][dataset]['windows'],
            target=baseline,
            source=estimate,
            source_vs_target={metric:metric_change(baseline[metric]['real'],estimate[metric]['real'],sign)
                              for metric,sign in METRICS.items()},
        )
    return dict(step=milestone,split='validation',datasets=datasets)


def summarize(root, steps, mvsec_reference, rgb_teacher=False):
    reference,candidate = ('source','rgb_teacher') if rgb_teacher else ('target','source')
    difference = f'{candidate}_vs_{reference}'
    folders = {arm:root/f'{arm}_{steps}' for arm in (reference,candidate)}
    configs = {arm:read_json(folder/'config.json') for arm,folder in folders.items()}
    variant = 'rgb_teacher_weight' if rgb_teacher else 'source_view_weight'
    shared = [{key:value for key,value in config.items() if key not in ('out',variant)}
              for config in configs.values()]
    config = shared[0]
    if (shared[0] != shared[1] or config['steps'] != steps
            or configs[reference].get(variant,0) != 0
            or configs[candidate].get(variant) != (.1 if rgb_teacher else .5)):
        raise ValueError(f'Training arms differ outside the declared {variant} contrast')
    if config['save_every'] != 4000:
        raise ValueError('Expected checkpoints every 4,000 steps')
    checkpoint_steps = sorted(set(range(4000,steps+1,4000)) | {steps})
    checkpoints = {}
    for arm,folder in folders.items():
        records = {step:checkpoint_record(folder/f'step_{step:06}.pt') for step in checkpoint_steps}
        latest = folder/'latest.pt'
        if checkpoint_members(Path(records[steps]['path'])) != checkpoint_members(latest):
            raise ValueError(f'Final checkpoint differs from the latest checkpoint for {arm}')
        records['latest'] = checkpoint_record(latest)
        checkpoints[arm] = records
    schedules = {}
    for arm,folder in folders.items():
        with (folder/'train.jsonl').open() as stream:
            rows = list(map(json.loads,stream))
        if len(rows) != steps or any(r['step'] != i for i,r in enumerate(rows,1)):
            raise ValueError(f'Incomplete training log for {arm}')
        if any(('source_view_loss' in row) != (arm != 'target') for row in rows):
            raise ValueError(f'Unexpected source-view loss logs for {arm}')
        if any(('rgb_teacher_loss' in row) != (arm == 'rgb_teacher') for row in rows):
            raise ValueError(f'Unexpected RGB teacher loss logs for {arm}')
        schedules[arm] = [(r['sequence'],r['frame']) for r in rows]
    if schedules[reference] != schedules[candidate]:
        raise ValueError('Training window order differs')
    expected_train = set.union(*(manifest_windows(config,dataset,'train')
                                 for dataset in ('dsec','eventscape')))
    width = len(expected_train)
    for start in range(0,steps,width):
        epoch = schedules[reference][start:start+width]
        if len(set(epoch)) != len(epoch) or set(epoch)-expected_train:
            raise ValueError('Training windows differ from the expected manifest')

    report = dict(steps=steps,unique_training_windows=width,
                  initial_checkpoint=checkpoint_record(Path(config['init_head'])),
                  checkpoints=checkpoints,
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
        changes = {metric:metric_change(scores[reference][metric]['matched'],
                                        scores[candidate][metric]['matched'],sign)
                   for metric,sign in METRICS.items()}
        groups = motion_groups(by_window,temporal_motion(config,dataset),
                               lambda row,condition,metric:row[f'{condition}_{metric}'])
        report[dataset] = dict(windows=len(expected),scores=scores,
                               motion_groups=groups,
                               scene_comparison=scene_comparison(by_window,
                                   lambda row,metric:row[f'matched_{metric}']), **{difference:changes})

    baseline = read_json(mvsec_reference)
    expected = index(baseline['results'],'flight','start')
    scores = {}
    by_window = {}
    for arm,folder in folders.items():
        rows = read_json(folder/'mvsec.json')
        if index(rows,'flight','start') != expected:
            raise ValueError(f'Unexpected MVSEC evaluation window for {arm}')
        scores[arm] = camera_control(rows)
        by_window[arm] = {(r['flight'],r['start']):r for r in rows}
    motion = {key:row['query_baseline_m'] for key,row in by_window[reference].items()}
    if any(abs(by_window[candidate][key]['query_baseline_m']-value) > 1e-6
           for key,value in motion.items()):
        raise ValueError('MVSEC query displacement differs between arms')
    changes = {camera:{metric:metric_change(scores[reference][camera][metric]['real'],
                                            scores[candidate][camera][metric]['real'],sign)
                       for metric,sign in METRICS.items()} for camera in ('predicted','calibrated')}
    report['mvsec'] = dict(windows=len(expected),scores=scores,
                           evggs=evggs_comparison(baseline,by_window),
                           predicted_camera_protocol='known intrinsics and event-predicted relative pose',
                           scene_comparison=scene_comparison(by_window,lambda row,metric:row['matched'][metric]),
                           motion_groups={
                               camera:motion_groups(by_window,motion,
                                   lambda row,condition,metric:row[f'{condition}{suffix}'][metric])
                               for camera,suffix in (('predicted',''),('calibrated','_calibrated'))
                           }, **{difference:changes})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('mvsec_reference',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--steps',type=int,default=12000)
    parser.add_argument('--validation-step',type=int,help='Summarize a shared validation milestone only')
    parser.add_argument('--rgb-teacher',action='store_true',help='Compare RGB feature supervision with source grayscale alone')
    args = parser.parse_args()
    if args.validation_step is None:
        report = summarize(args.root,args.steps,args.mvsec_reference,args.rgb_teacher)
    else:
        if args.rgb_teacher:
            parser.error('--validation-step is for the source-view comparison')
        report = summarize_validation(args.root,args.steps,args.validation_step)
    temporary = args.out.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.out)
