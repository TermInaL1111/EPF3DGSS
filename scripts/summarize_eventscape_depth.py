"""Verify matched distillation schedules and summarize EventScape depth."""

import argparse
import json
import math
from pathlib import Path
from statistics import fmean


SETTINGS = {'direct':('frozen',False), 'saag':('saag',False), 'realm':('saag',True)}


def read(path):
    return json.loads(path.read_text())


def schedule(folder):
    with (folder/'geometry/train.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream if line.endswith('\n')]
    return [(r['step'],r['epoch'],r['sequence'],tuple(r['frames']),r['K']) for r in rows]


def verify(folders, prefix=False):
    settings = {}
    runs = {}
    for arm,folder in folders.items():
        config = read(folder/'geometry/config.json')
        if (config['step_gating'],config['realm']) != SETTINGS[arm]:
            raise ValueError(f'Unexpected gates or masks for {arm}')
        if config['schedule_steps'] != 20000 or not config['eventscape'] or config['f3']:
            raise ValueError(f'Expected 20,000 EventScape steps without F³ for {arm}')
        settings[arm] = {k:v for k,v in config.items()
                         if k not in ('out','realm','realm_progress_epochs','step_gating')}
        runs[arm] = schedule(folder)
    base = runs['direct']
    if len(base) != 20000 or any(s[0] != i for i,s in enumerate(base,1)):
        raise ValueError('Incomplete direct distillation schedule')
    for arm,rows in runs.items():
        if settings[arm] != settings['direct'] or not rows or rows != base[:len(rows)]:
            raise ValueError(f'Training setup, sequence, clip or shared K differs for {arm}')
        if not prefix and len(rows) != len(base):
            raise ValueError(f'Distillation schedule is incomplete for {arm}')
    return min(map(len,runs.values()))


def compare(folders):
    import torch

    matched_steps = verify(folders)
    records = {}
    for arm,folder in folders.items():
        report = read(folder/'town05_depth.json')
        checkpoint = torch.load(folder/'geometry/latest.pt',weights_only=True,map_location='cpu')
        if checkpoint['step'] != 20000 or Path(report['checkpoint']).resolve() != (folder/'geometry/latest.pt').resolve():
            raise ValueError(f'Bad depth checkpoint provenance for {arm}')
        protocol = (report['clip_views'],report['recurrent_steps'],report['max_depth_m'],report['depth_alignment'])
        if protocol != (18,16,200,'median ratio per clip on jointly valid native pixels'):
            raise ValueError(f'Unexpected depth evaluation protocol for {arm}')
        rows = {(r['sequence'],r['start']):r for r in report['rows']}
        if len(rows) != 1368 or len(rows) != len(report['rows']):
            raise ValueError(f'Expected 1,368 unique Town05 windows for {arm}')
        records[arm] = rows
    base = records['direct']
    for arm,rows in records.items():
        if rows.keys() != base.keys():
            raise ValueError(f'Town05 windows differ for {arm}')
        for key,row in rows.items():
            reference = base[key]
            if (row['scored_pixels'] != row['labelled_pixels'] or
                    row['labelled_pixels'] != reference['labelled_pixels']):
                raise ValueError(f'Depth evaluation uses different pixel support for {arm}/{key}')
            if any(not math.isclose(row['teacher'][metric],reference['teacher'][metric],abs_tol=2e-6)
                   for metric in ('absrel','delta1')):
                raise ValueError(f'Same-window RGB teacher changed for {arm}/{key}')
    directions = {'absrel':-1,'delta1':1}
    scores = {arm:{metric:fmean(r['student'][metric] for r in rows.values()) for metric in directions}
              for arm,rows in records.items()}
    scores['teacher'] = {metric:fmean(r['teacher'][metric] for r in base.values()) for metric in directions}
    references = [('teacher','direct'),('direct','realm')] if 'realm' in records else [('teacher','direct')]
    if 'saag' in records:
        references += [('direct','saag')]
        if 'realm' in records:
            references += [('saag','realm')]
    relative = {f'{arm}_vs_{baseline}':{
        metric:100*sign*(scores[arm][metric]-scores[baseline][metric])/abs(scores[baseline][metric])
        if scores[baseline][metric] else None for metric,sign in directions.items()}
        for baseline,arm in references}
    return dict(matched_training_steps=matched_steps,windows=len(base),
                scores=scores,relative_improvement_pct=relative)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--arm',choices=('realm','saag'),action='append',required=True)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    folders = {arm:args.root/f'eventscape_{arm}_20k' for arm in ('direct',*args.arm)}
    if args.out:
        report = compare(folders)
        temporary = args.out.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temporary.replace(args.out)
    else:
        print(verify(folders,prefix=True),'matched training steps')
