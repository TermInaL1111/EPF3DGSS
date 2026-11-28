"""Compare matched photometric and LPIPS continuation on held-out MVSEC."""
import argparse
import json
from pathlib import Path

from summarize_multiscene_gaussians import METRICS, camera_control


def summarize(root, reference):
    arms = ('photometric', 'perceptual')
    configs = [json.loads((root/arm/'config.json').read_text()) for arm in arms]
    shared = [{key:value for key,value in config.items() if key not in ('out','lpips_weight')}
              for config in configs]
    if shared[0] != shared[1] or [c.get('lpips_weight',0) for c in configs] != [0,.5]:
        raise ValueError('Expected a matched continuation differing only in LPIPS supervision')
    schedules = []
    rows = {}
    reference_keys = [(row['flight'],row['start']) for row in reference]
    for arm in arms:
        with (root/arm/'train.jsonl').open() as stream:
            schedule = [(row['step'],row['sequence'],row['frame']) for row in map(json.loads,stream)]
        if len(schedule) != 4034 or any(row[0] != step for step,row in enumerate(schedule,1)):
            raise ValueError(f'Incomplete continuation for {arm}')
        if len({row[1:] for row in schedule}) != len(schedule):
            raise ValueError(f'Repeated training window for {arm}')
        schedules.append(schedule)
        rows[arm] = json.loads((root/arm/'mvsec.json').read_text())
        if len(set(reference_keys)) != 72 or reference_keys != [(r['flight'],r['start']) for r in rows[arm]]:
            raise ValueError(f'Incompatible MVSEC windows for {arm}')
    if schedules[0] != schedules[1]:
        raise ValueError('The continuation arms used different sample orders')
    scores = {'initial':camera_control(reference), **{arm:camera_control(rows[arm]) for arm in arms}}
    compared = {}
    for camera in ('predicted','calibrated'):
        compared[camera] = {}
        for metric,sign in METRICS.items():
            base,new = (scores[arm][camera][metric]['real'] for arm in arms)
            compared[camera][metric] = dict(delta=new-base,
                improvement_pct=100*sign*(new-base)/abs(base) if base else None)
    return dict(steps=4034, windows=72, scores=scores, perceptual_vs_photometric=compared)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ('root','reference','out'):
        parser.add_argument(path,type=Path)
    args = parser.parse_args()
    result = summarize(args.root,json.loads(args.reference.read_text()))
    temporary = args.out.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.out)
