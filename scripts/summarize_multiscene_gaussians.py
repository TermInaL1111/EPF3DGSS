"""Validate and compare the mixed-training Gaussian heads across all test sets."""

import argparse
import json
from pathlib import Path

from summarize_eventscape_render import compare
from summarize_evggs import aggregate, pair_with_epf
from summarize_render import summarize


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
    report = {'head_steps':config['steps'],'train_sequences':len(config['train'])}
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
        arms[arm]['event_and_zero'] = summarize(predictions)['all']
    report['mvsec'] = dict(windows=len(rows),image_size=[128,96],evggs=reference,arms=arms)
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
