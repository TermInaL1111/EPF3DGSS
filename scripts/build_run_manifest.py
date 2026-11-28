"""Record every head variant, checkpoint and frozen-head evaluation."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean
import sys

ROOT = Path('/data/userwj/epf-runs')
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results')

RUNS = [
    dict(id='octree_target4_12k', head='octree', train='DSEC+EventScape', target=4,
         steps=12000, dir='octree_target4_12k', status='finished',
         archive='octree_target4_12k.pt',
         tests={'dsec': 'val_test_dsec.json', 'eventscape': 'val_test_eventscape.json'}),
    dict(id='vg2gt_target4_12k', head='vg2gt (fine voxel)', train='DSEC+EventScape', target=4,
         steps=4600, dir='vg2gt_target4_12k', status='stopped at 4.6k (DSEC geometry collapse)',
         tests={'dsec': '../vg2gt_target4_eval_46k/val_test_dsec.json',
                'eventscape': '../vg2gt_target4_eval_46k/val_test_eventscape.json'}),
    dict(id='vg2gt_octree_target4_12k', head='vg2gt_octree', train='DSEC+EventScape', target=4,
         steps=12000, dir='vg2gt_octree_target4_12k', status='finished',
         tests={'dsec': 'val_test_dsec.json', 'eventscape': 'val_test_eventscape.json'}),
    dict(id='octree_pixel_es_12k', head='octree_pixel', train='EventScape', target=4,
         steps=12000, dir='octree_pixel_es_12k', status='finished',
         archive='octree_pixel_es_12k.pt',
         tests={'eventscape': 'val_test.json'}),
    dict(id='octree_pixel_es_4k_eval', head='octree_pixel', train='EventScape', target=4,
         steps=4200, dir='octree_pixel_es_eval_4k', status='intermediate evaluation',
         tests={'eventscape': 'val_test.json'}),
    dict(id='octree_pixel_dsec_zeroshot', head='octree_pixel', train='EventScape', target=4,
         steps=12000, dir='octree_pixel_dsec_zeroshot', status='zero-shot (failed)',
         tests={'dsec': 'val_test.json'}),
    dict(id='octree_pixel_mixed_12k', head='octree_pixel', train='DSEC+EventScape', target=4,
         steps=12000, dir='octree_pixel_mixed_12k', status='finished',
         archive='octree_pixel_mixed_12k.pt',
         tests={'dsec': 'val_test_dsec.json', 'eventscape': 'val_test_eventscape.json'}),
    dict(id='octree_pixel_center_es_12k', head='octree_pixel_center', train='EventScape', target=4,
         steps=12000, dir='octree_pixel_center_es_12k', status='finished',
         archive='octree_pixel_center_es_12k.pt',
         tests={'eventscape': 'val_test.json'}),
    dict(id='octree_pixel_center_es_7k_eval', head='octree_pixel_center', train='EventScape', target=4,
         steps=7500, dir='octree_pixel_center_es_eval_7k', status='intermediate evaluation',
         tests={'eventscape': 'val_test.json'}),
    dict(id='octree_pixel_center_mixed_12k', head='octree_pixel_center', train='DSEC+EventScape',
         target=4, steps=12000, dir='octree_pixel_center_mixed_12k', status='finished',
         tests={'dsec': 'val_test_dsec.json', 'eventscape': 'val_test_eventscape.json'}),
    dict(id='octree_pixel_fine_es_12k', head='octree_pixel_fine (2.5 cm)', train='EventScape',
         target=4, steps=4900, dir='octree_pixel_fine_es_12k',
         status='stopped at 4.9k (no gain vs 5 cm)', tests={}),
    dict(id='octree_pixel_deep_es_12k', head='octree_pixel_deep (48ch x3)', train='EventScape',
         target=4, steps=3100, dir='octree_pixel_deep_es_12k',
         status='stopped at 3.1k (+0.014 dB at 2k)', tests={}),
    dict(id='octree_pixel_detail_es_12k', head='octree_pixel + high-res F3', train='EventScape',
         target=4, steps=4500, dir='octree_pixel_detail_es_12k',
         status='stopped at 4.5k (no gain at 4k)', tests={}),
    dict(id='octree_pixel_es_40k', head='octree_pixel', train='EventScape', target=4,
         steps=3900, dir='octree_pixel_es_40k', status='stopped (marginal gain)',
         tests={}),
    dict(id='octree_target8_eval', head='octree', train='DSEC+EventScape', target=8,
         steps=12000, dir='octree_target8_eval', status='target-8 held-out evaluation',
         tests={'dsec': 'val_test_dsec.json', 'eventscape': 'val_test_eventscape.json'}),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def scores(path):
    rows = json.loads(path.read_text())
    return dict(
        windows=len(rows),
        psnr=fmean(row['matched_psnr'] for row in rows),
        ssim=fmean(row['matched_ssim'] for row in rows),
        lpips_vgg=fmean(row['matched_lpips_vgg'] for row in rows),
        lpips_alex=fmean(row['matched_lpips_alex'] for row in rows),
        zero_psnr=fmean(row['zero_psnr'] for row in rows),
        gaussians=fmean(row.get('matched_gaussian_count', 0) for row in rows),
    )


manifest = {'runs': []}
for spec in RUNS:
    directory = ROOT / spec['dir']
    entry = {key: spec[key] for key in ('id', 'head', 'train', 'target', 'steps', 'status')}
    entry['run_dir'] = str(directory)
    checkpoint = directory / 'latest.pt'
    entry['checkpoint'] = str(checkpoint) if checkpoint.is_file() else None
    entry['checkpoint_sha256'] = sha256(checkpoint) if checkpoint.is_file() else None
    entry['checkpoint_bytes'] = checkpoint.stat().st_size if checkpoint.is_file() else None
    archive = spec.get('archive')
    if archive:
        archived = Path('weights') / archive
        entry['archived_weights'] = str(archived)
        entry['archived_weights_sha256'] = sha256(archived) if archived.is_file() else None
    for name in ('config.json', 'train.jsonl', 'val_012000.json'):
        file = directory / name
        if file.is_file():
            entry[name.replace('.', '_')] = str(file)
    entry['eval'] = {}
    for dataset, relative in spec['tests'].items():
        path = (directory / relative).resolve()
        if path.is_file():
            entry['eval'][dataset] = dict(path=str(path), **scores(path))
    manifest['runs'].append(entry)

comparison = ROOT/'source_view_supervision/comparison.json'
if comparison.is_file():
    report = json.loads(comparison.read_text())
    steps = report['steps']
    for arm,checkpoints in report['checkpoints'].items():
        latest = checkpoints['latest']
        directory = Path(latest['path']).parent
        evaluations = {}
        for dataset in ('dsec','eventscape','mvsec'):
            result = report[dataset]
            metrics = result['scores'][arm]
            condition = 'matched'
            if dataset == 'mvsec':
                metrics = metrics['predicted']
                condition = 'real'
            filename = 'mvsec.json' if dataset == 'mvsec' else f'val_test_{dataset}.json'
            evaluations[dataset] = dict(
                path=str(directory/filename), windows=result['windows'],
                zero_psnr=metrics['psnr']['zero'],
                **{metric:metrics[metric][condition] for metric in ('psnr','ssim','lpips_vgg','lpips_alex')},
            )
        manifest['runs'].append(dict(
            id=f'source_view_{arm}_{steps}', head='octree_pixel', train='DSEC+EventScape',
            target=4, steps=steps, status='finished', run_dir=str(directory),
            checkpoint=latest['path'], checkpoint_sha256=latest['sha256'], checkpoint_bytes=latest['bytes'],
            initial_checkpoint=report['initial_checkpoint'], checkpoints=checkpoints,
            config_json=str(directory/'config.json'), train_jsonl=str(directory/'train.jsonl'),
            comparison_json=str(comparison), eval=evaluations,
        ))

OUT.mkdir(parents=True,exist_ok=True)
json_out = OUT/'run_manifest.json'
json_tmp = json_out.with_suffix('.json.tmp')
json_tmp.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
json_tmp.replace(json_out)
csv_out = OUT/'run_manifest.csv'
csv_tmp = csv_out.with_suffix('.csv.tmp')
with csv_tmp.open('w',newline='') as stream:
    writer = csv.writer(stream,lineterminator='\n')
    fields = ('id','head','train','target','steps','status')
    datasets = {'dsec':'dsec','eventscape':'es','mvsec':'mvsec'}
    metrics = ('psnr','lpips_vgg')
    writer.writerow([*fields,*(f'{prefix}_{metric}' for prefix in datasets.values() for metric in metrics),
                     'checkpoint'])
    for entry in manifest['runs']:
        values = [entry['eval'].get(dataset,{}).get(metric)
                  for dataset in datasets for metric in metrics]
        writer.writerow([*(entry[key] for key in fields),
                         *(f'{value:.4f}' if value is not None else '' for value in values),
                         entry['checkpoint'] or ''])
csv_tmp.replace(csv_out)
print(f'Recorded {len(manifest["runs"])} runs in {OUT}')
