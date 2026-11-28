"""Record every head variant, checkpoint and frozen-head evaluation."""

import hashlib
import json
from pathlib import Path
from statistics import fmean

ROOT = Path('/data/userwj/epf-runs')
OUT = Path('results')

RUNS = [
    dict(id='octree_target4_12k', head='octree', train='DSEC+EventScape', target=4,
         steps=12000, dir='octree_target4_12k', status='finished',
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
         tests={'eventscape': 'val_test.json'}),
    dict(id='octree_pixel_es_4k_eval', head='octree_pixel', train='EventScape', target=4,
         steps=4200, dir='octree_pixel_es_eval_4k', status='intermediate evaluation',
         tests={'eventscape': 'val_test.json'}),
    dict(id='octree_pixel_dsec_zeroshot', head='octree_pixel', train='EventScape', target=4,
         steps=12000, dir='octree_pixel_dsec_zeroshot', status='zero-shot (failed)',
         tests={'dsec': 'val_test.json'}),
    dict(id='octree_pixel_mixed_12k', head='octree_pixel', train='DSEC+EventScape', target=4,
         steps=12000, dir='octree_pixel_mixed_12k', status='finished',
         tests={'dsec': 'val_test_dsec.json', 'eventscape': 'val_test_eventscape.json'}),
    dict(id='octree_pixel_center_es_12k', head='octree_pixel_center', train='EventScape', target=4,
         steps=12000, dir='octree_pixel_center_es_12k', status='finished',
         tests={'eventscape': 'val_test.json'}),
    dict(id='octree_pixel_center_es_7k_eval', head='octree_pixel_center', train='EventScape', target=4,
         steps=7500, dir='octree_pixel_center_es_eval_7k', status='intermediate evaluation',
         tests={'eventscape': 'val_test.json'}),
    dict(id='octree_pixel_center_mixed_12k', head='octree_pixel_center', train='DSEC+EventScape',
         target=4, steps=12000, dir='octree_pixel_center_mixed_12k', status='running',
         tests={}),
    dict(id='octree_pixel_fine_es_12k', head='octree_pixel_fine (2.5 cm)', train='EventScape',
         target=4, steps=12000, dir='octree_pixel_fine_es_12k', status='running',
         tests={}),
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

(OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

lines = ['id,head,train,target,steps,status,dsec_psnr,dsec_lpips_vgg,es_psnr,es_lpips_vgg,checkpoint']
for entry in manifest['runs']:
    dsec = entry['eval'].get('dsec', {})
    es = entry['eval'].get('eventscape', {})
    lines.append(','.join(str(value) for value in (
        entry['id'], entry['head'], entry['train'], entry['target'], entry['steps'], entry['status'],
        f"{dsec.get('psnr', ''):.4f}" if dsec else '',
        f"{dsec.get('lpips_vgg', ''):.4f}" if dsec else '',
        f"{es.get('psnr', ''):.4f}" if es else '',
        f"{es.get('lpips_vgg', ''):.4f}" if es else '',
        entry['checkpoint'] or '',
    )))
(OUT / 'run_manifest.csv').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
