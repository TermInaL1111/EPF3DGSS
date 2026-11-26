"""Compare DSEC distillation runs on the same validation windows and epoch."""
import argparse
import json
from pathlib import Path
from statistics import fmean


def report_at(paths, epoch):
    reference = None
    reference_settings = None
    results = {}
    for path in paths:
        rows = json.loads((path/f'val_{epoch:03}.json').read_text())
        keys = [(row['seq'],tuple(row['frames'])) for row in rows]
        if not rows or len(set(keys)) != len(keys):
            raise ValueError(f'Empty or duplicate validation windows in {path}')
        if reference is None:
            reference = set(keys)
        if set(keys) != reference:
            raise ValueError(f'Validation windows differ in {path}')
        config = json.loads((path/'config.json').read_text())
        settings = {name:config[name] for name in
                    ('train','val','seed','epochs','train_pairs','val_pairs','steps_per_epoch')}
        if reference_settings is None:
            reference_settings = settings
        if settings != reference_settings:
            raise ValueError(f'Training or validation protocol differs in {path}')
        results[path.name] = dict(f3=bool(config['f3']),realm=config['realm'],windows=len(rows),
                                  matched=fmean(row['matched'] for row in rows),
                                  zero=fmean(row['empty'] for row in rows),
                                  matched_minus_zero=fmean(r['matched']-r['empty'] for r in rows))
    return dict(epoch=epoch,runs=results)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--epoch',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.write_text(json.dumps(report_at(args.runs,args.epoch),indent=2,allow_nan=False)+'\n')
