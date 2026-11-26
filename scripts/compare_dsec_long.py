"""Compare both finalized long-run arms on identical DSEC and MVSEC windows."""
import argparse
import json
from pathlib import Path
from statistics import fmean


def load(path):
    return json.loads(path.read_text())


def paired(first, second, keys, metrics):
    def index(rows):
        result = {tuple(row[key] for key in keys): row for row in rows}
        if not result or len(result) != len(rows):
            raise ValueError('Expected nonempty unique evaluation windows')
        return result

    with_f3, no_f3 = index(first), index(second)
    if with_f3.keys() != no_f3.keys():
        raise ValueError(f'Window sets differ for {keys}')
    result = {'windows': len(with_f3)}
    for metric, extract in metrics.items():
        a = [extract(row, 'matched') for row in with_f3.values()]
        b = [extract(no_f3[key], 'matched') for key in with_f3]
        z_a = [extract(row, 'zero') for row in with_f3.values()]
        z_b = [extract(no_f3[key], 'zero') for key in with_f3]
        result[metric] = dict(f3=fmean(a), no_f3=fmean(b), f3_minus_no_f3=fmean(x-y for x,y in zip(a,b)),
                              f3_zero=fmean(z_a), no_f3_zero=fmean(z_b))
    return result


def compare(f3, no_f3):
    paths = (f3,no_f3)
    geometry = [load(p/'geometry/config.json') for p in paths]
    gaussians = [load(p/'gaussians/config.json') for p in paths]
    for configs, excluded in ((geometry, {'out','f3'}), (gaussians, {'out','geometry_checkpoint'})):
        if {k:v for k,v in configs[0].items() if k not in excluded} != {
                k:v for k,v in configs[1].items() if k not in excluded}:
            raise ValueError('Training settings differ between arms')
    if not geometry[0]['f3'] or geometry[1]['f3'] or geometry[0]['max_steps'] != 20000:
        raise ValueError('Expected the paired 20,000-step F3 ablation')
    if gaussians[0]['steps'] != 5000 or gaussians[0]['geometry'] != 'predicted':
        raise ValueError('Expected a 5,000-step predicted Gaussian head')
    depth = [load(p/'depth.json') for p in paths]
    if any(d['step'] != 20000 or d['depth_cap_m'] != 50 for d in depth):
        raise ValueError('Depth reports must use the final geometry checkpoints and 50 m cap')
    render_metrics = ('psnr','ssim','ssim_evggs','lpips_vgg','lpips_alex')
    nested = dict((m,lambda r,c,m=m:r[c][m]) for m in render_metrics)
    flat = dict((m,lambda r,c,m=m:r[f'{c}_{m}']) for m in (*render_metrics,'overlap_psnr'))
    report = {}
    report['dsec_depth'] = paired(*(d['rows'] for d in depth), ('sequence','frame'),
                                  {m:(lambda r,c,m=m:r[c][m]) for m in
                                   ('absrel','delta1','valid_prediction_fraction')})
    report['dsec_render'] = paired(*(load(p/'gaussians/val_005000.json') for p in paths),
                                   ('sequence','frame'), flat)
    report['mvsec_render'] = paired(*(load(p/'mvsec.json') for p in paths), ('flight','start'), nested)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('f3','no_f3','out'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    result = compare(args.f3,args.no_f3)
    temporary = args.out.with_name(args.out.name+'.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.out)
