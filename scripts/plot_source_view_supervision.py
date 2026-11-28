"""Plot the same held-out target for both source-supervision arms."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def pixels(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('L'))


def read(path):
    return json.loads(path.read_text())


def temporal_panel(arms, dataset):
    rows = [read(arm/f'val_test_{dataset}.json')[0] for arm in arms]
    keys = [(row['sequence'],row['frame']) for row in rows]
    if keys[0] != keys[1]:
        raise ValueError(f'{dataset} previews use different windows')
    target,baseline,_ = np.hsplit(pixels(arms[0]/f'preview_test_{dataset}.png'),3)
    other,estimate,zero = np.hsplit(pixels(arms[1]/f'preview_test_{dataset}.png'),3)
    if not np.array_equal(target,other):
        raise ValueError(f'{dataset} targets differ')
    return target,baseline,estimate,zero


def mvsec_panel(arms, reference):
    rows = [read(arm/'mvsec_preview.json') for arm in arms]
    keys = [[(r['flight'],r['start']) for r in part] for part in rows]
    if keys[0] != keys[1] or not set(keys[0]) <= reference:
        raise ValueError('MVSEC previews differ from the paired evaluation')
    flight,start = [key for key in keys[0] if key[0]=='indoor_flying1'][1]
    name = f'{flight}_{start}'
    target = pixels(arms[0]/'mvsec_preview'/f'{name}_target.png')
    if not np.array_equal(target,pixels(arms[1]/'mvsec_preview'/f'{name}_target.png')):
        raise ValueError('MVSEC targets differ')
    return (target,pixels(arms[0]/'mvsec_preview'/f'{name}_matched.png'),
            pixels(arms[1]/'mvsec_preview'/f'{name}_matched.png'),
            pixels(arms[1]/'mvsec_preview'/f'{name}_zero.png'))


def plot(root, output, steps):
    arms = [root/f'{name}_{steps}' for name in ('target','source')]
    reference = {(r['flight'],r['start']) for r in read(arms[0]/'mvsec.json')}
    examples = {
        'DSEC':temporal_panel(arms,'dsec'),
        'EventScape':temporal_panel(arms,'eventscape'),
        'MVSEC':mvsec_panel(arms,reference),
    }
    fig,axes = plt.subplots(3,4,figsize=(10,7),layout='constrained')
    titles = ('Target','Target loss','Target and source losses','Source-loss model, zero events')
    for row,(dataset,views) in enumerate(examples.items()):
        for col,(axis,view) in enumerate(zip(axes[row],views,strict=True)):
            axis.imshow(view,cmap='gray',vmin=0,vmax=255)
            axis.set(xticks=[],yticks=[])
            if row==0:
                axis.set_title(titles[col],fontsize=9)
            if col==0:
                axis.set_ylabel(dataset)
    fig.savefig(output,dpi=160,facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--steps',type=int,default=12000)
    args = parser.parse_args()
    plot(args.root,args.out,args.steps)
