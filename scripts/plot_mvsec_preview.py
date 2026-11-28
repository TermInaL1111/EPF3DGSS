"""Plot EvGGS and EPF renderings on the identical MVSEC query frames."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text())


def by_window(rows):
    indexed = {(row['flight'],row['start']):row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError('Duplicate preview window')
    return indexed


def load_panel(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('L'))
    if pixels.shape != (96,128):
        raise ValueError(f'Unexpected comparison image dimensions in {path}')
    return pixels


def plot(run, published, output):
    baseline = by_window(read(run/'evggs.json')['results'])
    octree = by_window(read(run/'octree.json'))
    all_windows = by_window(read(published)['results'])
    expected = {}
    for flight in ('indoor_flying1','indoor_flying2','indoor_flying3'):
        keys = sorted(key for key in all_windows if key[0]==flight)
        if len(keys) != 24:
            raise ValueError(f'Expected 24 published windows for {flight}')
        expected.update((keys[i],all_windows[keys[i]]) for i in (0,11,23))
    if baseline.keys() != expected.keys() or octree.keys() != expected.keys():
        raise ValueError('Previews differ from the selected shared windows')
    fig,axes = plt.subplots(len(expected),4,figsize=(7.2,13),layout='constrained')
    for row,((flight,start),reference) in enumerate(expected.items()):
        key = (flight,start)
        prefix = f'{flight}_{start}'
        target = load_panel(run/'evggs'/f'{prefix}_target.png')
        if not np.array_equal(target,load_panel(run/'octree'/f'{prefix}_target.png')):
            raise ValueError(f'Target images differ for {prefix}')
        scores = [None,baseline[key]['matched']['psnr'],
                  octree[key]['matched_calibrated']['psnr'],octree[key]['zero_calibrated']['psnr']]
        if abs(scores[1]-reference['matched']['psnr']) > .05:
            raise ValueError(f'EvGGS score changed for {prefix}')
        panels = [
            target,
            load_panel(run/'evggs'/f'{prefix}_matched.png'),
            load_panel(run/'octree'/f'{prefix}_matched_calibrated.png'),
            load_panel(run/'octree'/f'{prefix}_zero_calibrated.png'),
        ]
        titles = ['Target gray', 'EvGGS', 'EPF voxels', 'EPF, zero events']
        for col,(ax,image,score) in enumerate(zip(axes[row],panels,scores)):
            ax.imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            ax.set_axis_off()
            if row==0:
                ax.set_title(titles[col],fontsize=11)
            label = f'Flight {flight[-1]} · frame {start+2}' if score is None else f'{score:.2f} dB'
            ax.text(.5,-.035,label,ha='center',va='top',transform=ax.transAxes,fontsize=8)
    fig.savefig(output,dpi=165,facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run','published','output'):
        parser.add_argument(key,type=Path)
    args = parser.parse_args()
    plot(args.run,args.published,args.output)
