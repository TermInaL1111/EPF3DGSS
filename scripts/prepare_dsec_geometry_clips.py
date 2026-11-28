"""Build nonoverlapping, timestamp-checked DSEC monocular geometry clips."""

import argparse
from contextlib import closing
import json
from pathlib import Path

import numpy as np

from dsec_data import DSECSequence


VIEWS = 18
EVENT_WINDOW_US = 50000
MAX_FRAME_GAP_US = 150000
DEPTH_CAP_M = 50


def sequence_clips(root, name):
    with closing(DSECSequence(root,name,window_us=EVENT_WINDOW_US,monocular=True)) as sequence:
        indices = np.asarray([
            i for i in sequence.indices
            if (sequence.root/f'disparity/event/{i:06}.png').is_file()
            and (sequence.root/f'images/left/rectified/{i:06}.png').is_file()
        ],dtype=np.int64)
        gaps = np.diff(sequence.times[indices])
        chunks = np.split(indices, np.flatnonzero((gaps<=0)|(gaps>MAX_FRAME_GAP_US))+1)
        clips = [chunk[i:i+VIEWS].tolist() for chunk in chunks
                 for i in range(0,len(chunk)-VIEWS+1,VIEWS)]
        if not clips:
            raise ValueError(f'No complete {VIEWS}-view clips for {name}')
        return dict(eligible_frames=len(indices),windows=clips)


def prepare(root, split):
    names = split['val']+split['test']
    if len(names) != len(set(names)) or set(names) & set(split['train']):
        raise ValueError('Overlapping DSEC splits')
    return dict(clip_views=VIEWS, event_window_us=EVENT_WINDOW_US, depth_cap_m=DEPTH_CAP_M,
                max_frame_gap_us=MAX_FRAME_GAP_US,
                sequences={name:sequence_clips(root,name) for name in names})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','split','out'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    report = prepare(args.root,json.loads(args.split.read_text()))
    temporary = args.out.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,separators=(',',':'))+'\n')
    temporary.replace(args.out)
