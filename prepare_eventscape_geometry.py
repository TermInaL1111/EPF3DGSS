"""Build validated RGB-D camera references for Town05 18-frame geometry clips."""

import argparse
from contextlib import closing
import json
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from eventscape_data import EventScapeArchive, EventScapeSequence
from eventscape_gaussian_data import reference_camera


def prepare(archive, limit=None):
    y, x = np.indices((256,512))
    tiles = ((y//32+x//32)%2).astype(bool)
    sequences = {}
    attempted = 0
    for name in archive.sequences[:limit]:
        source = EventScapeSequence(archive,name)
        for start in range(0,len(source.ends)-17,18):
            attempted += 1
            first = cv2.cvtColor(source.rgb(start),cv2.COLOR_RGB2GRAY)
            depth = source.depth(start)
            cameras = []
            for index in range(start+1,start+18):
                frame = cv2.cvtColor(source.rgb(index),cv2.COLOR_RGB2GRAY)
                pose = reference_camera(first,frame,depth,tiles)
                if pose is None:
                    break
                cameras.append(pose)
            if len(cameras) == 17:
                sequences.setdefault(name,{})[str(start)] = cameras
    errors = [camera['median_reprojection_px'] for rows in sequences.values()
              for cameras in rows.values() for camera in cameras]
    if not errors:
        raise ValueError('No complete, independently checked camera clips')
    return dict(reference='first-frame RGB-D PnP, independently checked spatial tiles',
                clip_views=18, windows=len(errors)//17, attempted=attempted,
                median_reprojection_px=median(errors), sequences=sequences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--sequence-limit', type=int)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(7)
    with closing(EventScapeArchive(args.archive)) as archive:
        report = prepare(archive,args.sequence_limit)
    temporary = args.out.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.out)
    print({key:value for key,value in report.items() if key != 'sequences'},flush=True)
