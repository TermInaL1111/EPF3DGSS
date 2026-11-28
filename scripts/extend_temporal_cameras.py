"""Add one more target camera offset to the temporal event manifests.

The existing DSEC and EventScape manifests store source-relative cameras for
the midpoint (2 frames) and the second source (4 frames).  This script keeps
those entries and appends a farther target camera, for example 8 frames, so the
same frozen head can be evaluated on a genuinely larger viewpoint change.
"""

import argparse
import json
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from dsec_temporal_gaussian_data import DSECTemporalGaussianSequence
from eventscape_gaussian_data import (ARCHIVES, EventScapeArchive, EventScapeGaussianSequence,
                                      NATIVE_K, reference_camera)


def as_dict(cameras):
    if isinstance(cameras, dict):
        return dict(cameras)
    return {str(2 + 2 * index): camera for index, camera in enumerate(cameras)}


def extend_dsec(root, record, offset):
    sequences = {}
    for name, row in record['sequences'].items():
        cameras = row['cameras']
        source = DSECTemporalGaussianSequence(root, name)
        y, x = np.indices((480, 640))
        tiles = ((y // 32 + x // 32) % 2).astype(bool)
        updated = {}
        for key, entries in cameras.items():
            index = int(key)
            entry = as_dict(entries)
            if str(offset) not in entry and index + offset < len(source.times):
                reference = source.gray(index)
                depth = cv2.resize(source.depth(index), (640, 480),
                                   interpolation=cv2.INTER_NEAREST_EXACT)
                camera = reference_camera(reference, source.gray(index + offset), depth,
                                          tiles, source.rgb_k, 50.)
                if camera:
                    entry[str(offset)] = camera
            updated[key] = entry
        source.close()
        sequences[name] = dict(row, cameras=updated)
        print(name, len(updated), flush=True)
    return dict(record, sequences=sequences, target_offsets=sorted(
        {int(k) for row in sequences.values() for cam in row['cameras'].values() for k in cam}))


def extend_eventscape(root, record, split, offset):
    sequences = {}
    with closing(EventScapeArchive(root / ARCHIVES[split])) as archive:
        for name, entries in record['sequences'].items():
            source = EventScapeGaussianSequence(archive, name, entries)
            depth_source = source.sequence
            y, x = np.indices((256, 512))
            tiles = ((y // 32 + x // 32) % 2).astype(bool)
            updated = {}
            for key, entries_at_window in entries.items():
                index = int(key)
                entry = as_dict(entries_at_window)
                if str(offset) not in entry and index + offset < len(depth_source.ends):
                    reference = cv2.cvtColor(depth_source.rgb(index), cv2.COLOR_RGB2GRAY)
                    target = cv2.cvtColor(depth_source.rgb(index + offset), cv2.COLOR_RGB2GRAY)
                    camera = reference_camera(reference, target, depth_source.depth(index),
                                              tiles, NATIVE_K, 200.)
                    if camera:
                        entry[str(offset)] = camera
                updated[key] = entry
            sequences[name] = updated
            print(name, len(updated), flush=True)
    return dict(record, sequences=sequences, target_offsets=sorted(
        {int(k) for row in sequences.values() for cam in row.values() for k in cam}))


def main(args):
    cv2.setNumThreads(1)
    for split in args.splits:
        path = args.cameras / f'{split}.json'
        record = json.loads(path.read_text())
        result = (extend_dsec(args.dsec_root, record, args.offset) if args.dataset == 'dsec'
                  else extend_eventscape(args.eventscape_root, record, split, args.offset))
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f'{split}.json').write_text(
            json.dumps(result, indent=1, allow_nan=False) + '\n')
        print('wrote', args.out / f'{split}.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('dsec', 'eventscape'), required=True)
    parser.add_argument('--cameras', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--offset', type=int, default=8)
    parser.add_argument('--splits', nargs='+', default=('test', 'val'))
    parser.add_argument('--dsec-root', type=Path)
    parser.add_argument('--eventscape-root', type=Path)
    args = parser.parse_args()
    main(args)
