"""Compare native-pixel frame log differences with time-aligned event counts."""

import argparse
import json
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import h5py
import numpy as np

from event_observation import polarity_counts
from eventscape_data import EventScapeArchive, EventScapeSequence
from mvsec_transfer import lower_bound_after


def log_image(image, response):
    value = image.astype(np.float32) / 255
    if response == 'inverse_srgb':
        value = np.where(value <= .04045, value / 12.92, ((value + .055) / 1.055) ** 2.4)
        # One dark digital step expressed in linear light.
        floor = 1 / (255 * 12.92)
    else:
        floor = 1 / 255
    if value.ndim == 3:
        value = value @ np.array([.2126, .7152, .0722], dtype=np.float32)
    return np.log(np.maximum(value, floor))


def unclipped(image):
    valid = (image > 2) & (image < 253)
    return valid.all(axis=-1) if image.ndim == 3 else valid


def starts(size, gap, limit):
    return np.linspace(1, size - 2 * gap - 1, min(limit, size - 2 * gap - 1), dtype=int)


def eventscape_windows(path, sequence_limit, window_limit):
    if path.name != 'Town01-03_train.zip':
        raise ValueError('Use the EventScape training archive for this diagnostic')
    with closing(EventScapeArchive(path)) as archive:
        for town in ('Town01', 'Town02', 'Town03'):
            names = [name for name in archive.sequences if name.startswith(town + '/')]
            for pos in np.linspace(0, len(names) - 1, min(sequence_limit, len(names)), dtype=int):
                seq = EventScapeSequence(archive, names[pos])
                for gap in (1, 4):
                    for start in starts(len(seq.ends), gap, window_limit):
                        stop = start + gap
                        counts = []
                        for left in (start, stop):
                            packet = np.concatenate([seq.packet(i) for i in range(left+1, left+gap+1)], axis=1)
                            seconds = packet[2].astype(np.float64) * 1e-6
                            keep = (seconds > seq.times[left]) & (seconds <= seq.times[left+gap])
                            counts.append(polarity_counts(packet[:, keep], (256, 512)))
                        yield seq.sequence, start, gap, seq.times[stop]-seq.times[start], seq.rgb(start), seq.rgb(stop), counts


def mvsec_windows(path, window_limit):
    if path.name != 'indoor_flying4_data.hdf5':
        raise ValueError('Use MVSEC indoor_flying4, the Gaussian training flight')
    with h5py.File(path) as data:
        camera = data['davis/left']
        events, frames, times = (camera[key] for key in ('events', 'image_raw', 'image_raw_ts'))
        for gap in (1, 4):
            for start in starts(len(times), gap, window_limit):
                stop = start + gap
                lo, mid, hi = [lower_bound_after(events, times[i]) for i in (start, stop, stop + gap)]
                counts = [polarity_counts(events[a:b].T, frames.shape[1:]) for a, b in ((lo, mid), (mid, hi))]
                yield 'indoor_flying4', start, gap, times[stop]-times[start], frames[start], frames[stop], counts


def summarize(windows, thresholds):
    # Metrics use a common support with at least one true event and unsaturated endpoints.
    summaries = defaultdict(list)
    for name, start, gap, duration, first, last, (aligned, later) in windows:
        valid = unclipped(first) & unclipped(last)
        mask = valid & (aligned.sum(0) > 0)
        if not mask.any():
            continue
        signed = [counts[0][mask] - counts[1][mask] for counts in (aligned, later)]
        for response in ('published_code', 'inverse_srgb'):
            change = log_image(last, response)[mask] - log_image(first, response)[mask]
            reliable = (np.abs(change) >= .05) & (signed[0] != 0)
            entry = dict(sequence=name, start=int(start), dt_ms=float(duration * 1000),
                         active=int(mask.sum()), unclipped=int(valid.sum()),
                         sign_agrees=int((np.sign(change[reliable]) == np.sign(signed[0][reliable])).sum()),
                         sign_total=int(reliable.sum()), abs_change_mean=float(np.abs(change).mean()),
                         thresholds={})
            for c in thresholds:
                residual = np.abs(change - c*signed[0])
                entry['thresholds'][str(c)] = dict(
                    aligned=float(residual.mean()), shifted=float(np.abs(change - c*signed[1]).mean()),
                    reversed=float(np.abs(change + c*signed[0]).mean()),
                    within_threshold=float((residual <= c).mean()))
            summaries[(gap, response)].append(entry)
    blocks = []
    for (gap, response), rows in summaries.items():
        def average(values):
            return float(np.mean(list(values)))

        signs = sum(row['sign_total'] for row in rows)
        blocks.append(dict(gap_frames=gap, response=response, windows=len(rows),
            mean_dt_ms=average(row['dt_ms'] for row in rows),
            active_fraction=float(sum(row['active'] for row in rows)/sum(row['unclipped'] for row in rows)),
            sign_agreement=sum(row['sign_agrees'] for row in rows)/signs if signs else None,
            no_change_mae=average(row['abs_change_mean'] for row in rows),
            threshold_mae={str(c): {key: average(row['thresholds'][str(c)][key] for row in rows)
                           for key in ('aligned', 'shifted', 'reversed', 'within_threshold')}
                           for c in thresholds}, rows=rows))
    if not blocks:
        raise ValueError('No windows with active pixels and unsaturated endpoints')
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('eventscape', 'mvsec'))
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--sequences-per-town', type=int, default=6)
    parser.add_argument('--windows-per-gap', type=int, default=10)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[.15, .2, .25])
    args = parser.parse_args()
    if min(args.sequences_per_town, args.windows_per_gap, *args.thresholds) <= 0:
        parser.error('Sample counts and thresholds must be positive')
    if args.dataset == 'eventscape':
        windows = eventscape_windows(args.source, args.sequences_per_town, args.windows_per_gap)
    else:
        windows = mvsec_windows(args.source, args.windows_per_gap)
    blocks = summarize(windows, args.thresholds)
    report = dict(dataset=args.dataset, source=args.source.name, blocks=blocks)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
