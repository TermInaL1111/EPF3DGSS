"""Read aligned single-camera EventScape frames and packets from the release ZIPs."""

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np


class EventScapeArchive:
    def __init__(self, path):
        self.archive = ZipFile(path)
        self.sequences = sorted(info.filename.rstrip('/') for info in self.archive.infolist()
                                if info.is_dir() and info.filename.count('/') == 2
                                and '/sequence_' in info.filename)
        if not self.sequences:
            self.close()
            raise ValueError(f'No EventScape sequences in {path}')

    def close(self):
        self.archive.close()


class EventScapeSequence:
    """Use the packet ending at each RGB timestamp, at the native 25 Hz cadence."""

    def __init__(self, archive, sequence):
        self.archive = archive.archive
        self.sequence = sequence
        town, number = sequence.split('/sequence_')
        self.prefix = f'{int(town.removeprefix("Town")):02d}_{int(number):03d}'
        rgb = self.timestamps('rgb')
        if rgb.shape != (len(rgb), 2) or not np.array_equal(rgb, self.timestamps('events')):
            raise ValueError(f'RGB and event timestamps disagree in {sequence}')
        ids, self.times = rgb.T
        if not np.array_equal(ids, np.arange(len(ids))) or not (np.diff(self.times) > 0).all():
            raise ValueError(f'Invalid RGB time index in {sequence}')
        self.ends = range(len(ids))
        self.pairs = list(zip(self.ends, self.ends[1:]))

    def timestamps(self, modality):
        return np.loadtxt(BytesIO(self.archive.read(
            f'{self.sequence}/{modality}/data/timestamps.txt')), ndmin=2)

    def member(self, index, modality, suffix):
        return f'{self.sequence}/{modality}/data/{self.prefix}_{index:04d}_{suffix}'

    def packet(self, index):
        name = self.member(index, 'events', 'events.npz')
        with np.load(BytesIO(self.archive.read(name)), allow_pickle=False) as data:
            x, y, p, t = (data[key] for key in ('x', 'y', 'p', 't'))
        if (not (x.shape == y.shape == p.shape == t.shape) or
                np.any(x >= 512) or np.any(y >= 256) or np.any(p > 1)):
            raise ValueError(f'Invalid EventScape event packet {name}')
        # Absolute microseconds are preserved for sensor-model supervision.
        return np.stack((x, y, t, p))

    def frame(self, index):
        packet = self.packet(index).astype(np.float64)
        packet[0] *= 640 / 512
        packet[1] *= 480 / 256
        name = self.member(index, 'rgb', 'image.png')
        bgr = cv2.imdecode(np.frombuffer(self.archive.read(name), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (256, 512, 3):
            raise ValueError(f'Invalid EventScape frame {name}')
        rgb = cv2.cvtColor(cv2.resize(bgr, (224, 168), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        return packet, rgb

    def depth(self, index):
        name = self.member(index, 'depth', 'depth.npy')
        depth = np.load(BytesIO(self.archive.read(name)), allow_pickle=False)
        if depth.shape != (256, 512):
            raise ValueError(f'Invalid EventScape depth map {name}')
        return depth.astype(np.float32)


def eventscape_sequences(root, stack):
    archives = {}
    for split, filename in (('train', 'Town01-03_train.zip'), ('val', 'Town05_val.zip')):
        archive = EventScapeArchive(Path(root) / filename)
        stack.callback(archive.close)
        archives[split] = archive
    return {split: {name: EventScapeSequence(archive, name) for name in archive.sequences}
            for split, archive in archives.items()}
