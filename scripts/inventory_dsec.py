"""List DSEC training assets from local ZIP indexes without extracting data."""
import argparse
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def inventory(root):
    archives, sequences = {}, {}
    for kind in ('events', 'images', 'disparity', 'calibration'):
        path = root/f'train_{kind}.zip'
        archives[kind] = path.stat().st_size
        with ZipFile(path) as archive:
            for entry in archive.infolist():
                parts = PurePosixPath(entry.filename).parts
                if entry.is_dir() or len(parts) < 2:
                    continue
                record = sequences.setdefault(parts[0],{}).setdefault(
                    kind, {'files':0,'uncompressed_bytes':0})
                record['files'] += 1
                record['uncompressed_bytes'] += entry.file_size
    result = []
    for name,components in sorted(sequences.items()):
        path = root/'extracted'/name
        extracted = all((path/p).is_file() for p in ('images/timestamps.txt',
                        'events/left/events.h5','calibration/cam_to_cam.yaml'))
        result.append(dict(sequence=name,components=components,
                           left_training_entrypoints_present=extracted))
    return dict(archive_bytes=archives,sequence_count=len(result),sequences=result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('out',type=Path)
    args = p.parse_args()
    args.out.write_text(json.dumps(inventory(args.root),indent=2)+'\n')
