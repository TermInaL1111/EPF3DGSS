"""Extract only the DSEC event, RGB and disparity files used by this project."""
import argparse
import os
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZipFile


def main(root, sequences):
    for archive, prefixes in {
        'train_events.zip': ('events/left/', 'events/right/'),
        'train_images.zip': ('images/',),
        'train_calibration.zip': ('calibration/',),
        'train_disparity.zip': ('disparity/image/', 'disparity/timestamps.txt'),
    }.items():
        with ZipFile(root/archive) as z:
            files = {seq: 0 for seq in sequences}
            for item in z.infolist():
                parts = PurePosixPath(item.filename).parts
                if not parts or parts[0] not in files or item.is_dir():
                    continue
                if not '/'.join(parts[1:]).startswith(prefixes):
                    continue
                if '..' in parts:
                    raise ValueError(f'Invalid archive path {item.filename}')
                files[parts[0]] += 1
                target = root/'extracted'/Path(*parts)
                if target.is_file() and target.stat().st_size == item.file_size:
                    continue
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary = target.with_name(target.name+f'.{os.getpid()}.partial')
                try:
                    with z.open(item) as src, temporary.open('wb') as dst:
                        shutil.copyfileobj(src,dst,length=4<<20)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            if not all(files.values()):
                raise ValueError(f'{archive} missing sequences: {files}')
            print(archive,files,flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('sequences',nargs='+')
    args = p.parse_args()
    main(args.root,args.sequences)
