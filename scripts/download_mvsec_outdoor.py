"""Download official MVSEC outdoor HDF5 pairs and calibration."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from secrets import token_hex
import shutil
import time
from zipfile import ZipFile

import h5py
import requests


# Official MVSEC HDF5 folders linked from https://daniilidis-group.github.io/mvsec/download/.
FILES = {
    'outdoor_day/outdoor_day1_data.hdf5': '1JLIrw2L24zIQBmqaWvef7G2t9tsMY3H0',
    'outdoor_day/outdoor_day1_gt.hdf5': '1wzUmTBxQ5wtSpB0KBogliB2IGTrCtJ7e',
    'outdoor_day/outdoor_day_calib.zip': '1Y0sPP0ebX_cEKUCVVhJLej9TZgCxS3ME',
    'outdoor_night/outdoor_night1_data.hdf5': '1z8b00gWoZnGuzAOSD49KFaX03q1UuKxc',
    'outdoor_night/outdoor_night1_gt.hdf5': '139dZNXHNUtSul0ZLmPu6N39gbvciQZme',
    'outdoor_night/outdoor_night_calib.zip': '1NGUBQ8b41b9murJualNeaO7M3nrIPjfm',
}
# Additional public HDF5 pairs in the official fallback Drive folder.
EXTRA_NIGHTS = {
    'outdoor_night/outdoor_night2_data.hdf5': '1rRvRC5ZG-autd9NFV9qyB1pwsJtPX0G2',
    'outdoor_night/outdoor_night2_gt.hdf5': '1-UAt4ZGIJ5JyxAd72AqXWbGoeJyylRoo',
    'outdoor_night/outdoor_night3_data.hdf5': '1ADVyZmczdaEBBeCo2ebwt5nvx4-ZZf8L',
    'outdoor_night/outdoor_night3_gt.hdf5': '1eMVz4BziJMBobmgsA0izEJrvDmSd-MDy',
}
CHUNK = 16 * 1024 * 1024


def fetch_range(file_id, first, last, path=None, expected_size=None):
    url = 'https://drive.usercontent.google.com/download'
    params = dict(id=file_id, export='download', confirm='t', chunk=f'{first}-{last}')
    for attempt in range(6):
        request_params = params if attempt == 0 else params | {'retry':token_hex(8)}
        try:
            with requests.get(url, params=request_params, headers={'Range': f'bytes={first}-{last}'},
                              stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                if response.status_code != 206 or not match:
                    raise OSError(f'No byte range {first}-{last}; HTTP {response.status_code}, '
                                  f'content type {response.headers.get("Content-Type")}')
                start, end, size = map(int, match.groups())
                if (start, end) != (first, last) or (expected_size and size != expected_size):
                    raise OSError('Unexpected file range or size')
                if path is None:
                    return size
                temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as output:
                    for data in response.iter_content(1 << 20):
                        output.write(data)
                if temporary.stat().st_size != end-start+1:
                    raise OSError('Incomplete response')
                temporary.replace(path)
                return size
        except (requests.RequestException, OSError):
            if attempt == 5:
                raise
            time.sleep(min(30, 2**attempt))


def inspect(path, name):
    if name.endswith('.zip'):
        with ZipFile(path) as archive:
            if archive.testzip() or not any('camchain' in item for item in archive.namelist()):
                raise ValueError(f'Invalid calibration archive {path}')
        return
    with h5py.File(path) as archive:
        left = archive['davis/left']
        keys = ('events', 'image_raw', 'image_raw_ts') if name.endswith('_data.hdf5') else (
            'depth_image_rect', 'depth_image_rect_ts', 'pose', 'pose_ts')
        if any(key not in left or len(left[key]) == 0 for key in keys):
            raise ValueError(f'Missing DAVIS data in {path}')


def download(target, file_id, pool):
    size = fetch_range(file_id, 0, 0)
    if target.exists():
        if target.stat().st_size != size:
            raise ValueError(f'Existing file has a different byte count: {target}')
        inspect(target, target.name)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    chunks = target.with_name(target.name+'.parts')
    chunks.mkdir(parents=True, exist_ok=True)
    pending = []
    parts = []
    for first in range(0, size, CHUNK):
        last = min(first+CHUNK, size)-1
        part = chunks/f'{first:012}'
        parts.append(part)
        if not part.exists() or part.stat().st_size != last-first+1:
            pending.append(pool.submit(fetch_range, file_id, first, last, part, size))
    print(f'{target.name} {size:,} bytes, {len(pending)} chunks remaining', flush=True)
    for i, future in enumerate(pending, 1):
        future.result()
        if i % 100 == 0:
            print(f'{target.name} {i}/{len(pending)} requested chunks verified', flush=True)
    temporary = target.with_name(target.name+'.tmp')
    with temporary.open('wb') as output:
        for part in parts:
            with part.open('rb') as source:
                shutil.copyfileobj(source, output, 1 << 20)
    if temporary.stat().st_size != size:
        raise ValueError(f'Assembled file has a different byte count: {target}')
    inspect(temporary, target.name)
    temporary.replace(target)
    shutil.rmtree(chunks)
    print(f'Ready {target}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--extra-nights', action='store_true', help='Download only the night2/night3 HDF5 pairs')
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for relative, file_id in (EXTRA_NIGHTS if args.extra_nights else FILES).items():
            download(args.root/relative, file_id, pool)
