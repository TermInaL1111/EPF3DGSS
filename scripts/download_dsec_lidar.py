"""Extract selected DSEC LiDAR/IMU drives from the official remote ZIP."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import struct
from tempfile import TemporaryFile
import time
from zipfile import ZIP_STORED, ZipFile
from zlib import crc32

import requests


URL = 'https://download.ifi.uzh.ch/rpg/DSEC/lidar_imu.zip'
BLOCK = 16 << 20


def fetch(first, last=None):
    byte_range = f'{first}-{last}' if last is not None else str(first)
    expected_size = last-first+1 if last is not None else -first
    for attempt in range(6):
        try:
            with requests.get(URL, params={'range':byte_range}, headers={'Range':f'bytes={byte_range}'},
                              stream=True, timeout=(20,90)) as response:
                response.raise_for_status()
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                if response.status_code != 206 or not match:
                    raise OSError(f'Server did not honor range {byte_range}')
                start, end, size = map(int,match.groups())
                expected_start = first if last is not None else size+first
                if start != expected_start or end-start+1 != expected_size:
                    raise OSError(f'Incorrect byte range {start}-{end} for {byte_range}')
                data = b''.join(response.iter_content(1 << 20))
                if len(data) != expected_size:
                    raise OSError(f'Incomplete response for {byte_range}')
                return data, size
        except (requests.RequestException, OSError):
            if attempt == 5:
                raise
            time.sleep(2**attempt)


def inventory():
    tail, size = fetch(-65536)
    with TemporaryFile() as sparse:
        sparse.seek(size-len(tail))
        sparse.write(tail)
        with ZipFile(sparse) as archive:
            return {info.filename:info for info in archive.infolist()}


def checksum(path):
    check = 0
    with path.open('rb') as source:
        for block in iter(lambda:source.read(BLOCK), b''):
            check = crc32(block,check)
    return check


def download_member(target, info):
    if info.compress_type != ZIP_STORED or info.flag_bits & 1:
        raise ValueError(f'Expected an uncompressed, unencrypted archive member: {info.filename}')
    if target.exists():
        if target.stat().st_size != info.file_size or checksum(target) != info.CRC:
            raise ValueError(f'Existing file does not match the archive: {target}')
        return
    header, _ = fetch(info.header_offset,info.header_offset+29)
    if header[:4] != b'PK\x03\x04':
        raise ValueError(f'Invalid local ZIP header for {info.filename}')
    filename_bytes, extra_bytes = struct.unpack_from('<HH',header,26)
    start = info.header_offset+30+filename_bytes+extra_bytes
    target.parent.mkdir(parents=True,exist_ok=True)
    partial = target.with_suffix(target.suffix+'.part')
    written = partial.stat().st_size if partial.exists() else 0
    if written > info.file_size:
        raise ValueError(f'Partial file exceeds the archive member: {partial}')
    print(f'{target} {written:,}/{info.file_size:,} bytes',flush=True)
    with partial.open('ab') as output:
        for offset in range(written,info.file_size,BLOCK):
            last = min(offset+BLOCK,info.file_size)-1
            data, _ = fetch(start+offset,start+last)
            output.write(data)
    if partial.stat().st_size != info.file_size or checksum(partial) != info.CRC:
        raise ValueError(f'Archive checksum mismatch for {partial}')
    partial.replace(target)
    print(f'Ready {target}',flush=True)


def download_drive(root, listing, drive):
    for filename in ('cam_to_lidar.yaml','lidar_imu.bag'):
        download_member(root/drive/filename,listing[f'data/{drive}/{filename}'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('drives',nargs='+')
    args = parser.parse_args()
    listing = inventory()
    drives = sorted(set(args.drives))
    for drive in drives:
        if f'data/{drive}/lidar_imu.bag' not in listing:
            parser.error(f'Unknown DSEC drive {drive}')
    with ThreadPoolExecutor(max_workers=min(2,len(drives))) as pool:
        list(pool.map(lambda drive:download_drive(args.root,listing,drive),drives))
