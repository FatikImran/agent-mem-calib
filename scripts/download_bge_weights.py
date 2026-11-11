"""Recover the pinned public BGE weight download using validated HTTP ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import httpx
from huggingface_hub import get_hf_file_metadata, hf_hub_url


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-partial', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    pin = json.loads(Path('data/retrieval-dev/bge-revision.json').read_text(encoding='utf8'))
    url = hf_hub_url(pin['repo_id'], 'pytorch_model.bin', revision=pin['revision'])
    metadata = get_hf_file_metadata(url)
    expected_hash, size = metadata.etag, metadata.size
    if not expected_hash or len(expected_hash) != 64 or not size:
        raise ValueError('Expected a SHA-256 LFS ETag and file size')
    int(expected_hash, 16)
    destination = Path('data/hf-cache/hub/models--BAAI--bge-m3/snapshots') / pin['revision'] / 'pytorch_model.bin'
    if destination.exists() and destination.stat().st_size == size and digest(destination) == expected_hash:
        print('Pinned model weights are already present and verified.', flush=True)
        return
    root = Path('data/retrieval-dev/weight-parts') / expected_hash
    root.mkdir(parents=True, exist_ok=True)
    block_size = 16 * 1024 * 1024
    starts = list(range(0, size, block_size))
    # Only complete prefix blocks of an interrupted HTTP stream can be reused.
    if args.seed_partial:
        with args.seed_partial.open('rb') as seed:
            for start in starts:
                part_size = min(block_size, size - start)
                if args.seed_partial.stat().st_size < start + part_size:
                    break
                part = root / f'{start:012d}.part'
                block = seed.read(part_size)
                if not part.exists():
                    part.write_bytes(block)
    def fetch(start):
        part_size = min(block_size, size - start)
        part = root / f'{start:012d}.part'
        if part.exists() and part.stat().st_size == part_size:
            return start, 'cached'
        end = start + part_size - 1
        for attempt in range(5):
            try:
                with httpx.Client(follow_redirects=True, timeout=60) as client:
                    with client.stream('GET', url, headers={'Range': f'bytes={start}-{end}'}) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get('content-range') != f'bytes {start}-{end}/{size}':
                            raise ValueError('Server did not honor exact requested range')
                        temporary = part.with_suffix('.tmp')
                        with temporary.open('wb') as stream:
                            for block in response.iter_bytes():
                                stream.write(block)
                        if temporary.stat().st_size != part_size:
                            raise ValueError('Incomplete range')
                        temporary.replace(part)
                return start, 'downloaded'
            except (httpx.HTTPError, OSError, ValueError):
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, start) for start in starts]
        for i, future in enumerate(as_completed(futures), 1):
            start, status = future.result()
            print(f'{i}/{len(starts)} ranges ready ({start}: {status})', flush=True)
    temporary = destination.with_suffix('.verified-download')
    with temporary.open('wb') as stream:
        for start in starts:
            with (root / f'{start:012d}.part').open('rb') as part:
                for block in iter(lambda: part.read(1024 * 1024), b''):
                    stream.write(block)
    observed = digest(temporary)
    if temporary.stat().st_size != size or observed != expected_hash:
        raise ValueError('Combined model fails authoritative SHA-256/size verification')
    temporary.replace(destination)
    manifest = {**pin, 'filename': 'pytorch_model.bin', 'bytes': size,
                'sha256': observed, 'verified_against': 'Hugging Face LFS ETag',
                'block_size': block_size, 'workers': args.workers}
    Path('data/retrieval-dev/weight-download.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    print('Pinned model weights verified and installed in the local snapshot.', flush=True)


if __name__ == '__main__':
    main()
