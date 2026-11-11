"""Verify benchmark identity against the inspected public Git revision."""
import hashlib
import json
from pathlib import Path


def verify():
    receipt = json.loads(Path('research/evidence/benchmark-revision.json').read_text(encoding='utf8'))
    observed = {}
    for name, expected in receipt['files'].items():
        raw = (Path('data/Mem2ActBench') / name).read_bytes()
        normalized = raw.replace(b'\r\n', b'\n')
        sha = hashlib.sha256(normalized).hexdigest()
        blob = hashlib.sha1(f'blob {len(normalized)}\0'.encode() + normalized).hexdigest()
        if sha != expected['sha256_lf'] or blob != expected['git_blob']:
            raise ValueError(f'Benchmark file differs from the recorded release: {name}')
        observed[name] = {'sha256_raw': hashlib.sha256(raw).hexdigest(),
            'sha256_lf': sha, 'git_blob': blob, 'records': len(normalized.splitlines())}
    return {'commit': receipt['commit'], 'canonicalization': receipt['canonicalization'],
            'files': observed, 'status': 'all files match the inspected public revision'}


if __name__ == '__main__':
    result = verify()
    Path('results/benchmark-release-verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(result['status'])
