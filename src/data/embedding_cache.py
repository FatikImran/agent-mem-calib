"""Validate completed BGE embedding artifacts before exposing vectors to retrieval."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def load_bge_cache(root):
    root = Path(root)
    summary = json.loads((root / 'embedding-summary.json').read_text(encoding='utf8'))
    config = summary['config']
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if key != summary['cache_key'] or not re.fullmatch('[0-9a-f]{64}', key):
        raise ValueError('Embedding configuration fingerprint mismatch')
    text_file = root / 'texts.json'
    if hashlib.sha256(text_file.read_bytes()).hexdigest() != config['texts_sha256']:
        raise ValueError('Embedding/text alignment mismatch')
    records = json.loads(text_file.read_text(encoding='utf8'))
    hashes = [r['sha256'] for r in records]
    if len(set(hashes)) != len(records) or any(hashlib.sha256(r['text'].encode('utf8')).hexdigest() != r['sha256'] for r in records):
        raise ValueError('Duplicate or invalid text identities')
    cache = root / 'embeddings' / key
    if json.loads((cache / 'config.json').read_text(encoding='utf8')) != config:
        raise ValueError('Cache configuration differs from completion manifest')
    arrays, offset = [], 0
    for name, digest in sorted(summary['parts'].items()):
        if name != f'{offset:06d}.npy':
            raise ValueError('Embedding batches are missing or out of order')
        path = cache / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Embedding batch hash mismatch')
        array = np.load(path, allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != 1024 or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError('Invalid BGE vector shape, dtype or values')
        if not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-4):
            raise ValueError('BGE vectors are not normalized')
        arrays.append(array)
        offset += len(array)
    if offset != len(records) or offset != summary['rows'] or not arrays:
        raise ValueError('Incomplete embedding cache')
    matrix = np.concatenate(arrays)
    return {r['sha256']: matrix[i] for i, r in enumerate(records)}, summary
