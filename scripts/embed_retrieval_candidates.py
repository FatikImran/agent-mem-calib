"""Download a revision-pinned BGE-M3 and resume fingerprinted embedding batches."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--benchmark', action='store_true', help='Measure CPU thread counts on a fixed 64-text subset, without building an index')
    args = parser.parse_args()
    root = Path('data/retrieval-dev')
    root.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(Path('data/hf-cache').resolve())
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
    from huggingface_hub import snapshot_download
    revision_path = root / 'bge-revision.json'
    canonical_pin = json.loads(Path('research/evidence/model-revisions.json').read_text(encoding='utf8'))['bge_m3']
    if revision_path.exists():
        pin = json.loads(revision_path.read_text(encoding='utf8'))
        if pin != canonical_pin:
            raise ValueError('Local BGE revision differs from the recorded experiment revision')
    else:
        pin = canonical_pin
        revision_path.write_text(json.dumps(pin, indent=2) + '\n', encoding='utf8')
    model_path = snapshot_download(**pin, allow_patterns=['*.json', '*.safetensors', 'pytorch_model.bin',
        'sentencepiece.bpe.model', '1_Pooling/*', 'README.md'], max_workers=2,
        local_files_only=args.offline)
    if not any(Path(model_path).glob('*.safetensors')) and not (Path(model_path) / 'pytorch_model.bin').exists():
        raise RuntimeError('Snapshot is missing model weights')
    snapshot_manifest = {p.relative_to(model_path).as_posix(): {
        'bytes': p.stat().st_size, 'sha256': file_sha256(p)}
        for p in sorted(Path(model_path).rglob('*')) if p.is_file()}
    receipt_path = Path('research/evidence/bge-weight-download.json')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf8'))
        if {k: receipt[k] for k in ('repo_id', 'revision')} != pin:
            raise ValueError('Model receipt and selected revision differ')
        observed = snapshot_manifest[receipt['filename']]
        if observed != {k: receipt[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Model weights differ from the verified experiment receipt')
    (root / 'bge-snapshot.json').write_text(json.dumps(snapshot_manifest, indent=2) + '\n', encoding='utf8')
    print(f'Model snapshot ready: {pin}', flush=True)
    if args.download_only:
        return
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    model = SentenceTransformer(model_path, device=args.device, trust_remote_code=False)
    records = json.loads((root / 'texts.json').read_text(encoding='utf8'))
    config = {**pin, 'texts_sha256': hashlib.sha256((root / 'texts.json').read_bytes()).hexdigest(),
        'model_snapshot_sha256': file_sha256(root / 'bge-snapshot.json'),
        'device': args.device, 'dtype': 'float32', 'batch_size': args.batch_size,
        'threads': args.threads, 'seed': 42, 'normalize_embeddings': True,
        'max_seq_length': model.max_seq_length,
        'packages': {n: importlib.metadata.version(n) for n in ('torch', 'sentence-transformers', 'transformers', 'numpy')}}
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    cache = root / 'embeddings' / key
    cache.mkdir(parents=True, exist_ok=True)
    (cache / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf8')
    # Validate actual encoder lengths before encoding; never silently truncate.
    lengths = [len(model.tokenizer(r['text'], truncation=False)['input_ids']) for r in records]
    if max(lengths) > model.max_seq_length:
        raise ValueError('Embedding tokenizer would truncate a candidate')
    if args.benchmark:
        if args.device != 'cpu':
            raise ValueError('This benchmark is CPU-only')
        texts = [r['text'] for r in records[:64]]
        model.encode(texts[:8], batch_size=8, normalize_embeddings=True, show_progress_bar=False)
        measurements = []
        reference = None
        for threads in (4, 8):
            torch.set_num_threads(threads)
            for repeat in range(2):
                began = perf_counter()
                values = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)
                elapsed = perf_counter() - began
                if reference is None:
                    reference = values
                measurements.append({'threads': threads, 'repeat': repeat, 'seconds': elapsed,
                    'max_abs_difference_from_first': float(np.max(np.abs(values - reference)))})
                print(measurements[-1], flush=True)
        Path('results/embedding-cpu-benchmark.json').write_text(json.dumps({
            'config': config, 'texts': 64, 'measurements': measurements}, indent=2) + '\n', encoding='utf8')
        return
    started = perf_counter()
    for start in range(0, len(records), 256):
        part = cache / f'{start:06d}.npy'
        size = min(256, len(records) - start)
        if part.exists():
            arr = np.load(part, allow_pickle=False)
            if arr.shape != (size, 1024) or not np.isfinite(arr).all():
                raise ValueError(f'Invalid cached batch {part}')
            continue
        values = model.encode([r['text'] for r in records[start:start + size]],
            batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False)
        if values.shape != (size, 1024) or not np.isfinite(values).all():
            raise ValueError('Invalid embedding output')
        temporary = part.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, values, allow_pickle=False)
        temporary.replace(part)
        print(f'{start + size}/{len(records)} embedded; {perf_counter() - started:.1f}s', flush=True)
    summary = {'cache_key': key, 'rows': len(records), 'max_encoder_tokens': max(lengths),
        'run_seconds': perf_counter() - started, 'config': config,
        'parts': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(cache.glob('*.npy'))}}
    (root / 'embedding-summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
