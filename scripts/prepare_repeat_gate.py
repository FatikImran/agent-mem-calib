"""Validate the published MemGate checkpoint and its example on a pinned encoder."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, choices=(2,3), required=True)
    parser.add_argument('--development', type=Path, help='Saved candidate prompt/ranking artifact to score with the approved gate-only adapter')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    repeat_dir = Path(f'results/repeat-{args.repeat}')
    repeat_dir.mkdir(parents=True, exist_ok=True)
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HOME'] = str(Path('data/hf-cache').resolve())
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '1'
    from huggingface_hub import snapshot_download
    import torch
    root = Path('data/retrieval-dev')
    root.mkdir(parents=True, exist_ok=True)
    revisions = json.loads(Path('research/evidence/model-revisions.json').read_text(encoding='utf8'))
    pin_file = root / 'minilm-revision.json'
    canonical_pin = revisions['memgate_encoder']
    if pin_file.exists():
        pin = json.loads(pin_file.read_text(encoding='utf8'))
        if pin != canonical_pin:
            raise ValueError('Local MiniLM revision differs from the recorded experiment revision')
    else:
        pin = canonical_pin
        pin_file.write_text(json.dumps(pin, indent=2) + '\n', encoding='utf8')
    encoder_path = snapshot_download(**pin, allow_patterns=['*.json', 'model.safetensors',
        'vocab.txt', '1_Pooling/*', 'README.md'], max_workers=2)
    source = Path('research/evidence/repo-audit/MemGate-memgate_big_module.py')
    if hashlib.sha256(source.read_bytes()).hexdigest() != revisions['memgate_source']['sha256']:
        raise ValueError('MemGate source differs from the inspected release')
    spec = importlib.util.spec_from_file_location('audited_memgate', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    encoder = module.EmbeddingEncoder(model_name=encoder_path, device='cpu')
    defender = module.BIGDefender(embed_encoder=encoder, embed_dim=384, intent_dim=256,
        threshold=.07, device='cpu', use_memgate_single_sided=True)
    checkpoint = Path('data/audit/MemGate-checkpoint.pt')
    expected = revisions['memgate_checkpoint']
    if not checkpoint.exists():
        if os.environ.get('HF_HUB_OFFLINE') == '1':
            raise FileNotFoundError('Pinned MemGate checkpoint is unavailable offline')
        import httpx
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix('.download')
        with httpx.stream('GET', expected['url'], follow_redirects=True, timeout=60) as response:
            response.raise_for_status()
            with temporary.open('wb') as stream:
                for block in response.iter_bytes():
                    stream.write(block)
        if temporary.stat().st_size != expected['bytes'] or hashlib.sha256(temporary.read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError('Downloaded MemGate checkpoint failed verification')
        temporary.replace(checkpoint)
    if checkpoint.stat().st_size != expected['bytes'] or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected['sha256']:
        raise ValueError('MemGate checkpoint differs from the recorded release')
    defender.load(str(checkpoint))
    loaded = {'threshold': defender.threshold, 'safety_memory': defender.safety_memory,
              'objective_memory': defender.objective_memory,
              'single_sided': defender.use_memgate_single_sided}
    defender.threshold = .07
    defender.eval()
    query = 'What restaurants should I try in Rome if I cannot eat gluten?'
    memories = ['The user strictly avoids gluten because of a severe intolerance.',
                'The user likes loud sports bars.',
                'The user prefers restaurants that clearly label allergens.']
    with torch.no_grad():
        kept, scores, _ = defender(query=query, candidates=memories, top_k=None, preserve_order=True)
        all_scores = defender.score_candidates(query, memories)
    # Rule-only audit of released wrapper behavior; no task-model inference.
    qa = Path('data/Mem2ActBench/Mem2ACTBench_small/qa_dataset.jsonl')
    wrapper_rows = []
    for line in qa.read_text(encoding='utf8').splitlines():
        task = json.loads(line)
        safety = defender.should_include_safety_memory(task['query'], None)
        objective = not safety and defender.should_include_objective_memory(task['query'])
        wrapper_rows.append({'task_id': task['qa_id'], 'safety': bool(safety), 'objective': bool(objective)})
    result = {'status': 'published-example smoke test; no benchmark inference',
        'encoder': pin, 'torch_version': torch.__version__, 'checkpoint_loaded_settings': loaded,
        'effective_threshold': .07, 'top_k': None, 'preserve_order': True,
        'query': query, 'candidates': memories, 'all_scores': all_scores,
        'kept': kept, 'kept_scores': scores,
        'extra_memories': [m for m in kept if m not in memories],
        'query_wrapper_audit': {'count': len(wrapper_rows),
            'safety_count': sum(r['safety'] for r in wrapper_rows),
            'objective_count': sum(r['objective'] for r in wrapper_rows),
            'rows': wrapper_rows, 'qa_sha256': hashlib.sha256(qa.read_bytes()).hexdigest()},
        'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, checkpoint)}}
    (repeat_dir / 'memgate-runtime.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: v for k, v in result.items() if k != 'query_wrapper_audit'}, indent=2))
    print({k: v for k, v in result['query_wrapper_audit'].items() if k != 'rows'})
    if args.development:
        from src.agent.memgate_adapter import configure_gate_only, filter_gate_only
        configure_gate_only(defender)
        data = json.loads(args.development.read_text(encoding='utf8'))
        rows = []
        with torch.no_grad():
            for row in data['rows']:
                for variant in row['variants']:
                    filtered = filter_gate_only(defender, variant['judge_input']['request'], variant['memories'])
                    rows.append({'chunk_limit': row['chunk_limit'], 'task_id': row['task_id'],
                        'condition': variant['condition'], **filtered})
        artifact = {'status': 'development gate scores; no tool-call generation',
            'source': str(args.development), 'source_sha256': hashlib.sha256(args.development.read_bytes()).hexdigest(),
            'retriever': data['retriever'], 'encoder': pin, 'checkpoint_hash': result['hashes'][str(checkpoint)],
            'guidance_disabled': True, 'rows': rows}
        destination = repeat_dir / f"memgate-{data['retriever']}-development.json"
        destination.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
        print(f'Saved {len(rows)} development gate-score rows to {destination}')


if __name__ == '__main__':
    main()
