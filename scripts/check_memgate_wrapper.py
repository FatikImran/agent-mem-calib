"""Prepare or run the approved affected-query wrapper sensitivity panel."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter

INPUT = Path('results/memgate-wrapper-inputs.json')
OUTPUT = Path('results/memgate-wrapper-traces.json')
FIRST_PASS = Path('results/first-pass')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, data):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    tmp.replace(p)


def prepare():
    os.environ.update(HF_HUB_OFFLINE='1', HF_HOME=str(Path('data/hf-cache').resolve()), TORCH_FORCE_WEIGHTS_ONLY_LOAD='1')
    import torch
    from huggingface_hub import snapshot_download
    pins = json.loads(Path('research/evidence/model-revisions.json').read_bytes())
    source = Path('research/evidence/repo-audit/MemGate-memgate_big_module.py')
    checkpoint = Path('data/audit/MemGate-checkpoint.pt')
    assert sha(source) == pins['memgate_source']['sha256']
    assert sha(checkpoint) == pins['memgate_checkpoint']['sha256']
    spec = importlib.util.spec_from_file_location('released_memgate', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    encoder = module.EmbeddingEncoder(model_name=snapshot_download(**pins['memgate_encoder'], local_files_only=True,
        allow_patterns=['*.json', 'model.safetensors', 'vocab.txt', '1_Pooling/*', 'README.md']), device='cpu')
    defender = module.BIGDefender(embed_encoder=encoder, embed_dim=384, intent_dim=256, threshold=.07, device='cpu', use_memgate_single_sided=True)
    defender.load(str(checkpoint))
    defender.threshold = .07
    defender.eval()
    baseline = Path('data/retrieval-dev/eval-baseline.json')
    paired = Path('data/retrieval-dev/eval-paired.json')
    audit = Path('results/memgate-runtime.json')
    affected = {r['task_id'] for r in json.loads(audit.read_bytes())['query_wrapper_audit']['rows'] if r['safety'] or r['objective']}
    assert len(affected) == 13
    pairs = {r['task_id']:r['variants'] for r in json.loads(paired.read_bytes())['rows']}
    split = json.loads(Path('research/evidence/heldout-split-v1.json').read_bytes())
    rows = []
    for parent in json.loads(baseline.read_bytes())['rows']:
        tid = parent['task_id']
        if tid not in affected:
            continue
        variants = [v for v in parent['variants'] if v['condition'] == 'baseline_hybrid']
        # The paired clean variant duplicates the hybrid baseline; record only once.
        variants += [v for v in pairs.get(tid, []) if v['condition'] != 'clean']
        for v in variants:
            payload = v['judge_input']
            query = payload['request']
            texts = [m['text'] for m in payload['memories']]
            safety = defender.should_include_safety_memory(query, None)
            objective = not safety and defender.should_include_objective_memory(query)
            assert safety or objective
            guidance = defender.safety_memory if safety else defender.objective_memory
            with torch.no_grad():
                full, _, _ = defender(query=query, candidates=texts, top_k=None, preserve_order=True)
                scores = defender.score_candidates(query, texts)
            gate = [text for text, score in zip(texts, scores) if score > .07]
            assert full == [guidance] + gate
            rows.append({'task_id':tid, 'condition':v['condition'], 'request':query, 'tool':payload['tool'],
                         'guidance_type':'safety' if safety else 'objective', 'scores':scores,
                         'gate_only':gate, 'full_wrapper':full})
    paths = [source, checkpoint, baseline, paired, audit, Path('research/evidence/model-revisions.json'), Path('research/evidence/heldout-split-v1.json'), Path(__file__)]
    save(INPUT, {'hashes':{str(p):sha(p) for p in paths}, 'scope':'13 affected release queries, hybrid clean plus selected paired injections; clean duplicate omitted', 'split_manifest':split, 'rows':rows})
    print('Prepared', len(rows), 'variants across', len(affected), 'parents.', flush=True)


def run():
    from src.agent.memory_prompts import generator_input, require_capacity
    libraries = Path(sys.prefix) / 'Lib/site-packages/nvidia'
    os.environ['PATH'] = os.pathsep.join([str(libraries / n / 'bin') for n in ('cublas','cuda_runtime')] + [os.environ['PATH']])
    import llama_cpp
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    assert llama_cpp.llama_supports_gpu_offload()
    prepared = json.loads(INPUT.read_bytes())
    for p, digest in prepared['hashes'].items():
        assert sha(p) == digest, p
    budget_path = Path('research/retrieval-budgets.json')
    budget = json.loads(budget_path.read_bytes())
    hashes = {str(p):sha(p) for p in [INPUT, budget_path, Path(__file__), Path('src/agent/memory_prompts.py')]}
    data = {'hashes':hashes, 'rows':[], 'calls':{}}
    if OUTPUT.exists():
        data = json.loads(OUTPUT.read_bytes())
        assert data['hashes'] == hashes
    model = llama_cpp.Llama(model_path='models/Qwen_Qwen3-4B-Q4_K_M.gguf', n_ctx=budget['context_tokens'], n_gpu_layers=-1, verbose=False)
    fmt = Jinja2ChatFormatter(model.metadata['tokenizer.chat_template'], '<|im_end|>', '')
    done = {(r['task_id'],r['condition'],r['arm']) for r in data['rows']}
    try:
        for row in prepared['rows']:
            primary_path = FIRST_PASS/'approved-evaluation'/(row['task_id'] + '.json')
            primary = json.loads(primary_path.read_bytes())
            for arm in ('gate_only', 'full_wrapper'):
                identity = (row['task_id'],row['condition'],arm)
                if identity in done:
                    continue
                memories = [{'id':f'm{i}', 'text':t} for i,t in enumerate(row[arm])]
                request = generator_input(row['request'], row['tool'], memories)
                request['messages'][-1]['content'] += '\n/no_think'
                request.update(temperature=0, seed=42, max_tokens=budget['generator_output_tokens'])
                key = hashlib.sha256(json.dumps(request, ensure_ascii=False).encode()).hexdigest()
                prompt = fmt(**{k:v for k,v in request.items() if k not in ('temperature','seed','max_tokens')}).prompt
                tokens = len(model.tokenize(prompt.encode(), add_bos=False, special=True))
                require_capacity(tokens, budget['generator_output_tokens'], budget['context_tokens'])
                reused = key in primary['calls']
                if key not in data['calls']:
                    if reused:
                        call = primary['calls'][key]
                        assert call['input'] == request
                    else:
                        start = perf_counter()
                        response = model.create_chat_completion(**request)
                        call = {'input':request, 'response':response, 'seconds':perf_counter()-start, 'prompt_tokens':tokens}
                    assert call['response']['usage']['prompt_tokens'] == tokens
                    data['calls'][key] = call
                data['rows'].append({'task_id':row['task_id'], 'condition':row['condition'], 'arm':arm, 'generation':key,
                    'reused_primary':reused, 'primary_trace_sha256':sha(primary_path), 'guidance_type':row['guidance_type']})
                save(OUTPUT, data)
                print(*identity, 'saved', flush=True)
    finally:
        model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','run'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else run()
