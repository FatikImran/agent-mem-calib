"""Run one fresh, approved evaluation repeat (pass 2 or 3)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter

from src.agent.memory_prompts import FILTER_PROMPT, generator_input, judge_input, judge_response_schema, require_capacity

ORIGINAL = Path('results/comparative-development.json')
REPLAY = Path('results/development-no-think.json')
HYBRID = Path('results/hybrid-development.json')
GATE = Path('results/memgate-hybrid-development.json')
BUDGET = Path('research/retrieval-budgets.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-replay', action='store_true', help='Require every request to match a saved development call')
    parser.add_argument('--repeat', type=int, choices=(2,3), required=True)
    parser.add_argument('--approved-evaluation', action='store_true', help='Run frozen 100-parent panel and both 400-query baselines')
    options = parser.parse_args()
    if not options.approved_evaluation or options.verify_replay:
        parser.error('Fresh repeats require --approved-evaluation and prohibit replay')
    if options.approved_evaluation and options.verify_replay:
        parser.error('Replay verification is development-only')
    out = Path('results/comparative-development-primary.json')
    libraries = Path(sys.prefix) / 'Lib/site-packages/nvidia'
    paths = [str(libraries / name / 'bin') for name in ('cublas', 'cuda_runtime')]
    os.environ['PATH'] = os.pathsep.join(paths + [os.environ['PATH']])
    import llama_cpp
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    if not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError('Use the verified GPU environment')
    budget = json.loads(BUDGET.read_bytes())
    if options.approved_evaluation:
        from scripts.prepare_approved_evaluation import main as verify_inputs
        verify_inputs()
        paired = Path('data/retrieval-dev/eval-paired.json')
        baseline = Path('data/retrieval-dev/eval-baseline.json')
        gate_path = Path(f'results/repeat-{options.repeat}/memgate-heldout100-development.json')
        gate = json.loads(gate_path.read_bytes())
        if gate['source_sha256'] != digest(paired):
            raise ValueError('Gate cache differs from the frozen inputs')
        pairs = {r['task_id']:r['variants'] for r in json.loads(paired.read_bytes())['rows']}
        records = json.loads(baseline.read_bytes())['rows']
        for record in records:
            record['variants'].extend(pairs.get(record['task_id'], []))
        evidence = [paired, baseline, gate_path, Path('research/evidence/paired-compute-100-v1.json')]
        trace_dir = Path(f'results/repeat-{options.repeat}/approved-evaluation')
        trace_dir.mkdir(parents=True, exist_ok=True)
    else:
        source = json.loads(HYBRID.read_bytes())
        gate = json.loads(GATE.read_bytes())
        if gate['source_sha256'] != digest(HYBRID):
            raise ValueError('Cached gate and retrieval artifacts differ')
        records = [r for r in source['rows'] if r['chunk_limit'] == budget['chunk_tokens']]
        if {r['task_id'] for r in records} != {'qa_063','qa_220','qa_311','qa_327'} or len(records) != 4:
            raise ValueError('Unexpected development manifest')
        evidence = [HYBRID,GATE]
    gates = {(r['task_id'],r['condition']):r for r in gate['rows'] if r['chunk_limit']==budget['chunk_tokens']}
    hashes = {str(p):digest(p) for p in [*evidence,BUDGET,Path(__file__),Path('src/agent/memory_prompts.py')]}
    data = {'hashes':hashes, 'budget':budget, 'seed':42, 'temperature':0, 'calls':{}, 'rows':[]}
    if not options.approved_evaluation and out.exists() and not options.verify_replay:
        data = json.loads(out.read_bytes())
        if data['hashes'] != hashes:
            raise ValueError('Existing trace has different inputs or code')
    saved = {}
    if options.verify_replay:
        baseline = json.loads(ORIGINAL.read_bytes())
        replay = json.loads(REPLAY.read_bytes())
        for name, expected in replay['hashes'].items():
            if digest(Path(name)) != expected:
                raise ValueError(f'Replay provenance mismatch: {name}')
        for path in (HYBRID, GATE, BUDGET):
            if baseline['hashes'][str(path)] != digest(path):
                raise ValueError(f'Historical evidence differs: {path}')
        for record in [*baseline['calls'].values(), *replay['calls'].values()]:
            encoded = json.dumps(record['input'],ensure_ascii=False).encode()
            saved[hashlib.sha256(encoded).hexdigest()] = record
        data['verified_sources'] = {str(p):digest(p) for p in (ORIGINAL, REPLAY)}
    model = llama_cpp.Llama(model_path='models/Qwen_Qwen3-4B-Q4_K_M.gguf', n_ctx=budget['context_tokens'], n_gpu_layers=-1, vocab_only=options.verify_replay, verbose=False)
    fmt = Jinja2ChatFormatter(model.metadata['tokenizer.chat_template'],'<|im_end|>','')
    def count(text):
        return len(model.tokenize(text.encode(), add_bos=False, special=True))
    def invoke(inputs, cap, response_format=None):
        config = dict(inputs, temperature=0, seed=42, max_tokens=cap)
        if response_format:
            config['response_format'] = response_format
        key = hashlib.sha256(json.dumps(config,ensure_ascii=False).encode()).hexdigest()
        if key not in data['calls']:
            tokens = count(fmt(**inputs).prompt)
            require_capacity(tokens, cap, budget['context_tokens'])
            if options.verify_replay:
                record = saved[key]
                if record['input'] != config:
                    raise ValueError('Saved input differs from constructed request')
                response, seconds = record['response'], record['seconds']
            else:
                start = perf_counter()
                response = model.create_chat_completion(**config)
                seconds = perf_counter()-start
            data['calls'][key] = {'input':config,'seconds':seconds,'prompt_tokens':tokens,'response':response}
            if response['usage']['prompt_tokens'] != tokens:
                raise ValueError('Runtime token accounting mismatch')
        return key, data['calls'][key]['response']['choices'][0]
    done = {(r['task_id'],r['condition'],r['arm']) for r in data['rows']}
    for task in records:
        if options.approved_evaluation:
            out = trace_dir / (task['task_id'] + '.json')
            data = {'hashes':hashes, 'budget':budget, 'seed':42, 'temperature':0, 'calls':{}, 'rows':[]}
            if out.exists():
                data = json.loads(out.read_bytes())
                if data['hashes'] != hashes:
                    raise ValueError(f'Checkpoint provenance mismatch: {out}')
            done = {(r['task_id'],r['condition'],r['arm']) for r in data['rows']}
        for v in task['variants']:
            tid,condition = task['task_id'],v['condition']
            payload = v['judge_input']
            query,schema,memories = payload['request'],payload['tool'],payload['memories']
            if condition in ('unrelated','partial') and count('[memory]: '+memories[-1]['text']+'\n') > budget['injection_reserve_tokens']:
                raise ValueError('Injection exceeds reserved space')
            arms = ('unfiltered',) if condition.startswith('baseline_') else ('unfiltered','memgate','self_recheck','judge')
            for arm in arms:
                if (tid,condition,arm) in done:
                    continue
                chosen,admitted,stage,status = memories,None,None,'ok'
                if arm == 'memgate':
                    kept = gates[tid,condition]['kept_ids']
                    chosen = [m for m in memories if m['id'] in kept]
                if arm == 'self_recheck':
                    context = ''.join('[memory]: '+m['text']+'\n' for m in memories)
                    inputs = {'messages':[{'role':'system','content':FILTER_PROMPT},{'role':'user','content':f'Question: {query}\n\nContext:\n{context}'}]}
                    stage,choice = invoke(inputs,budget['self_recheck_output_tokens'])
                    content = (choice['message'].get('content') or '').split('</think>')[-1].strip()
                    if choice['finish_reason']=='length' or '<think>' in content or not content:
                        status,content='incomplete_filter',''
                    chosen = [] if content in ('','NO RELEVANT CONTEXT') else [{'id':'filtered','text':content}]
                if arm == 'judge':
                    stage,choice = invoke(judge_input(query,schema,memories),budget['judge_output_tokens'],
                        {'type':'json_object','schema':judge_response_schema(schema,memories)})
                    admitted = {}
                    fields = schema['parameters']['properties']
                    try:
                        verdicts = json.loads(choice['message']['content'])
                        if choice['finish_reason']=='length' or set(verdicts)!={m['id'] for m in memories}:
                            raise ValueError()
                        for mid, verdict in verdicts.items():
                            if set(verdict)!=set(fields) or any(x not in ('applicable','inapplicable','uncertain') for x in verdict.values()):
                                raise ValueError()
                        admitted = {mid:[f for f,val in verdict.items() if val=='applicable'] for mid,verdict in verdicts.items()}
                    except (ValueError,TypeError,AttributeError):
                        status='invalid_judge'
                generation_request = generator_input(query,schema,chosen,admitted)
                generation_request['messages'][-1]['content'] += '\n/no_think'
                generation,_ = invoke(generation_request,budget['generator_output_tokens'])
                data['rows'].append({'task_id':tid,'condition':condition,'arm':arm,'stage':stage,'stage_status':status,'admissions':admitted,'generation':generation})
                tmp=out.with_suffix('.tmp')
                tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
                tmp.replace(out)
                print(len(data['rows']),tid,condition,arm,status,flush=True)
    model.close()

if __name__=='__main__':
    main()
