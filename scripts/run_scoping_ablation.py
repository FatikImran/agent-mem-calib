"""Remove field-scoping prefixes while holding saved judge admissions fixed."""
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from scripts.check_memgate_wrapper import save, sha
from src.agent.memory_prompts import generator_input, require_capacity

FIRST_PASS = Path('results/first-pass')

def main():
    libraries = Path(sys.prefix)/'Lib/site-packages/nvidia'
    os.environ['PATH'] = os.pathsep.join([str(libraries/n/'bin') for n in ('cublas','cuda_runtime')]+[os.environ['PATH']])
    import llama_cpp
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    assert llama_cpp.llama_supports_gpu_offload()
    source = FIRST_PASS/'approved-pass-summary.json'
    summary = json.loads(source.read_bytes())
    budget = json.loads(Path('research/retrieval-budgets.json').read_bytes())
    hashes = {str(p):sha(p) for p in [source,Path('research/retrieval-budgets.json'),Path(__file__),Path('src/agent/memory_prompts.py')]}
    out = Path('results/scoping-ablation.json')
    data = {'hashes':hashes,'rows':[],'calls':{}}
    if out.exists():
        data = json.loads(out.read_bytes())
        assert data['hashes'] == hashes
    done = {(r['task_id'],r['condition']) for r in data['rows']}
    model = llama_cpp.Llama(model_path='models/Qwen_Qwen3-4B-Q4_K_M.gguf',n_ctx=4096,n_gpu_layers=-1,verbose=False)
    fmt = Jinja2ChatFormatter(model.metadata['tokenizer.chat_template'],'<|im_end|>','')
    try:
        for tid,digest in summary['trace_sha256'].items():
            path = FIRST_PASS/'approved-evaluation'/(tid+'.json')
            assert sha(path) == digest
            trace = json.loads(path.read_bytes())
            for row in trace['rows']:
                if row['arm'] != 'judge' or (tid,row['condition']) in done:
                    continue
                payload = json.loads(trace['calls'][row['stage']]['input']['messages'][-1]['content'])
                kept = [m for m in payload['memories'] if row['admissions'].get(m['id'])]
                request = generator_input(payload['request'],payload['tool'],kept)
                request['messages'][-1]['content'] += '\n/no_think'
                config = dict(request,temperature=0,seed=42,max_tokens=budget['generator_output_tokens'])
                key = hashlib.sha256(json.dumps(config,ensure_ascii=False).encode()).hexdigest()
                tokens = len(model.tokenize(fmt(**request).prompt.encode(),add_bos=False,special=True))
                require_capacity(tokens,budget['generator_output_tokens'],budget['context_tokens'])
                reused = key in trace['calls']
                if key not in data['calls']:
                    if reused:
                        call = trace['calls'][key]
                        assert call['input'] == config
                    else:
                        start = perf_counter()
                        response = model.create_chat_completion(**config)
                        call = {'input':config,'response':response,'seconds':perf_counter()-start,'prompt_tokens':tokens}
                    assert call['response']['usage']['prompt_tokens'] == tokens
                    data['calls'][key] = call
                data['rows'].append({'task_id':tid,'condition':row['condition'],'generation':key,'primary_trace_sha256':digest,
                    'kept_ids':[m['id'] for m in kept],'admissions':row['admissions'],'reused_primary':reused})
                save(out,data)
                print(len(data['rows']),tid,row['condition'],flush=True)
        assert len(data['rows']) == 204
    finally:
        model.close()


if __name__ == '__main__':
    main()
