"""Replay one saved development prompt and retain native GPU offload evidence."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--context', type=int, choices=(2048, 4096), default=4096)
    args = parser.parse_args()
    log = Path(f'results/gpu-runtime-{args.context}.log')
    if not args.worker:
        libraries = Path(sys.prefix) / 'Lib/site-packages/nvidia'
        directory_names = ('cublas/bin', 'cuda_runtime/bin')
        paths = [str(libraries / name) for name in directory_names if (libraries / name).is_dir()]
        environment = os.environ.copy()
        environment['PATH'] = os.pathsep.join(paths + [environment.get('PATH', '')])
        with log.open('wb') as stream:
            run = subprocess.run([sys.executable, '-m', 'scripts.check_gpu_runtime', '--worker',
                                  '--context', str(args.context)], stderr=stream, env=environment,
                                 stdout=subprocess.PIPE)
        run.check_returncode()
        result = json.loads(run.stdout)
        result['gpu_and_driver'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
            '--format=csv,noheader'], text=True).strip()
        with Path('models/Qwen_Qwen3-4B-Q4_K_M.gguf').open('rb') as model_file:
            result['gguf_sha256'] = hashlib.file_digest(model_file, 'sha256').hexdigest()
        lines = log.read_text(encoding='utf8', errors='replace').splitlines()
        result['offload_evidence'] = [line for line in lines if re.search(r'offloaded [1-9]\d*/\d+ layers to GPU', line)]
        if not result['offload_evidence']:
            raise RuntimeError(f'Positive GPU layer count unavailable. Inspect {log}')
        Path(f'results/gpu-runtime-{args.context}.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
        print(result['offload_evidence'], result['seconds'], result['usage'], result['finish_reason'])
        return
    import llama_cpp
    if not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError('Installed backend reports GPU offload unavailable')
    source = Path('results/gap3-cap-remaining.json')
    reference = next(row for row in json.loads(source.read_bytes())['rows'] if row['task_id']=='qa_311' and row['source_arm']==['clean','unfiltered'])
    state = reference['input']
    schema = state['tool_schema']
    tools = [{'type':'function','function':{key:schema[key] for key in ('name','description','parameters')}}]
    with closing(llama_cpp.Llama(model_path='models/Qwen_Qwen3-4B-Q4_K_M.gguf', n_ctx=args.context, n_gpu_layers=-1, verbose=True)) as model:
        start = perf_counter()
        response = model.create_chat_completion(messages=state['messages']+[{'role':'user','content':state['query']}],
            tools=tools, tool_choice='auto', seed=42, temperature=0, max_tokens=1024)
        elapsed = perf_counter()-start
    print(json.dumps({'task':'qa_311', 'arm':'clean', 'context':args.context, 'version':llama_cpp.__version__,
        'module':llama_cpp.__file__, 'reference_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'seconds':elapsed, 'usage':response['usage'], 'finish_reason':response['choices'][0]['finish_reason'],
        'message':response['choices'][0]['message'], 'historical_cpu_seconds':reference['seconds'],
        'historical_cpu_usage':reference['usage']}))


if __name__ == '__main__':
    main()
