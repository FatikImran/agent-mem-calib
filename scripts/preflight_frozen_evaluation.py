"""Preflight the frozen evaluation inputs without generating predictions."""
import hashlib
import json
from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
from scripts.verify_heldout_split import build
from src.agent.memory_prompts import generator_input, judge_input, FILTER_PROMPT


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    rankings=read('results/retrieval-400.json')
    chunk_path=Path('data/retrieval-dev/chunks-128.json')
    assert hashlib.sha256(chunk_path.read_bytes()).hexdigest()==rankings['chunk_sha256']
    chunks={c['id']:c['text'] for c in read(chunk_path)}
    parents=read('research/evidence/paired-enrollment-v1.json')['parents']
    budget=read('research/retrieval-budgets.json')
    qa=Path('data/Mem2ActBench/Mem2ACTBench_small/qa_dataset.jsonl')
    assert hashlib.sha256(qa.read_bytes()).hexdigest()==rankings['qa_sha256']
    tasks={r['qa_id']:r for r in map(json.loads,qa.read_text(encoding='utf8').splitlines())}
    split=build()
    assert len(tasks)==400 and set(tasks)==set(rankings['rows'])=={r['task_id'] for r in parents}
    model=Llama(model_path='models/Qwen_Qwen3-4B-Q4_K_M.gguf',vocab_only=True,verbose=False)
    formatter=Jinja2ChatFormatter(model.metadata['tokenizer.chat_template'],'<|im_end|>','')
    def count(text): return len(model.tokenize(text.encode(),add_bos=False,special=True))
    def size(inputs): return count(formatter(**inputs).prompt)
    def gen(query,schema,mem,admit=None):
        req=generator_input(query,schema,mem,admit)
        req['messages'][-1]['content']+='\n/no_think'
        return size(req)
    rows=[]; baseline=[]
    for p in parents:
        tid=p['task_id'];task=tasks[tid]; query=task['query'];schema=task['target_tool_schema']
        assert tid in split[p['split']]
        for method in ['bm25','hybrid']:
            ranks=rankings['rows'][tid][method]
            ids=[r['id'] if method=='hybrid' else r[0] for r in ranks]
            assert len(set(ids))==len(ids)==budget['retrieved_chunks']
            mem=[{'id':f'm{i}','text':chunks[key]} for i,key in enumerate(ids)]
            baseline.append({'task_id':tid,'method':method,'prompt_tokens':gen(query,schema,mem)})
        if not p['eligible']: continue
        a=p['annotation']
        for condition in ['clean','unrelated','partial']:
            if condition!='clean' and not a[condition]:continue
            memories=list(mem)
            reserved=0
            if condition!='clean':
                memories.append({'id':'m5','text':a[condition]})
                reserved=count('[memory]: '+a[condition]+'\n')
            context=''.join('[memory]: '+m['text']+'\n' for m in memories)
            sizes={'generator':gen(query,schema,memories),'scoped_generator':gen(query,schema,memories,{m['id']:list(schema['parameters'].get('properties',{})) for m in memories}),
                   'judge':size(judge_input(query,schema,memories)),
                   'self_recheck':size({'messages':[{'role':'system','content':FILTER_PROMPT},{'role':'user','content':f'Question: {query}\n\nContext:\n{context}'}]})}
            overflow=[s for s,n in sizes.items() if n+budget[('generator' if s=='scoped_generator' else s)+'_output_tokens']+(budget['injection_reserve_tokens'] if condition=='clean' else 0)>budget['context_tokens']]
            if reserved>budget['injection_reserve_tokens']:overflow.append('injection')
            rows.append({'task_id':tid,'split':p['split'],'condition':condition,'input_tokens':sizes,'injection_tokens':reserved,'overflow':overflow})
    output={'budget':budget,'baseline':baseline,'paired':rows,'baseline_overflow':sum(r['prompt_tokens']+budget['generator_output_tokens']>budget['context_tokens'] for r in baseline),'paired_overflow':sum(bool(r['overflow']) for r in rows)}
    Path('results/frozen-capacity-preflight.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    print(len(baseline),len(rows),output['baseline_overflow'],output['paired_overflow'])
if __name__=='__main__':main()
