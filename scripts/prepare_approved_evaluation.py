"""Bind the approved paired sample and the two full-release baselines to cached inputs."""
import json
import hashlib
from pathlib import Path
from scripts.verify_heldout_split import build

E=Path('research/evidence')
def read(p): return json.loads(Path(p).read_bytes())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    proposal=E/'proposed-compute-subset-100.json'
    selection=read(proposal)
    enrollment=E/'paired-enrollment-v1.json'
    if digest(enrollment)!=selection['enrollment_sha256']: raise ValueError('Enrollment changed')
    eligible={r['task_id']:r for r in read(enrollment)['parents'] if r['eligible'] and r['split']=='heldout'}
    ids=set(selection['task_ids']);partial={i for i,r in eligible.items() if r['annotation']['partial']}
    assert len(ids)==len(selection['task_ids'])==100 and ids<=eligible.keys() and partial<=ids and len(partial)==4
    assert ids<=set(build()['heldout'])
    frozen=dict(selection,status='approved',proposal_sha256=digest(proposal))
    path=E/'paired-compute-100-v1.json'
    if not path.exists():
        with path.open('x',encoding='utf8') as f:json.dump(frozen,f,indent=2);f.write('\n')
    assert read(path)==frozen
    ranks=read('results/retrieval-400.json')
    qa=Path('data/Mem2ActBench/Mem2ACTBench_small/qa_dataset.jsonl')
    cf=Path('data/retrieval-dev/chunks-128.json')
    assert digest(qa)==ranks['qa_sha256'] and digest(cf)==ranks['chunk_sha256']
    chunks={c['id']:c['text'] for c in read(cf)}
    rows=[];base=[]
    for task in map(json.loads,qa.read_text(encoding='utf8').splitlines()):
        tid=task['qa_id']; variants=[]
        for method in ('bm25','hybrid'):
            rr=ranks['rows'][tid][method]
            keys=[r[0] if method=='bm25' else r['id'] for r in rr]
            mem=[{'id':f'm{i}','text':chunks[k]} for i,k in enumerate(keys)]
            def variant(condition,memories):
                return {'condition':condition,'memories':memories,'judge_input':{'request':task['query'],'tool':task['target_tool_schema'],'memories':memories}}
            variants.append(variant('baseline_'+method,mem))
        base.append({'chunk_limit':128,'task_id':tid,'variants':variants})
        if tid in ids:
            a=eligible[tid]['annotation']; pair=[variant('clean',mem)]
            for cond in ('unrelated','partial'):
                if a[cond]:pair.append(variant(cond,mem+[{'id':'m5','text':a[cond]}]))
            rows.append({'chunk_limit':128,'task_id':tid,'variants':pair})
    assert len(base)==400 and sum(len(r['variants']) for r in rows)==204
    for label,records in [('paired',rows),('baseline',base)]:
        dest=Path(f'data/retrieval-dev/eval-{label}.json')
        payload={'retriever':'heldout100','manifest_sha256':digest(path),'rows':records}
        if dest.exists() and read(dest)!=payload:raise ValueError(f'Existing input mismatch: {dest}')
        dest.write_text(json.dumps(payload,ensure_ascii=False)+'\n',encoding='utf8')
    print('Verified 100 paired parents, 204 variants, and 800 baseline inputs.')
if __name__=='__main__':main()
