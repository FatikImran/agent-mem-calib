"""Score only a complete, identically configured approved evaluation pass."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

from src.agent.tool_parser import parse_tool_call
from src.data.loader import load_tasks
from src.eval.parameter_f1 import compute_f1

def main():
    tasks={t.task_id:t for t in load_tasks()}
    frozen=json.loads(Path('phase2/evidence/paired-compute-100-v1.json').read_bytes())
    subset=set(frozen['task_ids'])
    enrollment=json.loads(Path('phase2/evidence/paired-enrollment-v1.json').read_bytes())
    annotations={p['task_id']:p.get('annotation') for p in enrollment['parents']}
    parent={t:t for t in tasks}
    def group(t):
        while parent[t]!=t:
            parent[t]=parent[parent[t]]
            t=parent[t]
        return t
    seen={}
    for tid,t in tasks.items():
        for key in [('q',' '.join(t.query.casefold().split()))]+[('s',s) for s in t.raw['source_conversation_ids'] if s]:
            if key in seen: parent[group(tid)]=group(seen[key])
            else: seen[key]=tid
    arms=('unfiltered','memgate','self_recheck','judge')
    rows=[]; provenance=None; digests={}
    for tid,t in tasks.items():
        path=Path('results/approved-evaluation')/(tid+'.json')
        trace=json.loads(path.read_bytes())
        digests[tid]=hashlib.sha256(path.read_bytes()).hexdigest()
        if provenance is None: provenance=trace['hashes']
        assert trace['hashes']==provenance
        expected={('baseline_bm25','unfiltered'),('baseline_hybrid','unfiltered')}
        a=annotations[tid]
        if tid in subset:
            conditions=['clean','unrelated']+(['partial'] if a['partial'] else [])
            expected.update((c,arm) for c in conditions for arm in arms)
        observed={(r['condition'],r['arm']) for r in trace['rows']}
        assert expected==observed and len(trace['rows'])==len(expected)
        for r in trace['rows']:
            assert r['task_id']==tid
            gen=trace['calls'][r['generation']]
            ch=gen['response']['choices'][0]
            call=parse_tool_call(ch['message'])
            args=call['arguments'] if call else {}
            correct=bool(call and call['name']==t.tool_schema['name'])
            tp=sum(k in args and str(args[k]).lower()==str(v).lower() for k,v in t.gold_arguments.items())
            paired=r['condition'] in ('clean','unrelated','partial')
            rows.append(dict(task=tid,group=group(tid),arm=r['arm'],condition=r['condition'],call=call,
                counts=[tp,len(args)-tp,len(t.gold_arguments)-tp], parsed=call is not None,correct_tool=correct,
                exact=correct and args==t.gold_arguments,
                uptake=(correct and args.get(a['target'])==a['candidate_value']) if paired else None,
                joint=(correct and args.get(a['target'])==a['target_gold'] and args.get(a['allowed'])==a['candidate_value']) if paired and a['partial'] else None,
                finish=ch['finish_reason'],fallback=r['stage_status'],seconds=gen['seconds'],tokens=gen['response']['usage']))
    panels=defaultdict(list)
    for r in rows:panels[r['arm']+'/'+r['condition']].append(r)
    metrics={}
    for name,items in panels.items():
        prec,rec,f1=compute_f1([r['call'] for r in items],[{'arguments':tasks[r['task']].gold_arguments} for r in items])
        metrics[name]=dict(n=len(items),precision=prec,recall=rec,f1=f1,parsed=sum(r['parsed'] for r in items),
            correct_tool=sum(r['correct_tool'] for r in items),exact=sum(r['exact'] for r in items),
            uptake=sum(r['uptake'] is True for r in items),joint=sum(r['joint'] is True for r in items),
            length_stops=sum(r['finish']=='length' for r in items),fallbacks=sum(r['fallback']!='ok' for r in items))
    def contrast(left,right):
        lhs={r['task']:r for r in panels[left]}
        rhs={r['task']:r for r in panels[right]}
        ids=sorted(lhs.keys() & rhs.keys())
        groups=defaultdict(list)
        for tid in ids: groups[group(tid)].append(tid)
        def values(mapping, batch):
            return np.array([*np.sum([mapping[t]['counts'] for t in batch],axis=0),sum(mapping[t]['uptake'] for t in batch),len(batch)])
        def estimates(arr):
            tp,fp,fn,u,n=arr.T
            denom=2*tp+fp+fn
            return np.stack([np.divide(2*tp,denom,out=np.zeros_like(tp,dtype=float),where=denom!=0),u/n],axis=-1)
        a=np.array([values(lhs,g) for g in groups.values()]);b=np.array([values(rhs,g) for g in groups.values()])
        point=estimates(b.sum(0))-estimates(a.sum(0))
        result=dict(reference=left,comparison=right,tasks=len(ids),groups=len(groups),delta_f1=point[0],delta_uptake=point[1],
            new_uptake=sum(bool(rhs[t]['uptake'] and not lhs[t]['uptake']) for t in ids),lost_uptake=sum(bool(lhs[t]['uptake'] and not rhs[t]['uptake']) for t in ids))
        if len(ids)>4:
            rng=np.random.default_rng(42); draws=rng.integers(len(groups),size=(10000,len(groups)))
            diff=estimates(b[draws].sum(1))-estimates(a[draws].sum(1))
            result['percentile_95_f1_uptake']=np.quantile(diff,[.025,.975],axis=0).tolist()
        return result
    contrasts=[contrast('unfiltered/clean','unfiltered/unrelated')]
    contrasts += [contrast('unfiltered/unrelated',arm+'/unrelated') for arm in arms[1:]]
    contrasts += [contrast(arm+'/unrelated',arm+'/partial') for arm in arms]
    output={'trace_sha256':digests,'metrics':metrics,'contrasts':contrasts,'rows':rows}
    dest=Path('results/approved-pass-summary.json');tmp=dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(output,indent=2)+'\n',encoding='utf8');tmp.replace(dest)
    print('Scored',len(rows),'arm outcomes.')
if __name__=='__main__':main()
