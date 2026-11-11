"""Score the supplemental fixed-admission ablation against the saved full judge."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scripts.check_memgate_wrapper import save, sha
from src.agent.tool_parser import parse_tool_call
from src.data.loader import load_tasks

FIRST_PASS = Path('results/first-pass')

def main():
    path = Path('results/scoping-ablation.json')
    trace = json.loads(path.read_bytes())
    for p,h in trace['hashes'].items():
        assert sha(p) == h
    primary = json.loads((FIRST_PASS/'approved-pass-summary.json').read_bytes())
    references = {(r['task'],r['condition']):r for r in primary['rows'] if r['arm']=='judge'}
    assert len(trace['rows']) == len(references) == 204
    assert {(r['task_id'],r['condition']) for r in trace['rows']} == set(references)
    tasks = {t.task_id:t for t in load_tasks()}
    ann = {p['task_id']:p.get('annotation') for p in json.loads(Path('research/evidence/paired-enrollment-v1.json').read_bytes())['parents']}
    rows = []
    for r in trace['rows']:
        tid = r['task_id']; task = tasks[tid]; a = ann[tid]
        assert sha(FIRST_PASS/'approved-evaluation'/(tid+'.json')) == r['primary_trace_sha256']
        call = parse_tool_call(trace['calls'][r['generation']]['response']['choices'][0]['message'])
        args = call['arguments'] if call else {}
        correct = bool(call and call['name']==task.tool_schema['name'])
        tp = sum(k in args and str(args[k]).lower()==str(v).lower() for k,v in task.gold_arguments.items())
        rows.append(dict(r,call=call,counts=[tp,len(args)-tp,len(task.gold_arguments)-tp],
            uptake=bool(correct and args.get(a['target'])==a['candidate_value']),
            joint=bool(correct and args.get(a['target'])==a['target_gold'] and args.get(a['allowed'])==a['candidate_value']) if a['partial'] else None,
            exact=bool(correct and args==task.gold_arguments)))
    def vector(r):return [*r['counts'],int(r['uptake']),1]
    def estimate(arr):
        tp,fp,fn,u,n = arr.T
        den=2*tp+fp+fn
        return np.stack([np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0),u/n],axis=-1)
    panels={}
    for condition in ('clean','unrelated','partial'):
        rs=[r for r in rows if r['condition']==condition]
        groups=defaultdict(list)
        for r in rs:groups[references[r['task_id'],condition]['group']].append(r)
        left=np.array([np.sum([vector(references[r['task_id'],condition]) for r in g],axis=0) for g in groups.values()])
        right=np.array([np.sum([vector(r) for r in g],axis=0) for g in groups.values()])
        point=estimate(right.sum(0));delta=point-estimate(left.sum(0))
        panel={'n':len(rs),'f1':float(point[0]),'uptake':sum(r['uptake'] for r in rs),'parsed':sum(r['call'] is not None for r in rs),'exact':sum(r['exact'] for r in rs),'joint':sum(r['joint'] is True for r in rs),'delta_f1_vs_full_judge':float(delta[0]),'delta_uptake_vs_full_judge':float(delta[1])}
        if len(rs)>4:
            draws=np.random.default_rng(42).integers(len(groups),size=(10000,len(groups)))
            diffs=estimate(right[draws].sum(1))-estimate(left[draws].sum(1))
            panel['paired_percentile_95_f1_uptake']=np.quantile(diffs,[.025,.975],axis=0).tolist()
        panels[condition]=panel
    save(Path('results/scoping-ablation-summary.json'),{'trace_sha256':sha(path),'panels':panels,'rows':rows})
    print(json.dumps(panels,indent=2))


if __name__=='__main__':main()
