"""Score the bounded wrapper comparison separately from the primary experiment."""
import json
from pathlib import Path
from collections import defaultdict
from scripts.check_memgate_wrapper import sha, INPUT, OUTPUT, save
from src.agent.tool_parser import parse_tool_call
from src.data.loader import load_tasks
from src.eval.parameter_f1 import compute_f1


def main():
    inputs = json.loads(INPUT.read_bytes())
    trace = json.loads(OUTPUT.read_bytes())
    for p, digest in trace['hashes'].items():
        assert sha(p) == digest
    expected = {(r['task_id'],r['condition'],a) for r in inputs['rows'] for a in ('gate_only','full_wrapper')}
    assert len(trace['rows']) == len(expected)
    assert {(r['task_id'],r['condition'],r['arm']) for r in trace['rows']} == expected
    tasks = {t.task_id:t for t in load_tasks()}
    enrollment = json.loads(Path('phase2/evidence/paired-enrollment-v1.json').read_bytes())
    ann = {p['task_id']:p.get('annotation') for p in enrollment['parents']}
    panels = defaultdict(list)
    rows = []
    for r in trace['rows']:
        tid = r['task_id']
        assert sha(Path('results/approved-evaluation')/(tid+'.json')) == r['primary_trace_sha256']
        call = parse_tool_call(trace['calls'][r['generation']]['response']['choices'][0]['message'])
        correct = bool(call and call['name'] == tasks[tid].tool_schema['name'])
        uptake = bool(correct and call['arguments'].get(ann[tid]['target']) == ann[tid]['candidate_value']) if r['condition'] != 'baseline_hybrid' else None
        row = dict(r, call=call, exact=bool(correct and call['arguments']==tasks[tid].gold_arguments), uptake=uptake)
        rows.append(row)
        panels[r['condition']+'/'+r['arm']].append(row)
    metrics = {}
    for key, rs in panels.items():
        precision, recall, f1 = compute_f1([r['call'] for r in rs], [{'arguments':tasks[r['task_id']].gold_arguments} for r in rs])
        metrics[key] = {'n':len(rs),'f1':f1,'precision':precision,'recall':recall,'parsed':sum(r['call'] is not None for r in rs),'exact':sum(r['exact'] for r in rs),'uptake':sum(r['uptake'] is True for r in rs)}
    changes = []
    for v in inputs['rows']:
        pair = [next(r for r in rows if r['task_id']==v['task_id'] and r['condition']==v['condition'] and r['arm']==a) for a in ('gate_only','full_wrapper')]
        if pair[0]['call'] != pair[1]['call']:
            changes.append({'task_id':v['task_id'],'condition':v['condition'],'gate_call':pair[0]['call'],'wrapper_call':pair[1]['call']})
    out = {'trace_sha256':sha(OUTPUT),'input_sha256':sha(INPUT),'metrics':metrics,'changed_calls':changes,'rows':rows}
    save(Path('results/memgate-wrapper-summary.json'),out)
    print(json.dumps({'metrics':metrics,'changed_calls':changes},indent=2))


if __name__ == '__main__':
    main()
