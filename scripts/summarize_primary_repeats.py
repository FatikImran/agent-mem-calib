"""Verify three complete passes and describe greedy execution repeatability."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import statistics


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    passes=[]
    for number,base in [(1,Path('results/first-pass')),(2,Path('results/repeat-2')),(3,Path('results/repeat-3'))]:
        p=base/'approved-pass-summary.json';s=json.loads(p.read_bytes())
        assert len(s['rows'])==1616 and len(s['trace_sha256'])==400
        calls={};timings=defaultdict(list)
        for tid,h in s['trace_sha256'].items():
            path=base/'approved-evaluation'/(tid+'.json')
            assert sha(path)==h
            trace=json.loads(path.read_bytes())
            for key,c in trace['calls'].items():
                assert hashlib.sha256(json.dumps(c['input'],ensure_ascii=False).encode()).hexdigest()==key
                calls[tid+'/'+key]=c
            for row in trace['rows']:
                keys=[row['generation']]+([row['stage']] if row['stage'] else [])
                timings[row['arm']+'/'+row['condition']].append(sum(trace['calls'][key]['seconds'] for key in keys))
        passes.append({'number':number,'summary_sha256':sha(p),'summary':s,'calls':calls,
            'unique_model_calls':len(calls),'saved_inference_seconds':sum(c['seconds'] for c in calls.values()),
            'panel_mean_llm_seconds':{k:statistics.mean(v) for k,v in timings.items()}})
    comparisons=[]
    ref=passes[0]
    rowkey=lambda r:(r['task'],r['condition'],r['arm'])
    reference={rowkey(r):r for r in ref['summary']['rows']}
    for item in passes[1:]:
        rows={rowkey(r):r for r in item['summary']['rows']}
        assert rows.keys()==reference.keys()
        same_keys=item['calls'].keys()==ref['calls'].keys()
        common=item['calls'].keys() & ref['calls'].keys()
        changed=[k for k in sorted(common) if item['calls'][k]['response']['choices']!=ref['calls'][k]['response']['choices']]
        changed_rows=[{'task':k[0],'condition':k[1],'arm':k[2]} for k in reference if rows[k]['call']!=reference[k]['call']]
        comparisons.append({'pass':item['number'],'same_model_request_keys':same_keys,'common_requests':len(common),
            'changed_response_choices':changed,'changed_parsed_outcomes':changed_rows})
    metrics={}
    for panel in ref['summary']['metrics']:
        metrics[panel]={}
        for metric in ('f1','parsed','correct_tool','exact','uptake','joint','fallbacks'):
            values=[p['summary']['metrics'][panel][metric] for p in passes]
            metrics[panel][metric]={'values':values,'mean':statistics.mean(values),'sample_sd':statistics.stdev(values)}
    output={'interpretation':'Three greedy execution passes at unchanged temperature zero and seed 42. Sample SD is between-run variation, not independent task evidence. Timings exclude retrieval and gate preprocessing and reflect different machine conditions. Within-pass exact-input reuse is retained; no inference reused across passes.',
        'passes':[{k:v for k,v in p.items() if k not in ('summary','calls')} for p in passes],
        'comparisons_to_pass_one':comparisons,'metrics':metrics}
    dest=Path('results/primary-repeatability-summary.json')
    dest.write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'passes':output['passes'],'comparisons':[{k:(len(v) if isinstance(v,list) else v) for k,v in c.items()} for c in comparisons]},indent=2))


if __name__=='__main__':main()
