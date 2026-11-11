"""Post-hoc diagnostic categories and complete example trajectories; no score changes."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from src.data.loader import load_tasks

FIRST_PASS = Path('results/first-pass')

def main():
    source = FIRST_PASS/'approved-pass-summary.json'
    scored = json.loads(source.read_bytes())
    tasks = {t.task_id: t for t in load_tasks()}
    enrollment = json.loads(Path('research/evidence/paired-enrollment-v1.json').read_bytes())
    annotations = {p['task_id']: p['annotation'] for p in enrollment['parents'] if p.get('annotation')}
    injection_verdicts = Counter()
    traces = {}
    for tid, digest in scored['trace_sha256'].items():
        p = FIRST_PASS/'approved-evaluation'/(tid + '.json')
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
        traces[tid] = json.loads(p.read_bytes())
    categories = defaultdict(Counter)
    records = []
    examples = []
    selected = {'qa_034', 'qa_040', 'qa_045', 'qa_094'}
    for row in scored['rows']:
        if row['condition'] not in ('clean', 'unrelated', 'partial'):
            continue
        trace = traces[row['task']]
        raw = next(r for r in trace['rows'] if r['condition'] == row['condition'] and r['arm'] == row['arm'])
        generation = trace['calls'][raw['generation']]
        if row['arm'] == 'judge' and row['condition'] == 'unrelated':
            stage = trace['calls'][raw['stage']]
            payload = json.loads(stage['input']['messages'][-1]['content'])
            mid = payload['memories'][-1]['id']
            verdict = json.loads(stage['response']['choices'][0]['message']['content'])[mid][annotations[row['task']]['target']]
            injection_verdicts[verdict] += 1
            injection_verdicts[verdict + '_uptake'] += bool(row['uptake'])
        category = ('no_parsed_call' if not row['parsed'] else
                    'wrong_tool' if not row['correct_tool'] else
                    'target_candidate_uptake' if row['uptake'] else
                    'exact_success' if row['exact'] else 'other_argument_mismatch')
        record = {'task': row['task'], 'arm': row['arm'], 'condition': row['condition'],
                  'category': category, 'stage_status': raw['stage_status']}
        # All 106 non-call responses in these two injected panels were read in full.
        # These are assistant qualitative labels, not independent human annotations.
        if category == 'no_parsed_call' and row['condition'] == 'unrelated' and row['arm'] in ('self_recheck', 'judge'):
            record['reviewed_response_type'] = ('prose_action_intent_without_call' if row['task'] == 'qa_094' and row['arm'] == 'judge'
                                                else 'requests_information_or_clarification')
            record['response'] = generation['response']['choices'][0]['message']
        records.append(record)
        categories[row['arm'] + '/' + row['condition']][category] += 1
        if row['task'] in selected and row['arm'] == 'judge' and row['condition'] == 'unrelated':
            examples.append({'task': row['task'], 'category': category,
                'gold_arguments': tasks[row['task']].gold_arguments,
                'raw_trace': (FIRST_PASS/'approved-evaluation'/(row['task'] + '.json')).as_posix(),
                'stage': trace['calls'][raw['stage']], 'admissions': raw['admissions'],
                'generation': generation, 'scored_row': row})
    result = {'summary_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'method': 'Post-hoc mutually exclusive categories in priority order: no parsed call, wrong tool, candidate uptake, exact success, other argument mismatch. Qualitative review covers every non-call in injected Self-ReCheck and judge panels only; labels describe response behavior, not whether clarification was warranted. Examples are first two exact judge successes by task ID, first judge uptake failure, and the sole reviewed prose-action-intent case. Not a random sample.',
        'counts': {k: dict(v) for k, v in categories.items()},
        'injected_target_judge_verdicts': dict(injection_verdicts),
        'records': records, 'examples': examples}
    (FIRST_PASS/'first-pass-failure-analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    print(json.dumps(result['counts'], indent=2))


if __name__ == '__main__':
    main()
