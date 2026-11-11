"""Derive descriptive execution costs from immutable first-pass traces."""
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path
import statistics

FIRST_PASS = Path('results/first-pass')

def main():
    summary = FIRST_PASS/'approved-pass-summary.json'
    scored = json.loads(summary.read_bytes())
    panels = defaultdict(list)
    unique_seconds = 0
    unique_calls = 0
    for tid, digest in scored['trace_sha256'].items():
        path = FIRST_PASS/'approved-evaluation'/(tid + '.json')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        trace = json.loads(path.read_bytes())
        unique_seconds += sum(c['seconds'] for c in trace['calls'].values())
        unique_calls += len(trace['calls'])
        for row in trace['rows']:
            keys = [row['generation']] + ([row['stage']] if row['stage'] else [])
            calls = [trace['calls'][k] for k in keys]
            panels[row['arm'] + '/' + row['condition']].append({
                'seconds': sum(c['seconds'] for c in calls),
                'prompt_tokens': sum(c['response']['usage']['prompt_tokens'] for c in calls),
                'completion_tokens': sum(c['response']['usage']['completion_tokens'] for c in calls),
                'stage_status': row['stage_status'],
            })
    result = {
        'summary_sha256': hashlib.sha256(summary.read_bytes()).hexdigest(),
        'unique_saved_model_calls': unique_calls,
        'unique_saved_inference_seconds': unique_seconds,
        'limitations': 'Per-outcome costs sum saved generator and upstream LLM calls, including reused calls. They exclude retrieval, gate encoding, initialization and orchestration. Unique totals count each saved call once per parent. Timings include heterogeneous operating conditions and are not controlled latency benchmarks. Missing tool calls are not automatically classified as clarification.',
        'panels': {},
    }
    for name, rows in panels.items():
        result['panels'][name] = {
            'n': len(rows),
            'mean_llm_seconds': statistics.mean(r['seconds'] for r in rows),
            'median_llm_seconds': statistics.median(r['seconds'] for r in rows),
            'mean_prompt_tokens': statistics.mean(r['prompt_tokens'] for r in rows),
            'mean_completion_tokens': statistics.mean(r['completion_tokens'] for r in rows),
            'stage_status_counts': dict(Counter(r['stage_status'] for r in rows)),
        }
    (FIRST_PASS/'approved-pass-efficiency.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
