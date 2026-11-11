"""Verify the locally retained held-out split against the pinned benchmark."""
import json
from collections import defaultdict
from pathlib import Path

from scripts.verify_benchmark_release import verify

OUT = Path('research/evidence/heldout-split-v1.json')
QA_RELATIVE_PATH = 'Mem2ACTBench_small/qa_dataset.jsonl'
EXPECTED_COUNTS = {
    'development_exposed': 46,
    'development_linked': 3,
    'development': 49,
    'heldout': 351,
}


def _task_ids(manifest, key):
    values = manifest.get(key)
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(f'Frozen split field must be a list of task IDs: {key}')
    if len(values) != len(set(values)):
        raise ValueError(f'Frozen split contains duplicate task IDs: {key}')
    if values != sorted(values):
        raise ValueError(f'Frozen split task IDs are not sorted: {key}')
    return set(values)


def validate_manifest(manifest, tasks, release):
    name = QA_RELATIVE_PATH
    expected_hash = release['files'][name]['sha256_lf']
    if manifest.get('version') != 1:
        raise ValueError('Unsupported frozen held-out split version')
    if manifest.get('release_commit') != release['commit']:
        raise ValueError('Frozen split benchmark revision differs from the verified release')
    if manifest.get('qa_sha256_lf') != expected_hash:
        raise ValueError('Frozen split QA dataset hash differs from the verified release')

    task_ids = [task.get('qa_id') for task in tasks]
    if any(not isinstance(task_id, str) for task_id in task_ids) or len(task_ids) != len(set(task_ids)):
        raise ValueError('Benchmark QA dataset contains invalid or duplicate task IDs')
    benchmark_ids = set(task_ids)

    exposed = _task_ids(manifest, 'development_exposed')
    linked = _task_ids(manifest, 'development_linked')
    development = _task_ids(manifest, 'development')
    heldout = _task_ids(manifest, 'heldout')
    if exposed & linked or exposed | linked != development:
        raise ValueError('Frozen development membership does not match exposed and linked tasks')
    if development & heldout or development | heldout != benchmark_ids:
        raise ValueError('Frozen development/held-out membership does not partition the benchmark')
    for key, expected in EXPECTED_COUNTS.items():
        if len(_task_ids(manifest, key)) != expected:
            raise ValueError(f'Unexpected frozen split count for {key}')
    if len(benchmark_ids) != sum(EXPECTED_COUNTS[key] for key in ('development', 'heldout')):
        raise ValueError('Benchmark QA task count differs from the approved split size')

    groups = defaultdict(set)
    for task in tasks:
        task_id = task['qa_id']
        for source_id in task.get('source_conversation_ids', []):
            if source_id:
                groups['source', source_id].add(task_id)
        query = ' '.join(task.get('query', '').casefold().split())
        if query:
            groups['query', query].add(task_id)
    if any(group & development and group & heldout for group in groups.values()):
        raise ValueError('A source or normalized query crosses the frozen split')
    return manifest


def build(manifest_path=OUT):
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f'Frozen split manifest is missing: {manifest_path}. '
            'This task-level file is intentionally local-only; restore your preserved '
            'research/evidence/heldout-split-v1.json to verify or prepare this exact evaluation.'
        )

    release = verify()
    qa_path = Path('data/Mem2ActBench') / QA_RELATIVE_PATH
    tasks = [
        json.loads(line)
        for line in qa_path.read_text(encoding='utf8').splitlines()
    ]
    manifest = json.loads(manifest_path.read_bytes())
    return validate_manifest(manifest, tasks, release)


if __name__ == '__main__':
    build()
    print('Verified 49 development and 351 held-out parent tasks.')
