"""Materialize two provisional development chunkings and a deduplicated text cache."""
import hashlib
import json
from pathlib import Path

from llama_cpp import Llama

from src.data.retrieval import build_chunks
from scripts.verify_benchmark_release import verify

ROOT = Path('data/retrieval-dev')
CORPUS = Path('data/Mem2ActBench/Mem2ACTBench/Mem2ACT_conversation.jsonl')
QA = Path('data/Mem2ActBench/Mem2ACTBench_small/qa_dataset.jsonl')
DEV_IDS = {'qa_063', 'qa_220', 'qa_311', 'qa_327'}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def main():
    verify()
    ROOT.mkdir(parents=True, exist_ok=True)
    llm = Llama(model_path='models/Qwen_Qwen3-4B-Q4_K_M.gguf', vocab_only=True, verbose=False)
    count = lambda s: len(llm.tokenize(s.encode('utf8'), add_bos=False, special=True))
    sessions = [json.loads(line) for line in CORPUS.read_text(encoding='utf8').splitlines()]
    texts = {}
    counts = {}
    for limit in (96, 128):
        chunks = list(build_chunks(sessions, count, limit))
        for chunk in chunks:
            key = hashlib.sha256(chunk['text'].encode('utf8')).hexdigest()
            chunk['text_sha256'] = key
            texts[key] = chunk['text']
        write_json(ROOT / f'chunks-{limit}.json', chunks)
        counts[str(limit)] = len(chunks)
        print(f'{limit}: {len(chunks)} chunks', flush=True)
    tasks = [json.loads(line) for line in QA.read_text(encoding='utf8').splitlines()
             if json.loads(line)['qa_id'] in DEV_IDS]
    # Only the query text goes into the embedding queue, never schema or answers.
    queries = {t['qa_id']: t['query'] for t in tasks}
    for text in queries.values():
        texts[hashlib.sha256(text.encode('utf8')).hexdigest()] = text
    write_json(ROOT / 'texts.json', [{'sha256': key, 'text': texts[key]} for key in sorted(texts)])
    write_json(ROOT / 'queries.json', queries)
    write_json(ROOT / 'manifest.json', {'status': 'development candidates; budgets not frozen',
        'counts': counts, 'unique_texts_including_queries': len(texts),
        'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (CORPUS, QA)},
        'chunking_script_sha256': hashlib.sha256(Path('src/data/retrieval.py').read_bytes()).hexdigest(),
        'artifact_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / 'chunks-96.json', ROOT / 'chunks-128.json', ROOT / 'texts.json', ROOT / 'queries.json')}})
    print(f'{len(texts)} unique texts to embed', flush=True)


if __name__ == '__main__':
    main()
