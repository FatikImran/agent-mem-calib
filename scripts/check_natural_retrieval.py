"""Approved 100-parent corpus-injection retrieval sensitivity, without generation."""
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from src.data.embedding_cache import load_bge_cache
from src.data.retrieval import HybridIndex, ranked, fuse


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    tmp.replace(path)


def normalize_path_key(path):
    normalized = path.replace('\\', '/')
    if normalized.startswith('phase2/'):
        normalized = 'research/' + normalized[len('phase2/'):]
    return normalized


def normalize_path_hashes(hashes):
    normalized = {}
    for path, value in hashes.items():
        key = normalize_path_key(path)
        if key in normalized:
            raise ValueError(f'duplicate hash path after normalization: {key}')
        normalized[key] = value
    return normalized


def hashes_compatible(saved, expected, *, complete, implementation_path):
    saved = normalize_path_hashes(saved)
    expected = normalize_path_hashes(expected)
    if complete:
        implementation_key = normalize_path_key(str(implementation_path))
        saved.pop(implementation_key, None)
        expected.pop(implementation_key, None)
    return saved == expected


def main():
    root = Path('data/retrieval-dev')
    cohort_path = Path('research/evidence/paired-compute-100-v1.json')
    enrollment_path = Path('research/evidence/paired-enrollment-v1.json')
    cohort = json.loads(cohort_path.read_bytes())
    parents = {p['task_id']:p for p in json.loads(enrollment_path.read_bytes())['parents']}
    variants = [{'task_id':tid,'condition':cond,'text':parents[tid]['annotation'][cond]}
                for tid in sorted(cohort['task_ids']) for cond in ('unrelated','partial')
                if parents[tid]['annotation'][cond]]
    assert len(cohort['task_ids']) == 100 and len(variants) == 104
    manifest = {'status':'approved 7 November 2026','source_hashes':{str(p):digest(p) for p in (cohort_path,enrollment_path)},
        'task_ids':cohort['task_ids'],'variants':variants,
        'protocol':'One unchanged injection added independently as one user-text chunk. Recompute BM25 statistics. Original query; top five total; BGE-M3 plus equal RRF constant 60 and stable ID ties. Verify clean rank parity first. Retrieval only, no generation.',
        'timing':'Supplemental protocol approved after primary results; no outcome-based subset selection.'}
    manifest_path = Path('research/evidence/natural-retrieval-100-v1.json')
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_bytes())
        existing_manifest['source_hashes'] = normalize_path_hashes(existing_manifest['source_hashes'])
        expected_manifest = dict(manifest)
        expected_manifest['source_hashes'] = normalize_path_hashes(expected_manifest['source_hashes'])
        if existing_manifest != expected_manifest:
            raise ValueError('frozen natural-retrieval manifest differs from the current protocol or sources')
    else:
        save(manifest_path,manifest)
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8)
    torch.manual_seed(42)
    vectors, embedding = load_bge_cache(root)
    chunks_path = root/'chunks-128.json'
    ranks_path = Path('results/retrieval-400.json')
    qa_path = Path('data/Mem2ActBench/Mem2ACTBench_small/qa_dataset.jsonl')
    old = json.loads(ranks_path.read_bytes())
    assert digest(chunks_path) == old['chunk_sha256'] and digest(qa_path) == old['qa_sha256']
    queries = {r['qa_id']:r['query'] for r in map(json.loads,qa_path.read_text(encoding='utf8').splitlines())}
    chunks = json.loads(chunks_path.read_bytes())
    key = lambda text:hashlib.sha256(text.encode()).hexdigest()
    texts = sorted({queries[t] for t in cohort['task_ids']} | {v['text'] for v in variants})
    cache = root/'natural-retrieval-vectors.npz'
    cache_meta = root/'natural-retrieval-vectors.json'
    config = {'texts':texts,'revision':embedding['config']['revision'],'normalize':True,'dtype':'float32'}
    if cache.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_bytes())
        assert meta['config'] == config and meta['sha256'] == digest(cache)
        matrix = np.load(cache,allow_pickle=False)['vectors']
    else:
        path = Path('data/hf-cache/hub/models--BAAI--bge-m3/snapshots')/config['revision']
        encoder = SentenceTransformer(str(path),device='cpu',trust_remote_code=False)
        assert all(len(encoder.tokenizer(t,truncation=False)['input_ids']) <= encoder.max_seq_length for t in texts)
        start = perf_counter()
        matrix = encoder.encode(texts,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        np.savez(cache,vectors=matrix)
        save(cache_meta,{'config':config,'sha256':digest(cache),'encoding_seconds':perf_counter()-start})
        del encoder
    assert matrix.shape == (len(texts),1024) and matrix.dtype == np.float32 and np.isfinite(matrix).all()
    assert np.allclose(np.linalg.norm(matrix,axis=1),1,atol=1e-4)
    vectors.update({key(t):v for t,v in zip(texts,matrix,strict=True)})
    base_vectors = np.array([vectors[c['text_sha256']] for c in chunks])
    base = HybridIndex(chunks,base_vectors)
    for tid in sorted(cohort['task_ids']):
        clean = base.search(queries[tid],vectors[key(queries[tid])])
        for method in ('bm25','dense','hybrid'):
            ids = lambda rs:[r['id'] if isinstance(r,dict) else r[0] for r in rs]
            assert ids(clean[method]) == ids(old['rows'][tid][method]), (tid,method)
    print('Verified clean top-five parity for all 100 parents and three rankings.',flush=True)
    sources = [manifest_path,chunks_path,ranks_path,qa_path,cache_meta,Path(__file__),Path('src/data/retrieval.py')]
    hashes = {str(p):digest(p) for p in sources}
    out = Path('results/natural-retrieval-100.json')
    result = {'hashes':hashes,'corpus_chunks':len(chunks),'embedding_cache_key':embedding['cache_key'],'rows':[]}
    expected_pairs = {(v['task_id'], v['condition']) for v in variants}
    if out.exists():
        result = json.loads(out.read_bytes())
        done = {(r['task_id'], r['condition']) for r in result['rows']}
        complete = done == expected_pairs and len(result['rows']) == len(expected_pairs)
        if not hashes_compatible(result['hashes'], hashes, complete=complete, implementation_path=Path(__file__)):
            raise ValueError('saved natural-retrieval result has changed inputs or incompatible generation provenance')
    done = {(r['task_id'],r['condition']) for r in result['rows']}
    for v in variants:
        tid,cond = v['task_id'],v['condition']
        if (tid,cond) in done:
            continue
        injection_id = 'z_injection_'+tid+'_'+cond
        augmented = HybridIndex(chunks+[{'id':injection_id,'text':v['text']}],np.vstack([base_vectors,vectors[key(v['text'])]]))
        qv = vectors[key(queries[tid])]
        lexical = ranked(augmented.ids,augmented.bm25.scores(queries[tid]))
        dense = ranked(augmented.ids,augmented.vectors @ (qv/np.linalg.norm(qv)))
        hybrid = fuse(lexical,dense,k=len(augmented.ids),constant=60)
        row = dict(v,injection_id=injection_id,rankings={})
        for method,ranking in [('bm25',lexical),('dense',dense),('hybrid',hybrid)]:
            ids = [r['id'] if isinstance(r,dict) else r[0] for r in ranking]
            prior = [r['id'] if isinstance(r,dict) else r[0] for r in old['rows'][tid][method]]
            row['rankings'][method] = {'injection_rank':ids.index(injection_id)+1,'top_five':ranking[:5],
                'retrieved':injection_id in ids[:5],'displaced_clean_ids':[i for i in prior if i not in ids[:5]]}
        result['rows'].append(row)
        save(out,result)
        print(len(result['rows']),tid,cond,'hybrid_rank',row['rankings']['hybrid']['injection_rank'],flush=True)
    assert len(result['rows']) == 104
    summary = {'raw_sha256':digest(out),'panels':{}}
    for condition in ('unrelated','partial'):
        rows = [r for r in result['rows'] if r['condition']==condition]
        summary['panels'][condition] = {'n':len(rows),**{m:sum(r['rankings'][m]['retrieved'] for r in rows) for m in ('bm25','dense','hybrid')}}
    save(Path('results/natural-retrieval-100-summary.json'),summary)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
