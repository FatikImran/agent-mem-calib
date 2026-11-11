"""Build query-only rankings for the full release before any tool-call generation."""
import hashlib
import json
import os
from pathlib import Path

from scripts.verify_benchmark_release import verify
from src.data.embedding_cache import load_bge_cache
from src.data.retrieval import HybridIndex


def main():
    os.environ['HF_HUB_OFFLINE']='1'
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8)
    torch.manual_seed(42)
    verify()
    root=Path('data/retrieval-dev')
    vectors,summary=load_bge_cache(root)
    config=summary['config']
    path=Path('data/hf-cache/hub/models--BAAI--bge-m3/snapshots')/config['revision']
    qa=Path('data/Mem2ActBench/Mem2ACTBench_small/qa_dataset.jsonl')
    queries=[(r['qa_id'],r['query']) for r in map(json.loads,qa.read_text(encoding='utf8').splitlines())]
    chunks_file=root/'chunks-128.json'
    manifest=json.loads((root/'manifest.json').read_bytes())
    if hashlib.sha256(chunks_file.read_bytes()).hexdigest()!=manifest['artifact_hashes'][chunks_file.name]:
        raise ValueError('Chunk identity mismatch')
    chunks=json.loads(chunks_file.read_bytes())
    index=HybridIndex(chunks,[vectors[c['text_sha256']] for c in chunks])
    def key(q):return hashlib.sha256(q.encode()).hexdigest()
    missing=sorted({q for _,q in queries if key(q) not in vectors})
    if missing:
        encoder=SentenceTransformer(str(path),device='cpu',trust_remote_code=False)
        if any(len(encoder.tokenizer(q,truncation=False)['input_ids'])>encoder.max_seq_length for q in missing):
            raise ValueError('Query embedding would truncate')
        encoded=encoder.encode(missing,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        vectors.update({key(q):v for q,v in zip(missing,encoded,strict=True)})
    rows={}
    dev={r['task_id']:r['rankings'] for r in json.loads(Path('results/hybrid-development.json').read_bytes())['rows'] if r['chunk_limit']==128}
    for tid,q in queries:
        ranking=index.search(q,vectors[key(q)])
        if tid in dev and ([x['id'] for x in ranking['hybrid']]!=[x['id'] for x in dev[tid]['hybrid']] or [x[0] for x in ranking['bm25']]!=[x[0] for x in dev[tid]['bm25']]):
            raise ValueError(f'Development rank mismatch: {tid}')
        rows[tid]=ranking
    assert len(rows)==400
    output={'embedding_cache_key':summary['cache_key'],'chunk_sha256':manifest['artifact_hashes'][chunks_file.name],
            'qa_sha256':hashlib.sha256(qa.read_bytes()).hexdigest(),'new_query_vectors':len(missing),'rows':rows}
    dest=Path('results/retrieval-400.json')
    tmp=dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(output,indent=2)+'\n',encoding='utf8');tmp.replace(dest)
    print('Ranked',len(rows),'queries; development matches:',len(dev))
if __name__=='__main__':main()
