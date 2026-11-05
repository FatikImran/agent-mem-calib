"""Query-only retrieval primitives; gold/source annotations never rank candidates."""
from collections import Counter
import math
import re

import numpy as np


def lexical_tokens(text):
    return re.findall(r"\w+", text.casefold())


def chunk_text(text, count_tokens, limit):
    """Lossless character-boundary partition, with each piece within token limit."""
    if limit < 1:
        raise ValueError('limit must be positive')
    while text:
        if count_tokens(text) <= limit:
            yield text
            return
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count_tokens(text[:mid]) <= limit:
                lo = mid
            else:
                hi = mid - 1
        if lo == 0:
            raise ValueError('A single character exceeds the token limit')
        boundary = text.rfind(' ', 0, lo + 1)
        end = boundary + 1 if boundary > lo // 2 else lo
        piece, text = text[:end], text[end:]
        # Token counts need not be monotonic across character prefixes.
        while count_tokens(piece) > limit:
            text = piece[-1] + text
            piece = piece[:-1]
        if not piece:
            raise ValueError('Cannot construct a bounded chunk')
        yield piece


def build_chunks(sessions, count_tokens, limit):
    for si, session in enumerate(sessions):
        for ti, turn in enumerate(session.get('turns', [])):
            if turn.get('role') != 'user':
                continue
            for ci, text in enumerate(chunk_text(turn.get('content', ''), count_tokens, limit)):
                if text.strip():
                    yield {'id': f's{si:05d}t{ti:05d}c{ci:05d}', 'text': text,
                           'session_id': session.get('session_id'), 'turn_index': ti,
                           'source_id': turn.get('source_id'), 'tokens': count_tokens(text)}


class BM25:
    """Okapi BM25 with positive log(1 + odds) IDF; query terms counted once."""
    def __init__(self, texts, k1=1.5, b=0.75):
        self.docs = [Counter(lexical_tokens(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs])
        self.average = float(self.lengths.mean()) if self.docs else 0
        self.df = Counter(term for doc in self.docs for term in doc)
        self.k1, self.b = k1, b

    def scores(self, query):
        scores = np.zeros(len(self.docs))
        if not self.average:
            return scores
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.average)
        for term in sorted(set(lexical_tokens(query))):
            freq = np.array([doc[term] for doc in self.docs])
            df = self.df[term]
            idf = math.log(1 + (len(self.docs) - df + .5) / (df + .5))
            scores += idf * freq * (self.k1 + 1) / (freq + norm)
        return scores


def ranked(ids, scores):
    if len(ids) != len(scores) or len(set(ids)) != len(ids):
        raise ValueError('IDs and scores must align and IDs must be unique')
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite retrieval score')
    return sorted(zip(ids, map(float, scores)), key=lambda pair: (-pair[1], pair[0]))


def fuse(bm25_ranking, dense_ranking, k=5, constant=60):
    if constant <= 0 or k < 1:
        raise ValueError('Invalid fusion configuration')
    br = {key: i for i, (key, _) in enumerate(bm25_ranking, 1)}
    dr = {key: i for i, (key, _) in enumerate(dense_ranking, 1)}
    if br.keys() != dr.keys():
        raise ValueError('Rankings must cover the same corpus')
    rows = [{'id': key, 'bm25_rank': br[key], 'dense_rank': dr[key],
             'rrf': 1 / (constant + br[key]) + 1 / (constant + dr[key])} for key in br]
    return sorted(rows, key=lambda row: (-row['rrf'], row['id']))[:k]


class HybridIndex:
    """Rank precomputed dense vectors alongside BM25 without loading annotations."""
    def __init__(self, chunks, vectors):
        self.ids = [c['id'] for c in chunks]
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.ndim != 2 or len(self.vectors) != len(self.ids):
            raise ValueError('Dense vectors must align with chunks')
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        if not np.isfinite(self.vectors).all() or np.any(norms == 0):
            raise ValueError('Dense vectors must be finite and nonzero')
        self.vectors = self.vectors / norms
        self.bm25 = BM25([c['text'] for c in chunks])

    def search(self, query, query_vector, k=5):
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.shape != (self.vectors.shape[1],) or not np.isfinite(vector).all() or not np.linalg.norm(vector):
            raise ValueError('Invalid query vector')
        lexical = ranked(self.ids, self.bm25.scores(query))
        dense = ranked(self.ids, self.vectors @ (vector / np.linalg.norm(vector)))
        return {'hybrid': fuse(lexical, dense, k=k), 'bm25': lexical[:k], 'dense': dense[:k]}
