"""Explicit gate-only adaptation of the inspected released MemGate wrapper."""


def configure_gate_only(defender):
    """Apply after loading: checkpoints restore guidance and threshold settings."""
    defender.threshold = 0.07
    defender.safety_memory = None
    defender.objective_memory = None
    defender.eval()
    return defender


def filter_gate_only(defender, query, memories):
    if defender.safety_memory is not None or defender.objective_memory is not None or defender.threshold != 0.07:
        raise ValueError('Gate-only configuration must be applied after checkpoint loading')
    texts = [m['text'] for m in memories]
    kept, kept_scores, _ = defender(query=query, candidates=texts, top_k=None, preserve_order=True)
    scores = defender.score_candidates(query, texts)
    indices = [i for i, score in enumerate(scores) if score > defender.threshold]
    if [texts[i] for i in indices] != kept:
        raise ValueError('Published wrapper output differs from thresholded gate scores')
    return {'scores': scores, 'kept_ids': [memories[i]['id'] for i in indices],
            'kept_texts': kept, 'kept_scores': kept_scores, 'threshold': defender.threshold}
