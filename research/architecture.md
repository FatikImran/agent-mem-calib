# Research workflow architecture

This diagram represents the approved comparison design. The evaluation uses 128-token chunks and a 4,096-token context. The historical source-linked Phase 1 loader is a separate development condition and does not supply evidence to the retrieval experiment.

```mermaid
flowchart TD
    Q[Request] --> B
    K[Target tool schema] --> J
    K --> G
    C[Full released conversation corpus] --> U[User-message text and deterministic chunks]
    U --> B[BM25 ranking]
    U --> D[BGE-M3 dense ranking]
    Q --> D
    B --> R[Reciprocal rank fusion: constant 60, top five]
    D --> R
    B -. Simpler baseline .-> P
    R --> P[Cache clean retrieved evidence]
    P --> V[Clean or one controlled injected memory]
    V --> N[Unfiltered]
    V --> F[Self-ReCheck relevance filter]
    V --> M[MemGate: learned gate only]
    V --> J[Qwen memory-parameter applicability judge]
    Q --> J
    Q --> F
    Q --> M
    J --> A[Validated parameter-scoped admissions]
    N --> G[Qwen tool-call generation]
    F --> G
    M --> G
    A --> G
    Q --> G
    G --> O[Tool call, missing fields, clarification or parse failure]
    O --> E[Paired evaluator]
    T[Gold arguments and injection annotations] --> E
    E --> S[Parameter F1, target-value uptake, joint correctness, reliability and cost]
```

The clean and injected variants reuse retrieved evidence. The controlled memory is supplied before every defense. Gold arguments, source-link annotations and injected-target labels are evaluator inputs; they are excluded from the judge and generator prompts. Provenance remains in saved traces. Prompt-capacity checks must reject overflow rather than silently change evidence across conditions.

MemGate retains its published checkpoint, encoder and threshold 0.07. The main comparator disables checkpoint-restored guidance messages, as explicitly approved. A separate check uses the full released wrapper on the 13 affected queries. A small natural-retrieval injection check remains separate from the main conditional-on-exposure experiment.

## Implementation status

| Component | Current implementation |
|---|---|
| Chunking, BM25, dense-vector ranking and RRF | `src/data/retrieval.py` |
| Completed-cache verification | `src/data/embedding_cache.py` |
| Shared prompts, parameter-scoped rendering, response schema and capacity guard | `src/agent/memory_prompts.py` |
| Approved gate-only MemGate adapter | `src/agent/memgate_adapter.py` |
| Development retrieval and prompt measurements | `scripts/check_hybrid_development.py` |
| Released MemGate example and development scoring | `scripts/probe_memgate_runtime.py` |
| Full comparative generation runner | Integration and development validation pending |

The diagram is not evidence that all comparisons have run. Development measurements, implementation checks and research outcomes are recorded separately.
