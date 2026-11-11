# Public release audit — 5 November 2026

Status: source audit completed; all five protocol recommendations approved on 5 November 2026. Numeric chunk and context budgets remain pending. No evaluation run or checkpoint inference was performed in this audit. No privately supplied evaluation package is available.

## Inspected sources

| Repository | Inspected HEAD |
|---|---|
| https://github.com/Cantaloupe-M/Mem2ActBench | c935c65be9aa19c930b60657c0c4b2b7ea13a34a |
| https://github.com/Kevin-Zh-CS/MemGate | 05f89d003854e99523b2c74df54db7b0d2fa2643 |
| https://github.com/yulinlp/OP-Bench | 44e7d0e526152997b04208218a8ad7362174dae4 |

Raw source snapshots and a machine-readable inventory are in `research/evidence/repo-audit`. OP-Bench Python snapshots were downloaded from the pinned commit; MemGate snapshots were downloaded from main during the audit and have local SHA-256 hashes. The checkpoint is stored under `data/audit` and hashed separately. Regenerate local inventories with `python -m scripts.record_repo_audit`. Hashes establish the inspected bytes; regenerating a manifest does not independently verify upstream authenticity.

The repositories were cloned with `--filter=blob:none --no-checkout`. GitHub API requests failed first through DNS and then rate limiting. Some Git blob fetches also failed. HEAD trees and refs were readable, and selected source files were downloaded directly. Historical file-content coverage is incomplete; absence claims below concern the inspected HEAD, not every historical commit or unpublished artifact. The pre-existing `data/Mem2ActBench` directory has no independent Git metadata: Git commands there resolve to the project repository. Its parent commit must not be cited as the benchmark release revision.

## Findings

### Mem2ActBench

The inspected tree contains data and construction scripts, but no identifiable evaluation runner implementing the paper's retrieval and scoring experiment. The paper specifies BGE-M3 and reports BM25, dense and hybrid retrieval at several k values; hybrid k=5 is the strongest passive retrieval setting in Table 4. The inspected release does not establish evaluation chunk boundaries, exact hybrid fusion, prompt packing/truncation or a verified table-specific task manifest.

The local full release has 558 QA records and 2,029 sessions; the small release has 400 QA records and 429 sessions. All 400 small query/gold pairs match the larger release (previous release inspection). The construction function uses random seed 42 to sample QA records and retains sessions linked to those records, then renumbers identifiers. This establishes how the function constructs a small release; it does not prove that the released small file is the exact paper evaluation manifest. The paper reports 400 tasks and 2,029 sessions.

Current source-linked loading is an oracle-location development condition. Source annotations must not choose retrieval candidates. The 429-session subset is selected using answer-source links and is a less demanding candidate pool than the full corpus. Numeric paper parity cannot be claimed with the present model substitution and unverified evaluation details.

### MemGate

The public example uses the released single-sided gate, 384-dimensional embeddings, CPU inference and threshold 0.07. Its default embedding encoder is all-MiniLM-L6-v2. BGE-M3 retrieval therefore requires a separate MiniLM embedding step for gate scoring. Reusing BGE vectors in this checkpoint is unsupported.

The example explicitly resets the threshold after checkpoint loading; loading can overwrite configuration. Candidate scores are thresholded, with an optional top-k cap. Follow-up inspection established that `preserve_order=True` is the default: retained candidates keep their input order unless explicitly requested otherwise. This corrects the initial audit's statement that retained candidates were always ranked. Loading can also restore safety/objective memories. Integration must inspect loaded settings and log any extra memory insertion. The checkpoint bytes were downloaded (36,785,077 bytes); successful runtime loading and behavior remain untested. The repository describes inference release rather than gate training reproduction.

### OP-Bench relevance filter

The released `src/opbench/postprocessing.py` implements `self_recheck`: keep original sentences directly useful for the question, do not rewrite or answer, and return `NO RELEVANT CONTEXT` if none apply. It sends question and context to a language model at temperature zero. This is a sentence-level relevance comparator, not a parameter-level applicability check. Using this prompt with local Qwen3-4B is a model/task adaptation. Preserve raw responses, the empty-context sentinel, errors and any departures from the released prompt.

## Recommendations — approved 5 November 2026

1. Use the 400 released small QA records with the full 2,029-session corpus. Describe this as a documented local reimplementation; do not claim exact numerical reproduction of the paper. Preserve the separate development/held-out distinction for the proposed method.
2. Use query-only BM25 plus dense BGE-M3 retrieval, fused by equal-weight reciprocal rank fusion with constant 60, taking five chunks. RRF is our explicit implementation choice, not a verified paper setting. Stable IDs break ties. Cache retrieval for all matched conditions and log both component ranks. Use BM25 alone as the inexpensive simpler retrieval baseline.
3. Develop deterministic chunks from user-message text in corpus order, retaining source/session/turn provenance only as metadata. Exclude structured historical tool-call payloads from this primary condition. This makes the memory evidence policy explicit and avoids indexing direct structured answer traces. Chunk size and prompt budget require a development-only fit check because the current runtime context is 2,048 tokens; do not silently truncate a task or change the budget across defenses. User-message text can itself contain quoted history; flag this in the corpus audit. Freeze the numeric packing settings before held-out inference, with approval.
4. For the main paired experiment, retrieve once on the clean request, then add one controlled memory before every defense, including MemGate. Give the clean arm the same reserved injection capacity. Keep legitimate evidence identical across the pair and preserve injection position across matched injection types. Report this as conditional on exposure. Run a small separately labeled corpus-injection sensitivity check to measure whether injections retrieve naturally; do not promise a second full benchmark or infer natural attack success from forced exposure.
5. Use the published MemGate checkpoint with its MiniLM encoder and 0.07 threshold; use the released Self-ReCheck prompt with local Qwen. Keep all candidate memories eligible for defense processing (do not accidentally discard the appended injection through a five-item cap). Log gate order changes and loaded settings. Do not tune these comparators on held-out results.

These choices limit training and repeated retrieval work, preserve a paired parameter-level study, and expose deviations from the release. They do not establish publishability or sufficient partial-relevance sample size. Runtime cost, valid-case yield, human review and the remaining generation/scoring settings still require verification before a full run.

## Public Git release protocol — 11 November 2026

The public repository will contain project code and non-text aggregate evidence only. Do not commit copies of the benchmark dataset, raw inference/retrieval traces, task-level outputs, or benchmark-derived prompt, memory, request, and response text. Aggregate counts, metrics, uncertainty estimates, and provenance hashes may be included when they do not reproduce benchmark content.

Obtain the benchmark separately from the official [Mem2ActBench repository](https://github.com/Cantaloupe-M/Mem2ActBench) at the pinned revision `c935c65be9aa19c930b60657c0c4b2b7ea13a34a`, using the clone and detached-checkout commands in the root README. Keep the downloaded data in the ignored local `data/Mem2ActBench` directory; do not vendor it into this project. The upstream README states “MIT License,” but this protocol does not assert that the label resolves redistribution rights for the dataset or its source materials.
