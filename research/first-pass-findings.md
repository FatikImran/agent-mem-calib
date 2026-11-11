# First evaluation pass · 7 November 2026

All 1,616 expected arm outcomes were scored from 400 parent trace files. Each trace SHA-256 matched the scored manifest. The paired cohort contains 100 held-out parents and four additional partial-relevance variants. The separate retrieval baseline covers all 400 release queries, including development tasks. These populations must not be conflated.

## Main paired results

Literal pooled parameter F1 is shown on a 0–100 scale. Uptake counts correct-tool calls that put the injected candidate value in the target field, divided by all 100 attempts. A parsed call does not imply correct arguments.

| Condition and defense | Parameter F1 | Target-value uptake | Parsed calls |
|---|---:|---:|---:|
| Clean, unfiltered | 21.08 | 1/100 | 83/100 |
| Inapplicable injection, unfiltered | 19.22 | 22/100 | 86/100 |
| Inapplicable injection, MemGate gate-only | 17.58 | 24/100 | 85/100 |
| Inapplicable injection, Self-ReCheck | 11.57 | 4/100 | 29/100 |
| Inapplicable injection, parameter judge | 16.90 | 15/100 | 65/100 |

H1 is supported in this first pass. Injection increases uptake by 21 percentage points, with a paired percentile bootstrap 95% interval of 13 to 29 points. The corresponding F1 change is -1.87 points, with interval -6.20 to +2.11. Increased uptake is clearer than an aggregate F1 decrease.

H2 receives mixed evidence. Self-ReCheck reduces uptake by 18 points and F1 by 7.65 points relative to injected unfiltered calls. Their intervals are -26 to -10 and -13.87 to -1.67 points respectively. MemGate increases uptake by 2 points, with interval -3 to +7, and reduces F1 by 1.64 points. Thus the predicted reduction in drift and minimum three-point cost do not both hold for MemGate in this configuration. The parameter judge reduces uptake by 7 points, with interval -13 to -1, while its F1 change of -2.32 points has interval -8.90 to +4.39. No general superiority claim follows.

Lower uptake must be considered alongside missing tool calls. Self-ReCheck produces only 29 parsed calls on 100 injected attempts. Its low uptake is not sufficient evidence that it preserves useful task completion. Missing calls require response-level review before distinguishing clarification, abstention and format failure.

H3 remains exploratory. On four partial-relevance cases, unfiltered and MemGate each show one uptake; Self-ReCheck and the judge show none. All four arms achieve zero jointly correct target-and-supported-field outcomes. Self-ReCheck produces no parsed calls, and the judge produces one. Relative to the matched unrelated cases, only MemGate shows an uptake increase, by one case. These observations do not establish the general H3 prediction or effective parameter-level separation.

## Retrieval baseline and execution

On all 400 release tasks, BM25 yields F1 22.17 and hybrid retrieval 24.59, a descriptive difference of 2.42 points. Exact tool-and-argument matches are 38 and 42 respectively. These are local reimplementation results, not reproduction of an unpublished paper-table manifest.

There are 19 incomplete Self-ReCheck filters across its 204 variants, handled using the frozen empty-context fallback. No judge validation failures and no generator length stops were recorded. These checks do not establish semantic correctness.

Saved traces contain 1,671 distinct model calls within parent files, totaling 9,773.90 measured inference seconds. On injected variants, median saved LLM-stage costs are 1.47 seconds unfiltered, 1.46 for MemGate generation, 17.66 for Self-ReCheck plus generation, and 7.09 for the judge plus generation. These exclude retrieval, gate encoding and initialization; calls reused across outcomes are attributed to each outcome. Heterogeneous machine load and the overnight suspension prevent interpreting these as controlled end-to-end latency measurements.

## Reproducibility and limits

Raw traces are in results/first-pass/approved-evaluation. Scored outcomes and paired intervals are in results/first-pass/approved-pass-summary.json. Execution aggregates are in results/first-pass/approved-pass-efficiency.json. Reproduce scoring with `python -m scripts.score_approved_evaluation` and descriptive costs with `python -m scripts.summarize_approved_pass` using the recorded environment. The latter verifies every raw trace against the scored hash manifest.

This is one greedy decoding pass. Bootstrap intervals measure sampled-task uncertainty, not repeated-run variability. The study uses one local quantized model, synthetic post-retrieval injections, literal argument scoring and a small partial-relevance cohort. Additional diagnostics must remain separately labeled and must not tune the frozen primary protocol on observed held-out results.
