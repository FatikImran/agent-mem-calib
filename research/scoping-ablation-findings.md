# Fixed-admission scoping ablation · 7 November 2026

This approved supplemental experiment removes the per-memory permitted-parameter prefixes from generator inputs while holding the saved judge verdicts and retained memory texts fixed. A memory is retained if the full judge admitted it for at least one parameter. Query, schema, memory order, decoding mode and budgets remain unchanged. All 204 paired variants were evaluated. Exact primary requests may be reused; this is a component ablation rather than a fresh repeatability pass.

| Panel | Full judge F1 | No-scoping F1 | Full judge uptake | No-scoping uptake | Full judge parsed | No-scoping parsed |
|---|---:|---:|---:|---:|---:|---:|
| Clean, 100 cases | 16.96 | 17.45 | 1 | 1 | 61 | 59 |
| Unrelated injection, 100 cases | 16.90 | 18.48 | 15 | 22 | 65 | 68 |
| Partial injection, four cases | 0.00 | 50.00 | 0 | 1 | 1 | 3 |

F1 is displayed on a 0–100 scale. Removing scope instructions increases unrelated-injection uptake by seven percentage points, with a paired grouped percentile bootstrap 95% interval of +2 to +12 points. The F1 change is +1.58 points, with interval -5.15 to +8.81. Thus scope instructions contribute resistance to target-value uptake in this panel, while the F1 difference is uncertain. This result does not establish improved overall task utility or solve the judge's scope-classification errors.

On the four partial cases, neither full nor ablated judge achieves joint correctness. The ablation increases parsed calls from one to three and uptake from zero to one. Its higher pooled F1 is not evidence of successful two-field separation. The partial panel remains too small for population claims.

The clean panel has unchanged uptake and an F1 change of +0.49 points, with interval -4.35 to +5.20. Confidence intervals use the frozen source/query grouping, 10,000 draws and seed 42. Comparisons are ablation minus full judge. All enrolled attempts remain in the denominator.

Raw requests, responses, admissions, retained IDs and primary-trace hashes are in results/scoping-ablation.json. Scores and paired intervals are in results/scoping-ablation-summary.json. Reproduce with the GPU environment using `python -m scripts.run_scoping_ablation`, then `python -m scripts.score_scoping_ablation`. The scorer verifies complete 204-variant coverage and source hashes. The experiment was specified after the primary results and does not change the primary protocol.
