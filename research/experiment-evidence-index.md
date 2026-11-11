# Experiment evidence index

This index maps experimental claims to public-facing findings and aggregate evidence. Three greedy primary passes are complete and verified. All scored metrics match across passes; one raw multi-call response differs without changing the first-call score. The public repeatability summary is results/primary-repeatability-summary.json.

| Experiment | Evidence | Scope and remaining limitation |
|---|---|---|
| E1 simpler retrieval baseline | research/first-pass-findings.md | 400 release queries, including development tasks; F1 0.221739 |
| E2 base-paper-related approach | research/first-pass-findings.md; research/retrieval-release-audit.md | Local reimplementation, F1 0.245902; not a matched reproduction of published paper-table numbers |
| E3 proposed parameter judge | research/first-pass-findings.md | 100 held-out paired parents plus four partial variants; no claim of general superiority |
| E4 gap-wise evaluation | research/first-pass-findings.md | H1 supported in this setting; H2 mixed; H3 exploratory and unresolved |
| E5 component ablation | scripts/run_scoping_ablation.py and scripts/score_scoping_ablation.py | Removes scoped-field instructions with fixed admissions; full judge removal already represented by unfiltered arm |
| E6 efficiency and reliability | results/first-pass/approved-pass-efficiency.json; results/primary-repeatability-summary.json | Three-pass metric and LLM-stage timing measurements; some pipeline costs excluded |
| Released-wrapper sensitivity | research/memgate-wrapper-findings.md | 13 affected clean queries and five injected variants |
| Corpus-injection sensitivity | results/natural-retrieval-100-summary.json | 104 injections; retrieval exposure only, no downstream generation |

## Gap-resolution evidence

| Gap | Method and metric | Observed first-pass result | Resolution |
|---|---|---|---|
| 1 | Matched clean/injected calls; target-value uptake and literal F1 | Unfiltered uptake 1/100 to 22/100; increase 21 points, paired 95% interval 13–29 | Addressed within the local forced-exposure experiment; no broad novelty claim |
| 2 | Defense comparisons on identical injected cases | MemGate uptake 24/100; Self-ReCheck 4/100 with F1 0.115702 versus unfiltered 0.192192; judge uptake 15/100 | Partially addressed; the predicted trade-off does not hold for every defense |
| 3 | Four matched partial cases; uptake and joint field correctness | Zero joint successes in every arm; tiny domain-concentrated sample | Unresolved; feasibility expansion remains separate future/exploratory work |

Published base-paper scores are not inserted into this table as comparable measurements because the released evaluation manifest and local model/protocol do not establish a matched comparison. Full benchmark and held-out paired populations must remain separate.

## Reproduction entry points

* Primary scoring: `python -m scripts.score_approved_evaluation` (requires locally generated evaluation traces).
* Efficiency: `python -m scripts.summarize_approved_pass`.
* Failure diagnostics: `python -m scripts.analyze_first_pass_failures`.
* Wrapper scoring: `python -m scripts.score_memgate_wrapper`.
* Natural retrieval: retrieval environment, `python -m scripts.check_natural_retrieval`.
* Scoping ablation: GPU environment, `python -m scripts.run_scoping_ablation`, then `python -m scripts.score_scoping_ablation`.
* Fresh repeat passes: `python -m scripts.run_approved_repeats` launches the pinned retrieval/GPU environments and separate pass scorers.

Raw traces and task-level outputs are intentionally excluded from the public Git release. Scoring commands require those local artifacts; reproduce them with the documented preparation and inference workflow before scoring. Supplemental diagnostics were specified after the primary pass and must be identified as such. The primary prompts, cohort and scores remain frozen.
