# MemGate wrapper sensitivity · 7 November 2026

The approved bounded comparison covers all 13 release queries whose released wrapper rules add guidance, comprising three safety and ten objective-guidance triggers. All 13 belong to the frozen held-out pool; five also belong to the prospectively selected 100-parent paired cohort. The panel contains 13 hybrid clean variants and five unrelated injection variants. No affected query contributes a partial-relevance variant. Clean paired inputs duplicate the hybrid clean inputs and are counted once.

Both arms use the pinned released checkpoint and MiniLM encoder, threshold 0.07, preserved candidate order and no top-k truncation at the gate. The full wrapper retains checkpoint guidance. For each variant, direct released-wrapper execution was checked against the gate's thresholded candidates preceded by the triggered guidance text. Generation uses the frozen local Qwen prompt and budgets. All measured prompts fit. Existing primary requests were reused only on exact request equality, covering 15 of the 36 arm outcomes. There are 34 unique saved requests because some variants coincide.

| Panel | Arm | N | Parameter F1 (0–100) | Parsed calls | Exact calls | Target uptake |
|---|---|---:|---:|---:|---:|---:|
| Affected hybrid clean | Gate-only | 13 | 24.00 | 11 | 1 | Not evaluated |
| Affected hybrid clean | Full wrapper | 13 | 23.08 | 12 | 0 | Not evaluated |
| Affected unrelated injection | Gate-only | 5 | 32.00 | 5 | 0 | 1 |
| Affected unrelated injection | Full wrapper | 5 | 38.46 | 5 | 0 | 1 |

Parsed call content changes on six of 13 clean variants and four of five injected variants. Guidance therefore changes generation despite identical gate decisions. The clean F1 change is -0.92 points; the injected change is +6.46 points. Injected target uptake remains one of five. These small descriptive panels support neither a general benefit nor equivalence between wrappers. The primary gate-only findings remain separate and unchanged.

Raw inputs, guidance, scores and provenance are in results/memgate-wrapper-inputs.json. Full generation traces and reuse provenance are in results/memgate-wrapper-traces.json. Scored outcomes and every changed call are in results/memgate-wrapper-summary.json.

Reproduction from the project root:

```powershell
.\data\runtime\retrieval\Scripts\python.exe -m scripts.check_memgate_wrapper prepare
.\data\runtime\generator-cuda\Scripts\python.exe -m scripts.check_memgate_wrapper run
python -m scripts.score_memgate_wrapper
```

Preparation is offline and checks pinned source and checkpoint hashes. The first preparation attempt requested an entire local encoder snapshot, including unused ONNX files, and failed before inference. Restricting the offline request to the same model files as the established encoder workflow resolved it without downloading or changing weights. Resume validates input and code hashes; scoring verifies complete arm coverage and source trace hashes.
