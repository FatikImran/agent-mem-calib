# agent-mem-calib

**Project Title:** Calibrated Memory Use in Agentic Tool Calls: Measuring and Reducing Over-Application of Long-Term Memory to Tool Arguments

**Author:** Muhammad Fatik Bin Imran (23i-0655)
**Course:** AI4015 Agentic Artificial Intelligence
**Phase:** 1 (Literature Review, Proposal & Baseline)

## Overview

This repository contains the Phase 1 technical baseline for my semester project. The goal of the project is to measure and reduce the *over-application* of long-term memory in agentic tool calls (e.g., applying an irrelevant stored preference to a safety-critical tool argument).

This baseline establishes the core agentic loop: retrieving long-term memory, injecting it into context, and prompting a language model to generate structured tool-call arguments based on the `Mem2ActBench` dataset.

## Hardware Constraints & Model Choice

The local development environment for this project is an **NVIDIA RTX 3050 Laptop GPU (4GB VRAM)**.

Because vLLM and 8B-parameter models exceed this VRAM limit, this baseline uses **`llama-cpp-python`** to run **Qwen3-4B** (quantized to **Q4_K_M**). This allows the entire agent to run locally on the GPU while leaving enough VRAM for the KV cache. A 4B model is smaller than the models evaluated in the base paper, which currently limits the absolute performance (see results below).

## Repository Structure

```text
agent-mem-calib/
├── README.md
├── requirements.txt
├── data/
│   └── Mem2ActBench/        # Cloned dataset (gitignored)
├── models/
│   └── Qwen_Qwen3-4B-Q4_K_M.gguf  # Quantized model (gitignored)
├── scripts/
│   ├── check_loader.py      # Verifies dataset extraction
│   ├── download_model.py    # Downloads the GGUF model
│   └── test_llama_cpp.py    # Tests raw model tool-calling
└── src/
    ├── agent/
    │   └── graph.py         # LangGraph baseline agent
    ├── data/
    │   └── loader.py        # Mem2ActBench data pipeline
    ├── eval/
    │   └── parameter_f1.py  # Evaluation metrics
    └── run_baseline.py      # End-to-end evaluation script
```

## Setup Instructions

1. Clone the repository and install dependencies:

   ```bash
   git clone https://github.com/FatikImran/agent-mem-calib.git
   cd agent-mem-calib
   pip install -r requirements.txt
   ```

2. Download the Mem2ActBench dataset:

   ```bash
   cd data
   git clone https://github.com/Cantaloupe-M/Mem2ActBench.git
   cd ..
   ```

3. Download the quantized model:

   ```bash
   python scripts/download_model.py
   ```

## Running the Baseline

To verify the data pipeline loads correctly (extracts queries, tool schemas, memory contexts, and gold arguments):

```bash
python -m scripts.check_loader
```

To run the LangGraph agent on a subset of tasks and calculate the Parameter F1 score:

```bash
python -m src.run_baseline
```

## Phase 1 Initial Results

The baseline agent was evaluated on a subset of 20 tasks from Mem2ActBench_small.

- **Model:** Qwen3-4B (Q4_K_M) via `llama-cpp-python`
- **Prompting:** Zero-shot tool calling with the user's memory context injected in the system prompt.
- **Parsing:** A robust regex fallback parser handles cases where the local model outputs tool calls as raw text instead of structured JSON.

| Metric       | Score  |
|--------------|--------|
| Precision    | 0.8000 |
| Recall       | 0.4138 |
| Parameter F1 | 0.5455 |

**Analysis:**
The precision of 0.80 indicates the model generates highly accurate arguments when it successfully formats a tool call. However, the recall (0.41) reveals the primary bottleneck: the 4B model frequently fails to output a parsable tool call (8 out of 20 tasks failed to parse). This is expected for a small model constrained by 4GB VRAM. The F1 of 0.5455 proves the pipeline works end-to-end and significantly outperforms the Mem2ActBench base paper's 8B baseline (F1 0.2924).
