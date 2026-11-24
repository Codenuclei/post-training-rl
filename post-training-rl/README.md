# Post-Training RL Pipeline

> **Goal**: RL post-training for PhD-level math (Putnam competition) using Qwen3.5-9B on Modal GPU infrastructure.
>
> **Baseline scores** (Qwen3.5-9B, no RL training): GSM8K 50%, Putnam-AXIOM 50%, AIME 50%, MATH-500 20%
>
> **Target after RL**: GSM8K 70-80%, Putnam 60-70%, AIME 60-70%

---

## Architecture

```
post-training-rl/
├── README.md              ← you are here
├── calculator/            ← Agentic calculator with tool-use harness
│   ├── calculator.py      ← 3 tools: calculator, formula_sheet, verify (55/55 tests)
│   ├── modal_calc.py      ← Modal integration with Qwen3.5-2B on T4
│   └── README.md          ← Calculator docs
├── pipeline/              ← RL post-training pipeline
│   ├── rl_training.py     ← GRPO + RIPO + Score Centering training loop
│   ├── benchmark.py       ← 20+ dataset benchmarking harness
│   ├── baseline_eval.py   ← Qwen3.5-9B baseline evaluation on Modal L40S
│   ├── run_pipeline.py    ← Orchestration: --test, --analyze, --train, --benchmark
│   ├── analyze_putnam.py  ← Putnam-AXIOM dataset analysis
│   ├── probe_formulas.py  ← Formula knowledge probe
│   ├── modal_formula_sheet.py ← Formula sheet generation
│   ├── analysis.json      ← Putnam dataset analysis (31KB)
│   ├── probe_results.json ← Formula probe results (17/17 correct)
│   ├── topic_to_formulas.json ← 90 formulas across 13 math topics
│   ├── baseline_scores.md ← Qwen3.5-9B baseline scores
│   ├── formula_sheet_prompt.txt ← Prompt template for formula generation
│   └── README.md          ← Pipeline docs
├── chat/                  ← Qwen3.5-9B interactive chat server
│   └── modal_chat.py      ← @app.cls pattern, model stays warm in VRAM
├── tiny-gpu/              ← 4GB VRAM solution
│   └── modal_4gb.py       ← Qwen3.5-2B on T4 with llama.cpp GGUF
├── docs/                  ← Lessons learned
│   ├── modal-lessons.md   ← 7 Modal infrastructure lessons
│   └── session-lessons.md ← 12 session lessons (model selection, thinking mode, etc.)
└── logs/                  ← Baseline evaluation logs
    ├── baseline_9b_results.log    ← Final 9B scores
    ├── baseline_async_full.log    ← Async parallel run
    ├── baseline_async_results.log ← Async results
    ├── baseline_clean_results.log ← Cleaned results
    ├── baseline_final_results.log ← Final results
    ├── baseline_fixed_results.log ← Fixed thinking mode results
    └── baseline_results.log       ← First 2B baseline attempt
```

---

## Quick Start

### Prerequisites

- Modal account with billing set up: `modal token new`
- Python 3.10+ with `.venv` at project root

### Run Local Tests (No GPU)

```bash
.venv/bin/python3 post-training-rl/pipeline/run_pipeline.py --test
```

### Baseline Evaluation (Modal L40S, ~$2)

```bash
modal run post-training-rl/pipeline/baseline_eval.py
```

### RL Training (Modal A100 80GB, ~$2.50/hr)

```bash
modal run post-training-rl/pipeline/run_pipeline.py::train
```

### Benchmark on 20+ Datasets

```bash
modal run post-training-rl/pipeline/run_pipeline.py::benchmark
```

### Interactive Chat (Modal L40S, ~$2/hr)

```bash
modal run post-training-rl/chat/modal_chat.py --interactive
```

---

## Key Techniques

### RIPO (Riemannian Isometric Policy Optimization)

Replaces Euclidean clipping in PPO with isometric updates on the Riemannian manifold. Dynamically adapts the clipping boundary to local geometry, permitting larger updates for low-probability actions while constraining high-probability actions. Up to 60% improvement over GRPO on AIME24.

**Paper**: "Beyond Euclidean Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian Isometric Policy Optimization" (ICML 2026)

### Score Centering

Additive drift correction for off-policy RL stability. Subtracts the expected score under the sampler distribution at each token prefix, canceling the drift term entirely without importance sampling ratios. No variance inflation, works under severe quantization.

**Paper**: "Score Centering Stabilizes Off-policy Reinforcement Learning" (arXiv 2609.20807, Together AI)

### Agentic Calculator

3-tool harness that amplifies small model capability:

1. **`calculator`**: Safe Python eval with sympy + math modules, timeout protection, code injection prevention
2. **`formula_sheet`**: 90 formulas across 13 math topics (number theory, combinatorics, algebra, calculus, geometry, sequences, probability, linear algebra, real analysis, complex analysis, ODEs, functional equations, trigonometry)
3. **`verify`**: Answer checking with math-verify library

The model reasons about *what* to compute; the tools handle execution. 55/55 edge case tests pass.

---

## Baseline Scores (Qwen3.5-9B, No RL)

| Benchmark | Score | Correct/Total |
|---|---|---|
| GSM8K | **50%** | 5/10 |
| Putnam-AXIOM | **50%** | 5/10 |
| AIME | **50%** | 5/10 |
| MATH-500 | **20%** | 2/10 |

**Setup**: Qwen3.5-9B BF16, Modal L40S (48GB), thinking mode ON, 2048 max tokens, batch size 3.

---

## Modal GPU Cost Reference

| GPU | VRAM | $/hr | Our Use Case |
|---|---|---|---|
| T4 | 16 GB | $0.59 | Tiny model inference (4GB solution) |
| L40S | 48 GB | $1.95 | Qwen3.5-9B chat + baseline eval |
| A100 80GB | 80 GB | $2.50 | RL training with GRPO/RIPO |
| H100 | 80 GB | $3.95 | FP8 training, multi-GPU scale |

---

## Formula Knowledge Probe

Qwen3.5-9B was probed on 17 Putnam-level math formulas. Result: **17/17 correct (100%)**.

The model knows all formulas but the formula sheet is kept as a convenience tool during RL training to reduce reasoning burden.

---

## Lessons Learned

- **Model selection**: 9B on L40S ($1.95/hr) is the sweet spot — 5x better baseline than 2B on T4
- **Thinking mode**: Disable for structured output, enable for reasoning evaluation
- **Save raw outputs**: Never trust automated scorers without manual spot-checking
- **Batch, don't loop**: Parallel generation is 5x faster than sequential
- **Use math-verify**: Don't roll your own answer verifier
- **Clamp Fisher info**: RIPO needs `fisher_info.clamp(min=1e-8)` to avoid NaNs
- **Check dataset splits**: Putnam-AXIOM uses `full_eval`/`variations`, not `val`/`test`
- **Force-add logs**: `.gitignore` blocks `*.log` — use `git add -f`

See `docs/session-lessons.md` for full details on all 12 lessons.

---

## Next Steps

1. **RL training** on Modal A100 80GB (~$2.50/hr) with Qwen3.5-9B
2. **Benchmark** on 20+ datasets to measure transfer
3. **Looped transformer** inference from scratch (original plan)

---

## Session Summary

Built over ~2 hours on November 24, 2026. Total Modal GPU cost: ~$3.30 across formula probes, baseline evaluations, and testing.

| Component | Status | Details |
|---|---|---|
| Agentic Calculator | ✅ 55/55 tests pass | 3 tools, safe eval, formula sheet, verify |
| Formula Probe | ✅ 17/17 formulas correct | Qwen3.5-9B knows all Putnam-level formulas |
| RL Training | ✅ All tests pass | GRPO + RIPO + Score Centering |
| Benchmark Harness | ✅ All tests pass | 20+ datasets |
| Modal Integration | ✅ Verified | Qwen3.5-9B chat on L40S, baseline eval |
| Baseline Scores | ✅ Saved | GSM8K 50%, Putnam 50%, AIME 50%, MATH-500 20% |
| Orchestration | ✅ Ready | `run_pipeline.py` with --test, --analyze, --train, --benchmark |
