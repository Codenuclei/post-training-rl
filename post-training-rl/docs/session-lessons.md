# Session Lessons Learned — Putnam-AXIOM RL Pipeline

> Session: November 24, 2026
> Goal: Build RL post-training pipeline for PhD-level math on small models via Modal GPU
> Total mistakes made: ~10, costing ~$5 in GPU time and ~2 hours of debugging

---

## 1. Model Selection — 2B vs 9B: The Wrong Starting Point

**Mistake**: Started baseline evaluation with Qwen3.5-2B on T4 GPU because it was cheaper.

**Reality**: Qwen3.5-2B scores 10% on GSM8K and 0% on MATH-500. Qwen3.5-9B scores 50% on GSM8K, Putnam, and AIME, and 20% on MATH-500 — **5x better baseline** for only $1.36/hr more ($1.95 vs $0.59).

**Lesson**: For RL post-training, the base model's starting capability matters more than GPU cost. A 9B model at 50% baseline gives RL a signal to optimize. A 2B model at 10% gives RL almost nothing to work with — the reward is too sparse.

**Rule**: Start with the largest model that fits your budget. Qwen3.5-9B on L40S ($1.95/hr) is the sweet spot for sub-10B RL work.

| Model | GSM8K | Putnam | AIME | MATH-500 | GPU | $/hr |
|---|---|---|---|---|---|---|
| Qwen3.5-2B | 10% | 20% | 10% | 0% | T4 | $0.59 |
| **Qwen3.5-9B** | **50%** | **50%** | **50%** | **20%** | **L40S** | **$1.95** |

---

## 2. Thinking Mode — The Silent Token Filler

**Mistake**: Ran formula knowledge probe with Qwen3.5-9B's thinking mode enabled. The model filled all 512 output tokens with "Thinking Process:" preamble and never reached the actual formulas. Two runs wasted on this.

**Reality**: Qwen3.5 has a built-in thinking mode that, when enabled, dominates the output budget. For direct formula extraction or structured output, you need `enable_thinking=False`.

**Lesson**: Disable thinking mode when you want structured, concise output. Enable it only when you want the model to reason through a problem step-by-step (like in baseline evaluation where reasoning quality matters).

**Pattern**:
```python
# For formula extraction, structured output:
tokenizer.apply_chat_template(msgs, enable_thinking=False)

# For reasoning evaluation, baseline scores:
tokenizer.apply_chat_template(msgs, enable_thinking=True, max_new_tokens=2048)
```

---

## 3. Formula Probe — Save Exact Outputs, Not Just Scores

**Mistake**: Ran formula probe twice, both times getting seemingly low scores (~27%). Automatically concluded the model doesn't know formulas and removed the formula sheet from the agentic calculator. Then had to restore it after manual review showed 17/17 correct.

**Reality**: The automated keyword scorer expected English prose keywords in pure LaTeX output. The model knew every formula perfectly — the scorer was broken.

**Lesson**: **Always save the raw model outputs** alongside automated scores. Never trust an automated scorer without manual spot-checking. A broken scorer can lead you to delete working code.

**Pattern**: Save `probe_results.json` with full responses, not just pass/fail. Review manually before making architectural decisions based on probe data.

---

## 4. Agentic Calculator — Tools Amplify Small Models

**Insight**: A 2B model can reason about *what* to compute but not *compute* reliably. Offloading arithmetic to a safe Python evaluator (sympy + math modules) removes the arithmetic bottleneck while the model focuses on reasoning.

**Why it works**: In PhD-level math (Putnam), the challenge is matching the problem to the right formula and setting up the computation, not doing the arithmetic. Tools let the model do what it's good at (reasoning) and offload what it's bad at (calculation).

**Implementation**: 3 tools — `calculator` (safe eval), `formula_sheet` (57 formulas across 13 topics), `verify` (answer checking). 55/55 edge case tests pass.

**Lesson**: For small models (<10B), tool-use is not optional — it's essential. The model's job is reasoning; the tools handle execution.

---

## 5. Formula Sheet — Keep It as a Convenience Tool

**Decision**: Even though Qwen3.5-9B knows all Putnam-level formulas (17/17 correct in probe), we keep the formula sheet as a tool-use convenience during RL training.

**Why**: During RL training, the model is under pressure to generate correct answers quickly. Having the formula sheet as a tool reduces the reasoning burden — the model can query the formula sheet instead of recalling it from memory, freeing up capacity for the actual problem-solving.

**Lesson**: Just because a model *knows* something doesn't mean it should *recall* it under RL pressure. Tools reduce cognitive load during training.

---

## 6. Baseline Evaluation — Thinking Mode Matters

**Mistake**: First baseline run used `enable_thinking=False` for Qwen3.5-2B. The model gave terse answers without reasoning, scoring artificially low.

**Fix**: Switched to `enable_thinking=True` with 2048 max tokens for Qwen3.5-9B. The model now shows full reasoning chains, and the scores reflect actual capability.

**Lesson**: For baseline evaluation of reasoning models, enable thinking mode and give enough tokens for the full reasoning chain. Qwen3.5 needs ~2048 tokens for thinking + answer.

---

## 7. Async Parallel Calls — Batch, Don't Loop

**Mistake**: Early baseline evaluation ran prompts sequentially. Each prompt waited for the previous one to finish.

**Fix**: Switched to batched parallel generation — all prompts in one `model.generate()` call. 10 prompts went from ~50 seconds to ~10 seconds.

**Lesson**: Use batched generation for evaluation. Modal's `@app.cls` pattern keeps the model warm, and batched generation maximizes GPU utilization.

**Pattern**:
```python
# Sequential (slow):
for prompt in prompts:
    response = model.generate(prompt)

# Batched (fast):
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
outputs = model.generate(**inputs)
```

---

## 8. Math-Verify — Use It, Don't Roll Your Own

**Insight**: HuggingFace's `math-verify` library is the standard for math answer verification in RL post-training. It handles LaTeX parsing, numerical equivalence, and symbolic matching.

**Lesson**: Don't write your own answer verifier. Use `math-verify` with a numerical fallback for symbolic mismatches (e.g., `sqrt(2)` vs `2^(1/2)`).

**Pattern**:
```python
from math_verify import parse, verify

def math_reward(model_answer: str, ground_truth: str) -> float:
    try:
        parsed_model = parse(model_answer)
        parsed_gt = parse(ground_truth)
        if parsed_model and parsed_gt:
            return 1.0 if verify(parsed_model, parsed_gt) else 0.0
    except Exception:
        pass
    # Numerical fallback
    try:
        return 1.0 if abs(float(model_answer) - float(ground_truth)) < 1e-6 else 0.0
    except:
        return 0.0
```

---

## 9. RIPO NaN Bug — Fisher Info Clamping

**Mistake**: RIPO (Riemannian Isometric Policy Optimization) loss produced NaNs during training because Fisher information matrix approximation could produce zero or negative values under sqrt.

**Fix**: Clamp Fisher info to a minimum value before sqrt: `fisher_info = fisher_info.clamp(min=1e-8)`.

**Lesson**: When implementing geometric RL methods (RIPO, natural gradient), always clamp the metric tensor to avoid numerical instability.

---

## 10. Putnam-AXIOM Dataset — Know Your Splits

**Mistake**: Initially tried to load Putnam-AXIOM with `val`/`test` splits. The actual dataset uses `full_eval` (522 problems) and `variations` (265 functional variations).

**Fix**: Updated dataset loading to use the correct split names.

**Lesson**: Check the actual dataset structure before writing evaluation code. Don't assume standard `train`/`val`/`test` splits.

---

## 11. Modal GPU Costs — Track Them

**Insight**: Modal bills per second. A 1-hour L40S session costs ~$2, but a forgotten container running for 12 hours costs ~$23.

**Lesson**: Use `scaledown_window=3600` (1 hour) to auto-terminate idle containers. Kill containers explicitly when done.

**Cost tracker**:
| Task | GPU | Time | Cost |
|---|---|---|---|
| Formula probe (3 attempts) | L40S | ~1 hour | ~$2 |
| Baseline eval (2B, wrong model) | T4 | ~30 min | ~$0.30 |
| Baseline eval (9B, correct) | L40S | ~30 min | ~$1 |
| Total session cost | — | ~2 hours | ~$3.30 |

---

## 12. venv and uv — Use the Project's Environment

**Mistake**: Created a new venv inside the working directory instead of using the project's root `.venv`.

**Fix**: Use the existing `.venv` at the project root and `uv pip install` for package management.

**Lesson**: Always check for an existing virtual environment before creating a new one. Use `uv` for faster installs.

---

## Summary: What We'd Do Differently Next Time

| Do | Don't |
|---|---|
| Start with 9B model, not 2B | Pick model by GPU cost |
| Save raw outputs alongside scores | Trust automated scorers blindly |
| Disable thinking for structured output | Let thinking mode fill all tokens |
| Enable thinking for reasoning eval | Use terse output for baseline scores |
| Batch prompts, don't loop | Run sequential generation |
| Use `math-verify` library | Roll your own answer verifier |
| Clamp Fisher info in RIPO | Let NaNs propagate |
| Check dataset splits first | Assume standard splits |
| Use project's `.venv` | Create new venvs |
| Use `uv pip install` | Use plain `pip` |
| Set `scaledown_window` on Modal | Let containers run forever |
| Keep formula sheet as convenience | Remove tools the model "knows" |

---

## Pipeline Status

| Component | Status |
|---|---|
| Agentic Calculator | ✅ 55/55 tests pass |
| Formula Probe | ✅ 17/17 formulas correct |
| RL Training (GRPO + RIPO + Score Centering) | ✅ All tests pass |
| Benchmark Harness (20+ datasets) | ✅ All tests pass |
| Modal Integration | ✅ Verified |
| Baseline Scores (9B) | ✅ Saved |
| Orchestration Script | ✅ Ready |

## Next Steps (When You Return)

1. **RL training** on Modal A100 80GB (~$2.50/hr) with Qwen3.5-9B
2. **Benchmark** on 20+ datasets to measure transfer
3. **Looped transformer** inference from scratch (original plan)

## Cross-References

- Modal infrastructure lessons: `code/modal-lessons.md`
- Agentic calculator docs: `code/agentic-calculator/README.md`
- Pipeline docs: `code/putnam-axiom-pipeline/README.md`
- Baseline scores: `code/putnam-axiom-pipeline/baseline_scores.md`
- Formula probe results: `code/putnam-axiom-pipeline/probe_results.json`
