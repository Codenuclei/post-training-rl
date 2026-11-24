# Modal Lessons Learned

> Derived from setting up Qwen3.5-9B chat on Modal (November 24, 2026).
> These mistakes cost ~30 minutes of debugging and ~$2 in GPU time. Don't repeat them.
>
> **See also**: `code/session-lessons.md` — full session lessons covering model selection, thinking mode bugs, formula probing, agentic calculator design, baseline evaluation, and RL training (12 lessons total).

---

## 1. GPU Selection — Match VRAM to Model, Not Price

**Mistake**: Picked T4 ($0.59/hr) as the default probe GPU.

**Reality**: T4 has 16GB VRAM. Qwen3.5-9B in BF16 needs ~18GB. Inkling 2 Mini needs ~20GB. Models literally don't fit.

**Lesson**: Match GPU VRAM to model size first, cost second.

**Rule of thumb**: `model_params × 2 bytes (BF16) × 1.5 (overhead)` = minimum VRAM.

| Model | Min VRAM | Practical Modal GPU |
|---|---|---|
| Qwen3.5-0.8B | ~1.2 GB | T4 (16GB) |
| Qwen3.5-4B | ~12 GB | L4 (24GB) |
| Qwen3.5-9B | ~27 GB | L40S (48GB) |
| Inkling 2 Mini | ~30 GB | L40S (48GB) |
| GRPO training (9B + LoRA) | ~50 GB | A100 80GB |

## 2. `@app.function` vs `@app.cls` — The 90-Second Tax

**Mistake**: Used `@app.function` for chat. Every prompt call reloaded the model from scratch inside the function body, even though the container stayed warm.

**Reality**: Paid ~90s cold start **per prompt**. A 10-question session took ~15 minutes of loading, ~50 seconds of actual generation.

**Lesson**: For any stateful workload (model in memory, database connection, file handle), use `@app.cls` with `@modal.enter()` to initialize once and `@modal.method()` for repeated calls.

**Pattern**:
```python
@app.cls(gpu="L40S", scaledown_window=3600)
class MyModel:
    @modal.enter()
    def load(self):
        self.model = ...  # loads once

    @modal.method()
    def predict(self, x):
        return self.model(x)  # model already in VRAM
```

**Timeline comparison**:

| Prompt | `@app.function` | `@app.cls` |
|---|---|---|
| 1st | ~90s (cold start + generate) | ~90s (cold start + generate) |
| 2nd | ~90s (reload + generate) | ~5s (generate only) |
| 3rd+ | ~90s each | ~5s each |

## 3. CLI Flag Naming — Modal's Click Quirk

**Mistake**: Went through `chat` → `use_chat` → `chat_` → `interactive` trying to get `--chat` to work.

**Reality**: Modal uses Click for CLI. Underscores become dashes: `use_chat` → `--use-chat`, `chat_` → `--chat-` (trailing dash, broken).

**Lesson**: Name Modal entry point parameters for the CLI flag you want. `interactive` → `--interactive`, `gpu_type` → `--gpu-type`. Avoid trailing underscores — Click doesn't handle them cleanly.

## 4. Model Class Selection — Text-Only on a Multimodal Model

**Mistake**: Used `AutoModelForVision2Seq` for Qwen3.5-9B, which is a multimodal model.

**Reality**: For text-only chat, `AutoModelForCausalLM` is correct. The vision class adds unnecessary image processing overhead and warnings.

**Lesson**: When using a multimodal model for text-only tasks, explicitly choose the causal LM class. Check the model card first.

## 5. Transformers Version — PyPI Lags Behind

**Mistake**: Installed `transformers>=4.48` from PyPI.

**Reality**: Qwen3.5 is new (March 2026). PyPI transformers may not have the latest model support. Missing optimized kernels (`causal_conv1d`, `flash-linear-attention`) fell back to slow PyTorch references.

**Lesson**: For cutting-edge models, install transformers from git HEAD:
```python
.pip_install("git+https://github.com/huggingface/transformers.git")
```
Don't forget to `apt_install("git")` first — the image build failed without it.

## 6. Billing — Modal Requires Payment Before GPU

**Mistake**: Hit `InvalidError: Please add a payment method to use L40S GPU functions` on first run.

**Lesson**: Modal requires a credit card on file before provisioning any GPU, even for short tests. Set this up first. Cost is per-second, so a 1-hour L40S session is ~$2.

## 7. Image Build — Include All Dependencies

**Mistake**: Forgot `git` in the image, needed for `git+https` transformers install.

**Lesson**: When your image build chain has dependencies (git → pip install from git), include all tools in the image definition. Test the image build separately before running the function.

---

## Quick Reference: Modal GPU Pricing (November 2026)

| GPU | VRAM | $/hr | Use Case |
|---|---|---|---|
| T4 | 16 GB | $0.59 | Sub-4B models only |
| L4 | 24 GB | $0.80 | 4-9B models (quantized) |
| L40S | 48 GB | $1.95 | 9B models (BF16), inference sweet spot |
| A100 80GB | 80 GB | $2.50 | RL training with GRPO/RIPO, LoRA fine-tuning |
| H100 | 80 GB | $3.95 | FP8 training, multi-GPU weight sync |
| B200 | 192 GB | $6.25 | Frontier scale, 100B+ models |

## Quick Reference: Modal Commands

```bash
modal token new                          # authenticate (one-time)
modal run script.py                      # run a function
modal run script.py --interactive        # run with CLI flag
modal shell --gpu L40S                   # interactive GPU shell
modal serve script.py                    # hot-reload web endpoint
modal app list                           # list deployed apps
```

## Do / Don't Checklist

| Do | Don't |
|---|---|
| Pick GPU by VRAM needs, not price | Start with T4 |
| Use `@app.cls` + `@modal.enter()` for stateful work | Use `@app.function` for model serving |
| Name CLI params for the flag you want | Use underscores that break in Click |
| Install transformers from git HEAD for new models | Rely on PyPI for latest model support |
| Include `git` in image if installing from git | Assume system tools are present |
| Set up billing before first GPU run | Discover billing requirement at runtime |
| Use `AutoModelForCausalLM` for text-only | Use vision class for text tasks |
