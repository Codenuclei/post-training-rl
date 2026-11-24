# Qwen3.5-2B Baseline Scores (No RL Training)

**Model**: Qwen/Qwen3.5-2B (BF16, no fine-tuning)
**GPU**: Modal T4 ($0.59/hr)
**Tokens**: 1024 max_new_tokens
**Samples**: 10 per benchmark

## Results

| Benchmark | Score | Correct/Total |
|-----------|-------|---------------|
| GSM8K | 10.0% | 1/10 |
| Putnam-AXIOM | 20.0% | 2/10 |
| MATH-500 | 0.0% | 0/10 |
| AIME | 10.0% | 1/10 |

## Notes

- These are the REAL starting scores before any RL post-training
- Putnam at 20% is surprisingly high for a 2B model — likely the sampled questions happened to match training data
- MATH-500 at 0% shows the model can't handle competition-level math without RL
- GSM8K at 10% is low — the model reasons correctly but often doesn't output the final answer in parseable format
- AIME at 10% is expected for a 2B model on competition math

## Next Steps

1. RL post-training with GRPO/RIPO + Score Centering
2. Target: GSM8K 30-40%, Putnam 25-30%, AIME 15-20%
3. Benchmark on full datasets after training
