# Agentic Calculator v2

Fail-safe tool-use harness for small models doing PhD-level math.

## Why Tools Amplify Capability

A 2B model can reason about **what** to compute but not **compute** reliably. With a calculator + formula sheet, it focuses on reasoning structure while offloading arithmetic to verified tools.

**Without tools** (2B model fails):
```
Q: "What is 7^100 mod 13?"
Model: "Let me think... 7^2 = 49, 49 mod 13 = 10... [gets lost]"
→ Wrong answer
```

**With tools** (2B model succeeds):
```
Q: "What is 7^100 mod 13?"
Model: [queries formula_sheet] → Fermat's Little Theorem
Model: "7^12 ≡ 1 (mod 13), so 7^100 ≡ 7^4 (mod 13)"
Model: [calls calculator] pow(7, 4, 13)
Calculator: "9"
Model: "Therefore, 7^100 mod 13 = 9."
→ Correct answer
```

## Three Tools

| Tool | Purpose | Example |
|------|---------|---------|
| `calculator` | Safe Python eval with math/sympy | `pow(7, 4, 13)` → `9` |
| `formula_sheet` | Query formulas by topic | `number_theory` → Fermat, Euler, CRT, etc. |
| `verify` | Check answer vs expected | `9, 9` → ✓ Correct |

## Formula Sheet Topics

- **number_theory**: Fermat's Little Theorem, Euler's Theorem, CRT, Wilson's, Legendre Symbol, Quadratic Reciprocity
- **combinatorics**: Binomial Coefficient, Pascal's Identity, Stars and Bars, Inclusion-Exclusion, Catalan Numbers
- **algebra**: Quadratic Formula, Vieta's, AM-GM, Cauchy-Schwarz, Binomial Theorem, Matrix ops
- **calculus**: Power/Product/Quotient/Chain rules, Integration by Parts, Taylor Series, L'Hôpital's
- **geometry**: Distance, Midpoint, Slope, Area, Pythagorean, Law of Cosines/Sines
- **sequences_series**: Arithmetic/Geometric sequences and series, Telescoping
- **probability**: Conditional, Bayes', Total Probability, Expected Value, Variance, Binomial
- **linear_algebra**: Dot/Cross product, Eigenvalues, Characteristic Polynomial, Trace, Determinant

## Safety Features

1. **No crashes** — every tool call returns a result or clear error message
2. **Timeout protection** — 10-second limit on each computation
3. **Code injection prevention** — rejects `__`, `import`, `exec`, `eval`, `os.`, `sys.`, etc.
4. **Max turns limit** — prevents infinite loops (default: 10 turns)
5. **Graceful degradation** — if model crashes, returns error without crashing the harness

## Usage

### Local Testing

```bash
# Run all 55 edge case tests + 4 agentic loop tests
cd hands-on-modern-rl
.venv/bin/python3 code/agentic-calculator/calculator.py
```

### Modal (Cloud GPU)

```bash
# Quick test (one question):
modal run code/agentic-calculator/modal_calc.py

# Interactive chat with calculator:
modal run code/agentic-calculator/modal_calc.py --interactive
```

Cost: ~$0.59/hr on T4 GPU (Qwen3.5-2B, 16GB VRAM).

### Python API

```python
from code.agentic-calculator.calculator import AgenticCalculator, safe_eval, query_formula_sheet

# Direct calculator use
result = safe_eval("pow(7, 100, 13)")  # "9"

# Formula sheet query
formulas = query_formula_sheet("number_theory")  # Fermat, Euler, etc.

# Agentic loop with your model
def my_model(messages):
    # ... your model generation logic ...
    return response

agent = AgenticCalculator(my_model, max_turns=5)
result = agent.run("What is 7^100 mod 13?")
# {
#   "final_answer": "Therefore, 7^100 mod 13 = 9.",
#   "tool_calls": [("formula_sheet", "number_theory", "..."), ("calculator", "pow(7, 4, 13)", "9")],
#   "total_turns": 2,
#   "success": True,
#   "error": None,
# }
```

## Test Coverage (55 tests, all passing)

- **Basic arithmetic**: +, -, *, /, //, %, **, negatives, floats
- **Modular arithmetic**: pow(a, b, m), large exponents, negative mod
- **Math module**: factorial, gcd, lcm, sqrt, pi, e, sin, cos, log, exp
- **Sympy**: integrate, diff, solve, simplify, expand, limit, Matrix, factorial
- **Multi-statement**: variable assignment, multiple assignments, list comprehensions
- **Error handling**: division by zero, invalid syntax, undefined vars, code injection attempts
- **Formula sheet**: all 8 topics, unknown topic handling, fuzzy match
- **Tool parsing**: valid calls, empty code blocks, no tools in text
- **Tool execution**: calculator, unknown tool, formula sheet
- **Verification**: correct answer, incorrect answer, numeric match

## Design Philosophy

For PhD-level math (Putnam, AIME, etc.), the pattern is **open-book exams**: formulas are available, the test is reasoning. The harness implements this pattern:

1. Model queries formula sheet for relevant formulas
2. Model reasons about which formula applies
3. Model uses calculator to compute the result
4. Model states the final answer

The model's job is **matching problem type → formula → computation**, not memorizing formulas or doing arithmetic in its head.

## Next Steps

- [ ] Putnam-AXIOM dataset analysis → identify question types
- [ ] Generate formula sheet JSON from Qwen for each question type
- [ ] RL post-training with tool use on Putnam-AXIOM
- [ ] Benchmark on AIME, LiveCodeBench, etc.
