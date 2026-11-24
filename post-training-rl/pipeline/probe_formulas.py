"""
Probe Qwen3.5-9B to see if it knows Putnam-level formulas.

If the model already knows the formulas, we skip the formula sheet entirely.
If it doesn't, we keep the sheet as a tool-use option.

This asks specific formula questions (not "do you know X" but "what is the formula for Y")
across all Putnam topics identified in the dataset analysis.

Usage:
    modal run code/putnam-axiom-pipeline/probe_formulas.py
"""

import modal
import json
import time

app = modal.App("putnam-formula-probe")
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.48",
        "accelerate>=1.0",
        "sentencepiece>=0.2",
    )
)

# Formula questions across Putnam topics — specific enough to test real knowledge
FORMULA_QUESTIONS = [
    # Number Theory
    {"topic": "number_theory", "question": "What is Euler's totient theorem formula? State it precisely with the congruence relation."},
    {"topic": "number_theory", "question": "What is the formula for the number of solutions to x^2 ≡ a (mod p) where p is an odd prime? Give the Legendre symbol expression."},
    
    # Combinatorics
    {"topic": "combinatorics", "question": "What is the inclusion-exclusion principle formula for |A₁ ∪ A₂ ∪ ... ∪ Aₙ|? Write the full sum with binomial coefficients."},
    {"topic": "combinatorics", "question": "What is the formula for the nth Catalan number Cₙ? Give both the binomial coefficient form and the recursive form."},
    
    # Algebra
    {"topic": "algebra", "question": "What is the determinant formula for a 3×3 matrix using the Leibniz formula (sum over permutations)? State it with the sign function."},
    {"topic": "algebra", "question": "What is the characteristic polynomial of a matrix A? How does Cayley-Hamilton theorem relate it to the matrix itself?"},
    
    # Calculus / Real Analysis
    {"topic": "calculus", "question": "What is Taylor's theorem with the Lagrange remainder term? Write the full expansion up to order n with the remainder Rₙ."},
    {"topic": "real_analysis", "question": "State the Cauchy-Schwarz inequality for integrals: ∫fg ≤ ? Write the precise inequality."},
    {"topic": "real_analysis", "question": "What is the statement of the Arzela-Ascoli theorem? What conditions make a subset of C[X] relatively compact?"},
    
    # Complex Analysis
    {"topic": "complex_analysis", "question": "Write Cauchy's integral formula for f⁽ⁿ⁾(a) — the nth derivative. Include the factorial and power terms."},
    {"topic": "complex_analysis", "question": "State the residue theorem for a closed contour integral. Write the sum over residues with 2πi."},
    
    # Differential Equations
    {"topic": "differential_equations", "question": "What is the integrating factor method for solving y' + P(x)y = Q(x)? Write the formula for the integrating factor μ(x) and the solution."},
    {"topic": "differential_equations", "question": "What is the general solution form for a second-order linear ODE ay'' + by' + cy = 0 when the characteristic equation has complex roots r = α ± βi?"},
    
    # Functional Equations
    {"topic": "functional_equations", "question": "What is the Cauchy functional equation f(x+y) = f(x) + f(y)? What is the general solution when f is assumed continuous?"},
    
    # Trigonometry
    {"topic": "trigonometry", "question": "What is the formula for the sum of sines: sin(x) + sin(2x) + ... + sin(nx)? Write the closed form."},
    
    # Sequences and Series
    {"topic": "sequences_series", "question": "What is the ratio test for convergence of ∑aₙ? State the condition on lim |aₙ₊₁/aₙ| for convergence and divergence."},
    
    # Linear Algebra
    {"topic": "linear_algebra", "question": "State the rank-nullity theorem for a linear transformation T: V → W. Write the equation relating rank and nullity."},
]

# Expected keywords in correct answers (for automated scoring)
EXPECTED_KEYWORDS = {
    "number_theory_0": ["φ", "phi", "totient", "≡", "congruent", "gcd"],
    "number_theory_1": ["Legendre", "symbol", "quadratic", "residue"],
    "combinatorics_0": ["inclusion", "exclusion", "∪", "union", "(-1)"],
    "combinatorics_1": ["Catalan", "binomial", "4n", "n+1", "recursive"],
    "algebra_0": ["permutation", "sign", "det", "determinant", "Leibniz"],
    "algebra_1": ["characteristic", "Cayley-Hamilton", "det", "p(A) = 0"],
    "calculus_0": ["Taylor", "remainder", "Lagrange", "n!", "derivative"],
    "real_analysis_0": ["Cauchy-Schwarz", "∫", "integral", "≤", "norm"],
    "real_analysis_1": ["Arzela-Ascoli", "equicontinuous", "compact", "bounded"],
    "complex_analysis_0": ["Cauchy", "integral", "n!", "derivative", "2πi"],
    "complex_analysis_1": ["residue", "2πi", "sum", "contour", "pole"],
    "differential_equations_0": ["integrating", "factor", "exp", "e^", "∫P"],
    "differential_equations_1": ["e^{αx}", "cos(βx)", "sin(βx)", "complex", "roots"],
    "functional_equations_0": ["Cauchy", "f(x) = cx", "linear", "continuous"],
    "trigonometry_0": ["sin", "sum", "closed", "n", "product"],
    "sequences_series_0": ["ratio", "limit", "converge", "diverge", "< 1"],
    "linear_algebra_0": ["rank", "nullity", "dimension", "dim", "kernel"],
}


@app.cls(
    gpu="L40S",
    image=image,
    scaledown_window=3600,
    timeout=600,
)
class FormulaProbe:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("Loading Qwen3.5-9B...")
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3.5-9B",
            torch_dtype="auto",
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
        print("✓ Model loaded")
    
    @modal.method()
    def ask_formula(self, topic: str, question: str) -> str:
        """Ask the model a specific formula question.
        
        Qwen3.5 has a built-in thinking mode that must be disabled.
        We set enable_thinking=False in the chat template to get direct answers.
        """
        prompt = f"""State the following formula precisely using LaTeX. Do not explain, do not reason, just write the formula.

Topic: {topic}
Question: {question}

Formula:"""
        
        messages = [{"role": "user", "content": prompt}]
        # CRITICAL: disable thinking mode so model gives direct answer
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
            )
        
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        # Strip any remaining thinking tags as fallback
        for open_tag, close_tag in [("<thinking>", "</thinking>"), ("<think>", "</think>")]:
            if open_tag in response:
                response = response.split(close_tag)[-1].strip()
        
        return response.strip()


def score_response(response: str, keywords: list) -> tuple:
    """Score a response based on expected keywords. Returns (score, matched_keywords)."""
    response_lower = response.lower()
    matched = [kw for kw in keywords if kw.lower() in response_lower]
    score = len(matched) / len(keywords) if keywords else 0.0
    return score, matched


@app.function(image=image, timeout=900, gpu="L40S")
def main():
    """Run the formula probe and report results.
    
    The FormulaProbe class loads the model in @modal.enter() automatically
    when the first @modal.method() is called. We keep the same container warm
    across all questions by using the class instance throughout.
    """
    import torch
    
    results = []
    topic_scores = {}
    
    print("=" * 70)
    print("PUTNAM FORMULA KNOWLEDGE PROBE — Qwen3.5-9B")
    print("=" * 70)
    
    # Create the class handle — model loads on first method call via @modal.enter()
    probe = FormulaProbe()
    
    for i, item in enumerate(FORMULA_QUESTIONS):
        topic = item["topic"]
        question = item["question"]
        
        # Find the right key index for this topic
        topic_count = sum(1 for q in FORMULA_QUESTIONS[:i] if q["topic"] == topic)
        key = f"{topic}_{topic_count}"
        
        keywords = EXPECTED_KEYWORDS.get(key, [])
        
        print(f"\n[{i+1}/{len(FORMULA_QUESTIONS)}] {topic}: {question[:60]}...")
        
        # First call triggers @modal.enter() to load the model
        # Subsequent calls reuse the loaded model in the same container
        response = probe.ask_formula.remote(topic, question)
        
        score, matched = score_response(response, keywords)
        
        # Show full response (not truncated) so we can verify
        display_response = response[:500] + "..." if len(response) > 500 else response
        
        results.append({
            "topic": topic,
            "question": question,
            "response": response,
            "score": score,
            "matched_keywords": matched,
            "expected_keywords": keywords,
        })
        
        if topic not in topic_scores:
            topic_scores[topic] = []
        topic_scores[topic].append(score)
        
        status = "✓" if score >= 0.5 else "✗"
        print(f"  Score: {score:.0%} ({len(matched)}/{len(keywords)} keywords)")
        print(f"  Response: {display_response}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    overall_scores = [r["score"] for r in results]
    avg_score = sum(overall_scores) / len(overall_scores)
    
    print(f"\nOverall average score: {avg_score:.1%}")
    print(f"Questions passed (≥50% keywords): {sum(1 for s in overall_scores if s >= 0.5)}/{len(overall_scores)}")
    
    print("\nTopic breakdown:")
    for topic, scores in sorted(topic_scores.items()):
        avg = sum(scores) / len(scores)
        status = "✓ KNOWS" if avg >= 0.6 else "⚠ PARTIAL" if avg >= 0.3 else "✗ MISSING"
        print(f"  {status} {topic}: {avg:.1%} ({len(scores)} questions)")
    
    # Decision
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)
    
    missing_topics = [t for t, scores in topic_scores.items() if sum(scores)/len(scores) < 0.3]
    partial_topics = [t for t, scores in topic_scores.items() if 0.3 <= sum(scores)/len(scores) < 0.6]
    
    if not missing_topics and not partial_topics:
        print("\n🎉 SKIP FORMULA SHEET — Model knows all Putnam-level formulas.")
        print("The model can reason about formulas without tool assistance.")
    elif not missing_topics:
        print(f"\n⚠ PARTIAL SHEET — Model partially knows: {', '.join(partial_topics)}")
        print("Consider a lightweight formula sheet for weak topics only.")
    else:
        print(f"\n✗ KEEP FORMULA SHEET — Model missing: {', '.join(missing_topics)}")
        if partial_topics:
            print(f"  Partial knowledge: {', '.join(partial_topics)}")
        print("The formula sheet tool is necessary for reliable performance.")
    
    # Save results
    output = {
        "model": "Qwen3.5-9B",
        "gpu": "L40S",
        "overall_score": avg_score,
        "topic_scores": {t: sum(s)/len(s) for t, s in topic_scores.items()},
        "results": results,
        "decision": "skip_sheet" if not missing_topics and not partial_topics else "keep_sheet",
    }
    
    # Save results to stdout as JSON for capture
    print("\n" + "=" * 70)
    print("JSON_RESULTS_START")
    print(json.dumps(output, indent=2))
    print("JSON_RESULTS_END")
    
    # Also print full responses for verification
    print("\n" + "=" * 70)
    print("FULL RESPONSES (for verification)")
    print("=" * 70)
    for r in results:
        print(f"\n[{r['topic']}]: {r['question'][:80]}...")
        print(f"  Score: {r['score']:.0%}")
        print(f"  Response: {r['response'][:300]}")
    
    return output
