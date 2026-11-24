"""
Baseline Evaluation — Qwen3.5-9B on Key Benchmarks (Async + Thinking Mode)

Runs the base model with THINKING ENABLED and PARALLEL batch generation.
Shows FULL model outputs so you can judge reasoning quality.

Usage:
    modal run code/putnam-axiom-pipeline/baseline_eval.py::main --sample-size 10
"""

import modal
import json
import re

app = modal.App("baseline-eval-async")
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.48",
        "accelerate>=1.0",
        "sentencepiece>=0.2",
        "datasets>=2.14",
        "math-verify>=0.2",
    )
)


def extract_answer(response: str) -> str:
    """Extract the final answer from a model response.
    
    Searches the ENTIRE response for the answer.
    """
    # Search entire response for answer patterns
    patterns = [
        r"####\s*(.+?)$",
        r"\\boxed\{([^}]*)\}",
        r"answer\s*[:=]\s*(.+?)$",
        r"is\\s*[:=]?\\s*([^\.]+)\\.$",
        r"=\\s*([0-9]+(?:\\.[0-9]+)?)\\s*$",
        r"\\$\\$\\s*([0-9]+(?:\\.[0-9]+)?)\\s*\\$\\$",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    # Fallback: try to find a number in the last few lines
    lines = response.strip().split("\n")
    for line in reversed(lines[-5:]):
        nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', line)
        if nums:
            return nums[-1]
    return lines[-1].strip() if lines else response


def verify_answer(model_answer: str, ground_truth: str) -> bool:
    """Verify using math-verify."""
    try:
        from math_verify import parse, verify
        parsed_model = parse(model_answer)
        parsed_gt = parse(ground_truth)
        if parsed_model and parsed_gt:
            return verify(parsed_model, parsed_gt)
    except Exception:
        pass
    return model_answer.strip() == ground_truth.strip()


@app.cls(
    gpu="L40S",
    image=image,
    scaledown_window=3600,
    timeout=3600,
)
class BaselineEvaluator:
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
    def evaluate_batch(self, prompts: list, batch_size: int = 5) -> list:
        """Evaluate a batch of prompts in PARALLEL using batched generation.
        
        Thinking mode is ENABLED so you see the full reasoning process.
        """
        import torch
        results = []
        
        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            
            # Tokenize batch
            messages_batch = [[{"role": "user", "content": p}] for p in batch_prompts]
            texts = [
                self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=True,  # Thinking ON for Qwen3.5 reasoning
                )
                for msgs in messages_batch
            ]
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
            
            # Generate in parallel
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,  # Thinking mode needs more tokens
                    temperature=0.0,
                    do_sample=False,
                )
            
            # Decode each response in the batch
            for i, output in enumerate(outputs):
                input_len = inputs["input_ids"][i].shape[0]
                generated = output[input_len:]
                response = self.tokenizer.decode(generated, skip_special_tokens=True)
                results.append(response)
        
        return results


def load_gsm8k(num_samples: int = 50):
    """Load GSM8K test set."""
    from datasets import load_dataset
    import re
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(num_samples))
    questions = [item['question'] for item in ds]
    answers = []
    for item in ds:
        match = re.search(r"####\s*(.+)", item['answer'])
        answers.append(match.group(1).strip() if match else item['answer'])
    return questions, answers


def load_putnam(num_samples: int = 10):
    """Load Putnam-AXIOM dataset. Answer extracted from \\boxed{} in solution."""
    from datasets import load_dataset
    import re
    ds = load_dataset("Putnam-AXIOM/putnam-axiom-dataset-ICML-2025-522", split="full_eval")
    ds = ds.select(range(min(num_samples, len(ds))))
    questions = [item['problem'] for item in ds]
    answers = []
    for item in ds:
        # Extract answer from \\boxed{...} in solution
        boxed = re.findall(r'\\boxed\{([^}]*)\}', item['solution'])
        answers.append(boxed[-1] if boxed else item['solution'][:100])
    return questions, answers


def load_math500(num_samples: int = 10):
    """Load MATH dataset."""
    from datasets import load_dataset
    import re
    ds = load_dataset("HuggingFaceH4/MATH", split="test")
    ds = ds.select(range(min(num_samples, len(ds))))
    questions = [item['problem'] for item in ds]
    answers = []
    for item in ds:
        boxed = re.findall(r'\\boxed\{([^}]*)\}', item['solution'])
        answers.append(boxed[-1] if boxed else item['solution'][:100])
    return questions, answers


def load_aime(num_samples: int = 10):
    """Load AIME dataset from di-zhang-fdu."""
    from datasets import load_dataset
    ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
    # Sample recent years
    ds = ds.select(range(min(num_samples, len(ds))))
    questions = [item['Question'] for item in ds]
    answers = [str(item['Answer']) for item in ds]
    return questions, answers


@app.function(image=image, timeout=7200, gpu="L40S")
def main(sample_size: int = 10):
    """Run baseline evaluation with async parallel calls and thinking mode."""
    print("=" * 70)
    print("BASELINE EVALUATION — Qwen3.5-9B (Thinking ON, Parallel)")
    print("=" * 70)
    print(f"\nSample size per benchmark: {sample_size}")
    print("GPU: L40S ($1.95/hr)")
    print("Thinking mode: ENABLED")
    print("Generation: PARALLEL batched")
    print()

    evaluator = BaselineEvaluator()
    all_results = {}

    benchmarks = [
        ("GSM8K", lambda: load_gsm8k(sample_size)),
        ("Putnam-AXIOM", lambda: load_putnam(sample_size)),
        ("MATH-500", lambda: load_math500(sample_size)),
        ("AIME", lambda: load_aime(sample_size)),
    ]

    for name, loader in benchmarks:
        print(f"\n{'=' * 70}")
        print(f"BENCHMARK: {name}")
        print(f"{'=' * 70}")

        try:
            questions, answers = loader()
            print(f"Loaded {len(questions)} questions")

            # Build prompts — force #### <answer> format
            prompts = [
                f"""Solve this math problem. Think step by step, then give the final answer.

Problem: {q}

End your response with the final answer in this exact format: #### <answer>

For example: #### 42

Solution:"""
                for q in questions
            ]

            # ASYNC parallel evaluation — all prompts in one batched call
            print(f"\nGenerating responses in parallel (batch size 3)...")
            responses = evaluator.evaluate_batch.remote(prompts, batch_size=3)

            # Score and display
            results = []
            correct = 0
            for i, (q, a, resp) in enumerate(zip(questions, answers, responses)):
                model_answer = extract_answer(resp)
                is_correct = verify_answer(model_answer, a)
                if is_correct:
                    correct += 1

                results.append({
                    "question": q[:200],
                    "ground_truth": a[:100],
                    "model_answer": model_answer[:100],
                    "full_response": resp[:1000],  # Full response for inspection
                    "correct": is_correct,
                })

            total = len(results)
            accuracy = correct / total if total > 0 else 0

            all_results[name] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
                "accuracy_pct": f"{accuracy:.1%}",
                "samples": results[:5],  # Show first 5 full samples
            }

            print(f"\n✓ {name}: {correct}/{total} = {accuracy:.1%}")

            # Show individual results with FULL responses
            print(f"\n--- Sample Outputs ({name}) ---")
            for i, r in enumerate(results[:5]):
                status = "✓ CORRECT" if r["correct"] else "✗ WRONG"
                print(f"\n[{i+1}] {status}")
                print(f"  Q: {r['question'][:150]}...")
                print(f"  GT: {r['ground_truth'][:80]}")
                print(f"  Model answer: {r['model_answer'][:80]}")
                print(f"  Full response: {r['full_response'][:400]}...")

        except Exception as e:
            import traceback
            print(f"✗ {name} failed: {e}")
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

    # Final summary
    print("\n" + "=" * 70)
    print("BASELINE SCORES SUMMARY")
    print("=" * 70)
    for name, result in all_results.items():
        if "error" in result:
            print(f"  ✗ {name}: ERROR — {result['error']}")
        else:
            print(f"  ✓ {name}: {result['accuracy_pct']} ({result['correct']}/{result['total']})")

    print("\n" + "=" * 70)
    print("JSON_RESULTS_START")
    print(json.dumps(all_results, indent=2))
    print("JSON_RESULTS_END")

    return all_results
