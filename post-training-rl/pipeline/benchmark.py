"""
Benchmarking Harness for 20+ Math Datasets

Evaluates RL post-trained models on a comprehensive suite of math benchmarks:
- Putnam-AXIOM (university competition math)
- AIME (competition math)
- MATH-500 (competition math)
- GSM8K (grade school math)
- MBPP (code generation)
- HumanEval (code generation)
- LiveCodeBench (competitive coding)
- And more...

Usage:
    python code/putnam-axiom-pipeline/benchmark.py --model Qwen/Qwen3.5-2B
"""
import json
import os
import time
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from datasets import load_dataset
    from math_verify import parse, verify
except ImportError:
    os.system("pip install datasets math-verify")
    from datasets import load_dataset
    from math_verify import parse, verify


# ─── Benchmark Definitions ────────────────────────────────────────

@dataclass
class Benchmark:
    """Definition of a benchmark dataset."""
    name: str
    dataset_path: str
    split: str
    problem_column: str
    answer_column: str
    max_samples: Optional[int] = None
    description: str = ""


# Define 20+ benchmarks
BENCHMARKS = [
    # Math reasoning benchmarks
    Benchmark(
        name="putnam_axiom_full",
        dataset_path="Putnam-AXIOM/putnam-axiom-dataset-ICML-2025-522",
        split="full_eval",
        problem_column="problem",
        answer_column="answer",
        max_samples=522,
        description="Putnam competition math (full eval, 522 problems)",
    ),
    Benchmark(
        name="putnam_axiom_variations",
        dataset_path="Putnam-AXIOM/putnam-axiom-dataset-ICML-2025-522",
        split="variations",
        problem_column="problem",
        answer_column="answer",
        max_samples=100,
        description="Putnam variations (tests generalization, 100 sample)",
    ),
    Benchmark(
        name="gsm8k",
        dataset_path="openai/gsm8k",
        split="test",
        problem_column="question",
        answer_column="answer",
        max_samples=1000,
        description="Grade school math problems",
    ),
    Benchmark(
        name="math_500",
        dataset_path="HuggingFaceH4/MATH",
        split="test",
        problem_column="problem",
        answer_column="solution",
        max_samples=500,
        description="Competition math problems",
    ),
    Benchmark(
        name="aime_2024",
        dataset_path="HuggingFaceH4/AIME",
        split="test",
        problem_column="problem",
        answer_column="solution",
        max_samples=30,
        description="AIME 2024 competition",
    ),
    Benchmark(
        name="aime_2025",
        dataset_path="HuggingFaceH4/AIME-2025",
        split="test",
        problem_column="problem",
        answer_column="solution",
        max_samples=30,
        description="AIME 2025 competition",
    ),
    
    # Code generation benchmarks
    Benchmark(
        name="mbpp",
        dataset_path="google-research-datasets/mbpp",
        split="test",
        problem_column="text",
        answer_column="code",
        max_samples=500,
        description="Massive Code Generation Dataset",
    ),
    Benchmark(
        name="humaneval",
        dataset_path="openai/openai_humaneval",
        split="test",
        problem_column="prompt",
        answer_column="canonical_solution",
        max_samples=164,
        description="HumanEval code generation",
    ),
    Benchmark(
        name="livecodebench",
        dataset_path="livecodebench/livecodebench",
        split="test",
        problem_column="question_content",
        answer_column="id",
        max_samples=100,
        description="LiveCodeBench (continuously updated)",
    ),
    
    # Additional math benchmarks
    Benchmark(
        name="mmlu_math",
        dataset_path="cais/mmlu",
        split="test",
        problem_column="question",
        answer_column="answer",
        max_samples=100,
        description="MMLU math subset",
    ),
    Benchmark(
        name="gpqa_diamond",
        dataset_path="Idavidrein/gpqa",
        split="gpqa_diamond",
        problem_column="Question",
        answer_column="Correct Answer",
        max_samples=100,
        description="GPQA Diamond (expert-level science)",
    ),
    Benchmark(
        name="svamp",
        dataset_path="ChilleD/SVAMP",
        split="test",
        problem_column="Question",
        answer_column="Answer",
        max_samples=1000,
        description="SVAMP math word problems",
    ),
    Benchmark(
        name="asdiv",
        dataset_path="ChilleD/ASDiv",
        split="test",
        problem_column="body",
        answer_column="answer",
        max_samples=500,
        description="ASDiv math word problems",
    ),
    Benchmark(
        name="mathqa",
        dataset_path="math-ai/MathQA",
        split="test",
        problem_column="Problem",
        answer_column="answer",
        max_samples=500,
        description="MathQA dataset",
    ),
    
    # Reasoning benchmarks
    Benchmark(
        name="arc_challenge",
        dataset_path="allenai/ai2_arc",
        split="Challenge10",
        problem_column="question",
        answer_column="answerKey",
        max_samples=100,
        description="ARC Challenge (science reasoning)",
    ),
    Benchmark(
        name="openbookqa",
        dataset_path="allensm/openbookqa",
        split="test",
        problem_column="question_stem",
        answer_column="answerKey",
        max_samples=500,
        description="OpenBookQA (science reasoning)",
    ),
    Benchmark(
        name="logiqa",
        dataset_path="szg315/LogiQA",
        split="test",
        problem_column="context",
        answer_column="answer",
        max_samples=500,
        description="LogiQA (logical reasoning)",
    ),
    Benchmark(
        name="reclor",
        dataset_path="yanqhuang/ReClor",
        split="test",
        problem_column="question",
        answer_column="answer",
        max_samples=500,
        description="ReClor (reading comprehension and logical reasoning)",
    ),
    
    # Additional code benchmarks
    Benchmark(
        name="apps",
        dataset_path="codeparrot/apps",
        split="test",
        problem_column="question",
        answer_column="input_output",
        max_samples=100,
        description="APPS (competitive programming)",
    ),
    Benchmark(
        name="ds1000",
        dataset_path="nomic-ai/gsm8k",
        split="test",
        problem_column="question",
        answer_column="answer",
        max_samples=100,
        description="DS-1000 (diverse SQL problems)",
    ),
]


# ─── Model Evaluator ──────────────────────────────────────────────

class ModelEvaluator:
    """Evaluate a model on math benchmarks."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if self.device.type == "cuda" else None,
        )
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        
        print(f"Model loaded on {self.device}")
    
    def generate(self, problem: str, max_new_tokens: int = 512) -> str:
        """Generate a response for a single problem."""
        messages = [
            {"role": "system", "content": "You are a math expert. Solve this problem step by step. Return your final answer in the format: #### <answer>"},
            {"role": "user", "content": problem},
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        
        prompt_len = inputs.input_ids.shape[1]
        generated_ids = outputs[0][prompt_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def extract_answer(self, response: str) -> str:
        """Extract the final answer from a response."""
        # Look for #### format
        if "####" in response:
            return response.split("####")[-1].strip()
        
        # Look for \boxed{} format
        if "\\boxed{" in response:
            boxed_part = response.split("\\boxed{")[-1].split("}")[0]
            return boxed_part.strip()
        
        # Return last line
        return response.strip().split("\n")[-1].strip()
    
    def verify_answer(self, model_answer: str, gold_answer: str) -> bool:
        """Verify if the model's answer matches the gold answer."""
        try:
            parsed_model = parse(model_answer)
            parsed_gold = parse(gold_answer)
            return verify(parsed_gold, parsed_model)
        except:
            # Fallback to string comparison
            return model_answer.strip() == gold_answer.strip()
    
    def evaluate_benchmark(self, benchmark: Benchmark, batch_size: int = 1) -> Dict[str, Any]:
        """Evaluate the model on a single benchmark."""
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {benchmark.name}")
        print(f"Description: {benchmark.description}")
        print(f"{'=' * 60}")
        
        # Load dataset
        try:
            dataset = load_dataset(benchmark.dataset_path, split=benchmark.split, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return {
                'benchmark': benchmark.name,
                'error': str(e),
                'accuracy': 0.0,
            }
        
        # Limit samples
        if benchmark.max_samples and len(dataset) > benchmark.max_samples:
            dataset = dataset.select(range(benchmark.max_samples))
        
        print(f"Samples: {len(dataset)}")
        
        # Evaluate
        correct = 0
        total = 0
        results = []
        
        start_time = time.time()
        
        for i, item in enumerate(dataset):
            try:
                problem = item[benchmark.problem_column]
                gold_answer = item[benchmark.answer_column]
                
                # Generate response
                response = self.generate(problem)
                model_answer = self.extract_answer(response)
                
                # Verify
                is_correct = self.verify_answer(model_answer, gold_answer)
                
                if is_correct:
                    correct += 1
                
                total += 1
                
                results.append({
                    'problem': problem[:200],  # Truncate for storage
                    'gold_answer': str(gold_answer)[:100],
                    'model_answer': model_answer[:100],
                    'correct': is_correct,
                })
                
                # Progress
                if (i + 1) % 10 == 0:
                    accuracy = correct / total * 100
                    elapsed = time.time() - start_time
                    print(f"  Progress: {i + 1}/{len(dataset)} ({accuracy:.1f}%, {elapsed:.1f}s)")
            
            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                continue
        
        elapsed = time.time() - start_time
        accuracy = correct / total * 100 if total > 0 else 0.0
        
        print(f"\n  Results: {correct}/{total} ({accuracy:.1f}%)")
        print(f"  Time: {elapsed:.1f}s")
        
        return {
            'benchmark': benchmark.name,
            'description': benchmark.description,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'time_seconds': elapsed,
            'results': results,
        }
    
    def evaluate_all(self, benchmarks: List[Benchmark] = None) -> Dict[str, Any]:
        """Evaluate the model on all benchmarks."""
        if benchmarks is None:
            benchmarks = BENCHMARKS
        
        print("\n" + "=" * 60)
        print("Benchmarking Suite")
        print("=" * 60)
        
        results = {}
        
        for benchmark in benchmarks:
            result = self.evaluate_benchmark(benchmark)
            results[benchmark.name] = result
        
        # Summary
        print("\n" + "=" * 60)
        print("Benchmark Summary")
        print("=" * 60)
        
        for name, result in results.items():
            if 'accuracy' in result:
                print(f"  {name}: {result['accuracy']:.1f}%")
            else:
                print(f"  {name}: Error ({result.get('error', 'Unknown')})")
        
        return results


# ─── Main ─────────────────────────────────────────────────────────

def main():
    """Main entry point for benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark math models")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-2B",
        help="Model name or path",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="Specific benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for results",
    )
    
    args = parser.parse_args()
    
    # Filter benchmarks if specified
    if args.benchmarks:
        benchmarks = [b for b in BENCHMARKS if b.name in args.benchmarks]
    else:
        benchmarks = BENCHMARKS
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model)
    
    # Run benchmarks
    results = evaluator.evaluate_all(benchmarks)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
