"""
Orchestration script for the Putnam-AXIOM RL Post-Training Pipeline.

This script runs the entire pipeline end-to-end:
1. Analyze Putnam-AXIOM dataset
2. Generate formula sheet (optional, uses Modal)
3. Run RL training with GRPO/RIPO + Score Centering
4. Benchmark on 20+ datasets

Usage:
    # Full pipeline (requires Modal GPU):
    python code/putnam-axiom-pipeline/run_pipeline.py --full

    # Just analyze dataset:
    python code/putnam-axiom-pipeline/run_pipeline.py --analyze

    # Just train (requires Modal A100):
    python code/putnam-axiom-pipeline/run_pipeline.py --train

    # Just benchmark (requires Modal GPU):
    python code/putnam-axiom-pipeline/run_pipeline.py --benchmark

    # Quick local test (no GPU needed):
    python code/putnam-axiom-pipeline/run_pipeline.py --test
"""
import argparse
import json
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def run_analysis():
    """Step 1: Analyze Putnam-AXIOM dataset."""
    print("=" * 60)
    print("STEP 1: Analyzing Putnam-AXIOM Dataset")
    print("=" * 60)
    
    from analyze_putnam import download_putnam_axiom, analyze_question_types, identify_formulas_needed, save_analysis
    
    dataset = download_putnam_axiom()
    analysis = analyze_question_types(dataset)
    topic_to_formulas = identify_formulas_needed(analysis)
    save_analysis(analysis, 'code/putnam-axiom-pipeline/analysis.json')
    
    with open('code/putnam-axiom-pipeline/topic_to_formulas.json', 'w') as f:
        json.dump(topic_to_formulas, f, indent=2)
    
    print("✓ Analysis complete")
    return analysis

def run_formula_sheet_generation():
    """Step 2: Generate formula sheet using Qwen3.5-9B on Modal."""
    print("=" * 60)
    print("STEP 2: Generating Formula Sheet (Modal)")
    print("=" * 60)
    print("This step requires Modal authentication and GPU access.")
    print("Run: modal run code/putnam-axiom-pipeline/modal_formula_sheet.py")
    return True

def run_training():
    """Step 3: Run RL training with GRPO/RIPO + Score Centering."""
    print("=" * 60)
    print("STEP 3: RL Training (Modal A100 80GB)")
    print("=" * 60)
    print("This step requires Modal authentication and A100 GPU access.")
    print("Run: modal run code/putnam-axiom-pipeline/modal_rl_training.py")
    return True

def run_benchmark(model_path=None):
    """Step 4: Benchmark on 20+ datasets."""
    print("=" * 60)
    print("STEP 4: Benchmarking (Modal GPU)")
    print("=" * 60)
    print("This step requires Modal authentication and GPU access.")
    print("Run: modal run code/putnam-axiom-pipeline/modal_benchmark.py")
    return True

def run_tests():
    """Quick local tests (no GPU needed)."""
    print("=" * 60)
    print("QUICK TESTS (No GPU)")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Calculator
    print("\n--- Test 1: Agentic Calculator ---")
    try:
        sys.path.insert(0, 'code/agentic-calculator')
        from calculator import query_formula_sheet, safe_eval, AgenticCalculator
        
        # Test formula sheet
        result = query_formula_sheet('calculus')
        assert 'Power Rule' in result, "Formula sheet missing Power Rule"
        print("  ✓ Formula sheet works")
        
        # Test safe eval
        result = safe_eval('integrate(x**2, x)')
        assert 'x**3' in str(result), f"Integration failed: {result}"
        print("  ✓ Safe eval works")
        
        # Test new topics
        result = query_formula_sheet('ode')
        assert 'Separable' in result, "ODE formulas missing"
        print("  ✓ ODE formulas work")
        
        result = query_formula_sheet('complex')
        assert 'Cauchy' in result, "Complex analysis formulas missing"
        print("  ✓ Complex analysis formulas work")
        
    except Exception as e:
        print(f"  ✗ Calculator test failed: {e}")
        all_passed = False
    
    # Test 2: RL Components
    print("\n--- Test 2: RL Components ---")
    try:
        import torch
        sys.path.insert(0, 'code/putnam-axiom-pipeline')
        from rl_training import RIPOClippingLoss, ScoreCenteringLoss, math_reward
        
        # Test RIPO
        ripo = RIPOClippingLoss(epsilon=0.2)
        batch_size, seq_len = 4, 16
        old_log_probs = torch.randn(batch_size, seq_len)
        old_probs = torch.exp(old_log_probs) / torch.exp(old_log_probs).sum(dim=-1, keepdim=True)
        new_log_probs = old_log_probs + torch.randn(batch_size, seq_len) * 0.1
        advantages = torch.randn(batch_size, seq_len)
        loss = ripo(new_log_probs, old_log_probs, advantages, old_probs)
        assert not torch.isnan(loss), "RIPO loss is NaN"
        print(f"  ✓ RIPO loss: {loss.item():.4f}")
        
        # Test Score Centering
        sc = ScoreCenteringLoss(top_k=128)
        top_k = 128
        vocab_size = 100
        train_log_probs = torch.randn(batch_size, seq_len, top_k)
        samp_log_probs = torch.randn(batch_size, seq_len, top_k)
        topk_ids = torch.randint(0, vocab_size, (batch_size, seq_len, top_k))
        sampled_token = torch.randint(0, vocab_size, (batch_size, seq_len))
        samp_token_log_prob = torch.randn(batch_size, seq_len)
        adv = torch.randn(batch_size, seq_len)
        sc_loss = sc(train_log_probs, samp_log_probs, topk_ids, sampled_token, samp_token_log_prob, adv)
        assert not torch.isnan(sc_loss), "Score centering loss is NaN"
        print(f"  ✓ Score centering loss: {sc_loss.item():.4f}")
        
        # Test math reward
        rewards = math_reward(['\\boxed{42}', 'Wrong'], ['42', '42'])
        assert rewards[0] == 1.0, f"Correct answer not rewarded: {rewards[0]}"
        assert rewards[1] == 0.0, f"Wrong answer rewarded: {rewards[1]}"
        print(f"  ✓ Math reward: {rewards}")
        
    except Exception as e:
        print(f"  ✗ RL test failed: {e}")
        all_passed = False
    
    # Test 3: Dataset Loading
    print("\n--- Test 3: Dataset Loading ---")
    try:
        from datasets import load_dataset
        dataset = load_dataset("Putnam-AXIOM/putnam-axiom-dataset-ICML-2025-522")
        assert 'full_eval' in dataset, "Dataset missing full_eval split"
        assert len(dataset['full_eval']) == 522, f"Expected 522 problems, got {len(dataset['full_eval'])}"
        print(f"  ✓ Putnam-AXIOM: {len(dataset['full_eval'])} problems")
    except Exception as e:
        print(f"  ✗ Dataset test failed: {e}")
        all_passed = False
    
    # Test 4: Benchmark compilation
    print("\n--- Test 4: Benchmark Harness ---")
    try:
        import py_compile
        py_compile.compile('code/putnam-axiom-pipeline/benchmark.py', doraise=True)
        print("  ✓ Benchmark harness compiles")
    except Exception as e:
        print(f"  ✗ Benchmark test failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)
    
    return all_passed

def main():
    parser = argparse.ArgumentParser(description="Putnam-AXIOM RL Pipeline")
    parser.add_argument('--full', action='store_true', help='Run full pipeline')
    parser.add_argument('--analyze', action='store_true', help='Analyze dataset only')
    parser.add_argument('--train', action='store_true', help='Train only')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark only')
    parser.add_argument('--test', action='store_true', help='Run local tests')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3.5-2B', help='Model path')
    
    args = parser.parse_args()
    
    if not any([args.full, args.analyze, args.train, args.benchmark, args.test]):
        # Default: run tests
        args.test = True
    
    start_time = time.time()
    
    if args.test:
        success = run_tests()
    elif args.analyze:
        run_analysis()
        success = True
    elif args.full:
        run_analysis()
        run_formula_sheet_generation()
        run_training()
        run_benchmark(args.model)
        success = True
    elif args.train:
        run_training()
        success = True
    elif args.benchmark:
        run_benchmark(args.model)
        success = True
    
    elapsed = time.time() - start_time
    print(f"\nPipeline completed in {elapsed:.1f}s")
    
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()
