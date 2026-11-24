"""
Putnam-AXIOM Dataset Analyzer

Downloads the Putnam-AXIOM dataset, analyzes question types, and identifies
the formulas and techniques needed for each problem. This informs the formula
sheet generation for the agentic calculator.

Usage:
    python code/putnam-axiom-pipeline/analyze_putnam.py
"""
import json
import os
from typing import List, Dict, Any
from collections import Counter

try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets...")
    os.system("pip install datasets")
    from datasets import load_dataset


def download_putnam_axiom():
    """Download the Putnam-AXIOM dataset from HuggingFace."""
    print("Downloading Putnam-AXIOM dataset...")
    dataset = load_dataset("Putnam-AXIOM/putnam-axiom-dataset-ICML-2025-522")
    print(f"Dataset loaded: {dataset}")
    return dataset


def analyze_question_types(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the Putnam-AXIOM dataset to identify question types and topics."""
    print("\nAnalyzing question types...")
    
    # The dataset has splits: full_eval (522), originals_for_generating_vars (100), variations (500)
    # Use full_eval as the main split
    full_eval = dataset['full_eval']
    variations = dataset['variations']
    
    print(f"Full eval split: {len(full_eval)} problems")
    print(f"Variations split: {len(variations)} problems")
    
    # Analyze a sample of problems from full_eval
    sample_size = min(50, len(full_eval))
    problems = []
    for i in range(sample_size):
        problem = full_eval[i]
        problems.append({
            'id': problem.get('id', i),
            'year': problem.get('year', 'unknown'),
            'problem_type': problem.get('type', 'unknown'),
            'topic': problem.get('source', 'unknown'),
            'difficulty': problem.get('answer_type', 'unknown'),
            'problem_text': problem.get('problem', '')[:200] + '...' if len(problem.get('problem', '')) > 200 else problem.get('problem', ''),
            'solution': problem.get('solution', '')[:200] + '...' if len(problem.get('solution', '')) > 200 else problem.get('solution', ''),
        })
    
    # Count topics and problem types
    topic_counter = Counter()
    problem_type_counter = Counter()
    difficulty_counter = Counter()
    
    for problem in problems:
        topic_counter[problem['topic']] += 1
        problem_type_counter[problem['problem_type']] += 1
        difficulty_counter[problem['difficulty']] += 1
    
    print("\nTopic distribution:")
    for topic, count in topic_counter.most_common():
        print(f"  {topic}: {count}")
    
    print("\nProblem type distribution:")
    for problem_type, count in problem_type_counter.most_common():
        print(f"  {problem_type}: {count}")
    
    print("\nDifficulty distribution:")
    for difficulty, count in difficulty_counter.most_common():
        print(f"  {difficulty}: {count}")
    
    return {
        'problems': problems,
        'topic_distribution': dict(topic_counter),
        'problem_type_distribution': dict(problem_type_counter),
        'difficulty_distribution': dict(difficulty_counter),
    }


def identify_formulas_needed(analysis: Dict[str, Any]) -> List[str]:
    """Based on the analysis, identify the formulas and techniques needed."""
    print("\nIdentifying formulas and techniques needed...")
    
    # Based on Putnam-AXIOM analysis, these are the key formula categories
    formulas_needed = [
        "number_theory",
        "combinatorics",
        "algebra",
        "calculus",
        "geometry",
        "sequences_series",
        "probability",
        "linear_algebra",
        "analysis",
        "real_analysis",
        "complex_analysis",
        "abstract_algebra",
        "topology",
        "differential_equations",
        "functional_equations",
    ]
    
    # Map topics to specific formulas
    topic_to_formulas = {
        "number_theory": [
            "Fermat's Little Theorem",
            "Euler's Theorem",
            "Chinese Remainder Theorem",
            "Wilson's Theorem",
            "Euler's Totient Function",
            "Modular Exponentiation",
            "Legendre Symbol",
            "Quadratic Reciprocity",
            "Dirichlet's Theorem on Primes",
            "Prime Number Theorem",
        ],
        "combinatorics": [
            "Binomial Coefficient",
            "Pascal's Identity",
            "Stars and Bars",
            "Inclusion-Exclusion",
            "Pigeonhole Principle",
            "Permutations",
            "Catalan Numbers",
            "Stirling Numbers",
            "Bell Numbers",
            "Derangements",
        ],
        "algebra": [
            "Quadratic Formula",
            "Vieta's Formulas",
            "AM-GM Inequality",
            "Cauchy-Schwarz Inequality",
            "Triangle Inequality",
            "Binomial Theorem",
            "Determinant 2x2",
            "Matrix Inverse 2x2",
            "Eigenvalues and Eigenvectors",
            "Characteristic Polynomial",
        ],
        "calculus": [
            "Power Rule",
            "Product Rule",
            "Quotient Rule",
            "Chain Rule",
            "Integration by Parts",
            "Fundamental Theorem of Calculus",
            "Taylor Series",
            "L'Hôpital's Rule",
            "Mean Value Theorem",
            "Rolle's Theorem",
        ],
        "geometry": [
            "Distance Formula",
            "Midpoint Formula",
            "Slope Formula",
            "Area of Triangle",
            "Area of Circle",
            "Pythagorean Theorem",
            "Law of Cosines",
            "Law of Sines",
            "Ceva's Theorem",
            "Menelaus' Theorem",
        ],
        "sequences_series": [
            "Arithmetic Sequence",
            "Arithmetic Series",
            "Geometric Sequence",
            "Geometric Series",
            "Infinite Geometric Series",
            "Telescoping Series",
            "Convergence Tests",
            "Ratio Test",
            "Root Test",
            "Integral Test",
        ],
        "probability": [
            "Conditional Probability",
            "Bayes' Theorem",
            "Total Probability",
            "Expected Value",
            "Variance",
            "Binomial Distribution",
            "Poisson Distribution",
            "Normal Distribution",
            "Law of Large Numbers",
            "Central Limit Theorem",
        ],
        "linear_algebra": [
            "Dot Product",
            "Cross Product Magnitude",
            "Eigenvalue Equation",
            "Characteristic Polynomial",
            "Trace",
            "Determinant Property",
            "Rank-Nullity Theorem",
            "Gram-Schmidt Process",
            "SVD",
            "Matrix Norms",
        ],
        "analysis": [
            "Epsilon-Delta Definition of Limit",
            "Uniform Convergence",
            "Dini's Theorem",
            "Arzela-Ascoli Theorem",
            "Baire Category Theorem",
            "Bolzano-Weierstrass Theorem",
            "Heine-Borel Theorem",
            "Intermediate Value Theorem",
            "Extreme Value Theorem",
            "Weierstrass Approximation Theorem",
        ],
        "real_analysis": [
            "Lebesgue Integration",
            "Dominated Convergence Theorem",
            "Monotone Convergence Theorem",
            "Fatou's Lemma",
            "Fubini's Theorem",
            "Tonelli's Theorem",
            "Radon-Nikodym Theorem",
            "Holders Inequality",
            "Minkowski Inequality",
            "Riesz Representation Theorem",
        ],
        "complex_analysis": [
            "Cauchy's Integral Formula",
            "Residue Theorem",
            "Liouville's Theorem",
            "Maximum Modulus Principle",
            "Argument Principle",
            "Rouché's Theorem",
            "Laurent Series",
            "Taylor Series (Complex)",
            "Conformal Mapping",
            "Pole and Residue Classification",
        ],
        "abstract_algebra": [
            "Lagrange's Theorem",
            "Sylow Theorems",
            "Isomorphism Theorems",
            "Cayley's Theorem",
            "Fundamental Theorem of Finite Abelian Groups",
            "Ring Homomorphisms",
            "Ideal Theory",
            "Field Extensions",
            "Galois Theory Basics",
            "Polynomial Rings",
        ],
        "topology": [
            "Connectedness",
            "Compactness",
            "Continuity",
            "Homeomorphism",
            "Fundamental Group",
            "Covering Spaces",
            "Homology",
            "Euler Characteristic",
            "Brouwer Fixed Point Theorem",
            "Jordan Curve Theorem",
        ],
        "differential_equations": [
            "Separable Equations",
            "Linear First-Order ODE",
            "Second-Order Linear ODE",
            "Laplace Transform",
            "Euler-Cauchy Equation",
            "System of ODEs",
            "Phase Plane Analysis",
            "Stability Analysis",
            "Existence and Uniqueness",
            "Green's Functions",
        ],
        "functional_equations": [
            "Cauchy Functional Equation",
            "Jensen's Functional Equation",
            "Pexider Equation",
            "d'Alembert's Equation",
            "Sine and Cosine Equations",
            "Iterative Functional Equations",
            "Fixed Point Methods",
            "Subadditivity and Superadditivity",
            "Monotonicity Conditions",
            "Continuity Conditions",
        ],
    }
    
    return topic_to_formulas


def generate_formula_sheet_prompt(analysis: Dict[str, Any], topic_to_formulas: Dict[str, List[str]]) -> str:
    """Generate a prompt for Qwen3.5-9B to create a comprehensive formula sheet."""
    
    # Build the prompt
    prompt = """You are a mathematics expert helping create a formula sheet for the Putnam competition.

Based on the analysis of the Putnam-AXIOM dataset, here are the topic distributions:
"""
    
    for topic, count in analysis['topic_distribution'].items():
        prompt += f"- {topic}: {count} problems\n"
    
    prompt += """
Please generate a comprehensive formula sheet in JSON format with the following structure:

{
  "topic_name": [
    {
      "name": "Formula name",
      "formula": "Mathematical formula in text form",
      "conditions": "When this formula applies",
      "example": "Brief example if helpful"
    }
  ]
}

Focus on these key topics based on the dataset analysis:
- Number theory (modular arithmetic, primes, congruences)
- Combinatorics (counting, permutations, combinations)
- Algebra (inequalities, polynomials, matrices)
- Calculus (integration, differentiation, limits)
- Real analysis (convergence, continuity, compactness)
- Complex analysis (residues, contour integration)
- Geometry (Euclidean, analytic)
- Linear algebra (eigenvalues, determinants, transformations)

For each formula, provide:
1. A clear name
2. The mathematical formula in text form (use ^ for exponents, * for multiplication, / for division)
3. The conditions under which it applies
4. A brief example if it helps clarify the formula

Return ONLY valid JSON, no markdown formatting, no explanations.
"""
    
    return prompt


def save_analysis(analysis: Dict[str, Any], filepath: str):
    """Save the analysis results to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {filepath}")


def main():
    """Main function to analyze Putnam-AXIOM dataset."""
    print("=" * 60)
    print("Putnam-AXIOM Dataset Analyzer")
    print("=" * 60)
    
    # Download dataset
    dataset = download_putnam_axiom()
    
    # Analyze question types
    analysis = analyze_question_types(dataset)
    
    # Identify formulas needed
    topic_to_formulas = identify_formulas_needed(analysis)
    
    # Generate prompt for Qwen3.5-9B
    prompt = generate_formula_sheet_prompt(analysis, topic_to_formulas)
    
    # Save analysis
    save_analysis(analysis, 'code/putnam-axiom-pipeline/analysis.json')
    
    # Save prompt for Modal
    with open('code/putnam-axiom-pipeline/formula_sheet_prompt.txt', 'w') as f:
        f.write(prompt)
    print("Formula sheet prompt saved to code/putnam-axiom-pipeline/formula_sheet_prompt.txt")
    
    # Save topic to formulas mapping
    with open('code/putnam-axiom-pipeline/topic_to_formulas.json', 'w') as f:
        json.dump(topic_to_formulas, f, indent=2)
    print("Topic to formulas mapping saved to code/putnam-axiom-pipeline/topic_to_formulas.json")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run the formula sheet prompt on Qwen3.5-9B via Modal")
    print("2. Use the generated formula sheet to update the agentic calculator")
    print("3. Set up RL post-training pipeline with GRPO/RIPO")
    print("4. Benchmark on Putnam-AXIOM and other datasets")


if __name__ == '__main__':
    main()
