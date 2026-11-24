"""
Agentic Calculator v2: Fail-safe tool-use harness for small models.

Design principles:
1. No crashes — every tool call returns either a result or a clear error message
2. Formula sheet support — model can query formulas before computing
3. Comprehensive edge case coverage — all math operations tested
4. Timeout protection — prevents infinite loops in eval
5. Sanitization — no code injection, no file I/O, no network access

Why tools amplify capability:
- A 2B model can reason about *what* to compute but not *compute* reliably.
- With a calculator + formula sheet, it focuses on reasoning structure.
- Qwen3.5-9B knows Putnam-level formulas (17/17 correct on probe),
  but the formula sheet reduces reasoning burden during RL training.
- The model's job is to match problem type → formula → computation.

Tools available:
1. calculator — safe Python eval with math/sympy
2. formula_sheet — query formulas by topic
3. verify — check if an answer matches expected format

This is the foundation for RL post-training with tool use on Putnam-AXIOM.
"""
import re
import json
import time
import signal
import math
import sympy
from typing import Optional, List, Tuple, Dict, Any


# ─── Timeout Protection ─────────────────────────────────────────────

class TimeoutError(Exception):
    """Raised when a tool call exceeds the time limit."""
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Tool execution timed out")


# ─── Safe Calculator ───────────────────────────────────────────────

ALLOWED_NAMES = {
    'pow': pow,
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'sum': sum,
    'len': len,
    'range': range,
    'list': list,
    'int': int,
    'float': float,
    'str': str,
    'bool': bool,
    'tuple': tuple,
    'dict': dict,
    'set': set,
    'True': True,
    'False': False,
    'None': None,
}

# Add math functions, but protect builtins like pow (math.pow doesn't support 3 args)
PROTECTED = set(ALLOWED_NAMES.keys())
ALLOWED_NAMES.update({
    k: getattr(math, k) for k in dir(math)
    if not k.startswith('_') and k not in PROTECTED
})

# Add sympy functions
ALLOWED_NAMES.update({
    'sympy': sympy,
    'factorial': math.factorial,
    'gcd': math.gcd,
    'lcm': math.lcm,
    'simplify': sympy.simplify,
    'expand': sympy.expand,
    'factor': sympy.factor,
    'solve': sympy.solve,
    'dsolve': sympy.dsolve,
    'integrate': sympy.integrate,
    'diff': sympy.diff,
    'limit': sympy.limit,
    'Matrix': sympy.Matrix,
    'eye': sympy.eye,
    'zeros': sympy.zeros,
    'ones': sympy.ones,
    'Symbol': sympy.Symbol,
    'symbols': sympy.symbols,
    'Integer': sympy.Integer,
    'Rational': sympy.Rational,
    'pi': sympy.pi,
    'E': sympy.E,
    'oo': sympy.oo,
    'I': sympy.I,
    'sqrt': sympy.sqrt,
    'log': sympy.log,
    'exp': sympy.exp,
    'sin': sympy.sin,
    'cos': sympy.cos,
    'tan': sympy.tan,
    'asin': sympy.asin,
    'acos': sympy.acos,
    'atan': sympy.atan,
    'sinh': sympy.sinh,
    'cosh': sympy.cosh,
    'tanh': sympy.tanh,
})


def safe_eval(expression: str, timeout_seconds: float = 10.0) -> str:
    """Evaluate a Python expression in a restricted environment.
    
    Safety features:
    - No __builtins__ access
    - No file I/O (open, import, etc.)
    - No network access
    - Timeout protection (default 10s)
    - Only math operations, sympy, and basic Python builtins
    
    Args:
        expression: Python expression to evaluate
        timeout_seconds: Maximum execution time in seconds
    
    Returns:
        String representation of the result or error message
    """
    # Set up timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    
    try:
        # Build the namespace with sympy pre-imported and default symbols
        namespace = dict(ALLOWED_NAMES)
        namespace['sympy'] = sympy
        namespace['x'] = sympy.Symbol('x')
        namespace['y'] = sympy.Symbol('y')
        namespace['z'] = sympy.Symbol('z')
        namespace['n'] = sympy.Symbol('n')
        namespace['k'] = sympy.Symbol('k')
        namespace['t'] = sympy.Symbol('t')
        # For dsolve: y must be a function of x, not a symbol
        namespace['y'] = sympy.Function('y')
        namespace['f'] = sympy.Function('f')
        namespace['g'] = sympy.Function('g')
        namespace['u'] = sympy.Function('u')
        namespace['v'] = sympy.Function('v')
        
        # Sanitize: reject dangerous patterns
        dangerous_patterns = [
            '__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
            'os.', 'sys.', 'subprocess', 'socket', 'http', 'urllib',
            'pickle', 'marshal', 'shutil', 'glob', 'pathlib',
        ]
        for pattern in dangerous_patterns:
            if pattern in expression.lower():
                return f"Error: Dangerous pattern detected: {pattern}"
        
        # Check if it's a simple expression or multi-statement code
        if ';' in expression:
            # Semicolon-separated statements: exec all but the last, eval the last
            parts = [p.strip() for p in expression.split(';') if p.strip()]
            for part in parts[:-1]:
                exec(part, {"__builtins__": {}}, namespace)
            result = eval(parts[-1], {"__builtins__": {}}, namespace)
        elif '\n' in expression and '=' in expression:
            # Multi-line with assignments: exec all but the last line, eval the last
            lines = [l.strip() for l in expression.split('\n') if l.strip() and not l.strip().startswith('#')]
            for line in lines[:-1]:
                exec(line, {"__builtins__": {}}, namespace)
            result = eval(lines[-1], {"__builtins__": {}}, namespace)
        else:
            # Simple expression
            result = eval(expression, {"__builtins__": {}}, namespace)
        
        # Convert sympy objects to string
        if hasattr(result, 'simplify'):
            result = sympy.simplify(result)
        if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, sympy.Basic)):
            return str(list(result))
        return str(result)
    
    except TimeoutError:
        return "Error: Computation timed out (exceeded 10 seconds)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    finally:
        # Cancel timeout and restore old handler
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


# ─── Formula Sheet ─────────────────────────────────────────────────

FORMULA_SHEET: Dict[str, List[Dict[str, str]]] = {
    "number_theory": [
        {"name": "Fermat's Little Theorem", "formula": "a^(p-1) ≡ 1 (mod p) for prime p and gcd(a,p)=1"},
        {"name": "Euler's Theorem", "formula": "a^φ(n) ≡ 1 (mod n) for gcd(a,n)=1"},
        {"name": "Chinese Remainder Theorem", "formula": "System of congruences has unique solution mod lcm of moduli"},
        {"name": "Wilson's Theorem", "formula": "(p-1)! ≡ -1 (mod p) for prime p"},
        {"name": "Euler's Totient Function", "formula": "φ(n) = n * ∏(1 - 1/p) for all prime factors p of n"},
        {"name": "Modular Exponentiation", "formula": "a^b mod m can be computed efficiently using repeated squaring"},
        {"name": "Legendre Symbol", "formula": "(a/p) = a^((p-1)/2) mod p for odd prime p"},
        {"name": "Quadratic Reciprocity", "formula": "(p/q)(q/p) = (-1)^((p-1)/2 * (q-1)/2) for odd primes p,q"},
    ],
    "combinatorics": [
        {"name": "Binomial Coefficient", "formula": "C(n,k) = n! / (k! * (n-k)!)"},
        {"name": "Pascal's Identity", "formula": "C(n,k) = C(n-1,k-1) + C(n-1,k)"},
        {"name": "Stars and Bars", "formula": "Number of ways to distribute n identical items into k bins: C(n+k-1, k-1)"},
        {"name": "Inclusion-Exclusion", "formula": "|A∪B| = |A| + |B| - |A∩B|"},
        {"name": "Pigeonhole Principle", "formula": "If n items are put into m containers and n > m, at least one container has ≥2 items"},
        {"name": "Permutations", "formula": "P(n,k) = n! / (n-k)!"},
        {"name": "Catalan Numbers", "formula": "C_n = (2n)! / ((n+1)! * n!)"},
    ],
    "algebra": [
        {"name": "Quadratic Formula", "formula": "x = (-b ± √(b²-4ac)) / 2a for ax²+bx+c=0"},
        {"name": "Vieta's Formulas", "formula": "For ax²+bx+c=0 with roots r1,r2: r1+r2=-b/a, r1*r2=c/a"},
        {"name": "AM-GM Inequality", "formula": "(x1+x2+...+xn)/n ≥ (x1*x2*...*xn)^(1/n)"},
        {"name": "Cauchy-Schwarz Inequality", "formula": "(∑a_i*b_i)² ≤ (∑a_i²)(∑b_i²)"},
        {"name": "Triangle Inequality", "formula": "|a+b| ≤ |a| + |b|"},
        {"name": "Binomial Theorem", "formula": "(x+y)^n = ∑ C(n,k) * x^k * y^(n-k)"},
        {"name": "Determinant 2x2", "formula": "det([[a,b],[c,d]]) = ad - bc"},
        {"name": "Matrix Inverse 2x2", "formula": "[[a,b],[c,d]]^(-1) = (1/(ad-bc)) * [[d,-b],[-c,a]]"},
    ],
    "calculus": [
        {"name": "Power Rule", "formula": "d/dx [x^n] = n*x^(n-1)"},
        {"name": "Product Rule", "formula": "d/dx [f*g] = f'*g + f*g'"},
        {"name": "Quotient Rule", "formula": "d/dx [f/g] = (f'*g - f*g') / g²"},
        {"name": "Chain Rule", "formula": "d/dx [f(g(x))] = f'(g(x)) * g'(x)"},
        {"name": "Integration by Parts", "formula": "∫u dv = uv - ∫v du"},
        {"name": "Fundamental Theorem of Calculus", "formula": "∫[a,b] f(x)dx = F(b) - F(a) where F' = f"},
        {"name": "Taylor Series", "formula": "f(x) = ∑ f^(n)(a)/n! * (x-a)^n"},
        {"name": "L'Hôpital's Rule", "formula": "lim f(x)/g(x) = lim f'(x)/g'(x) when 0/0 or ∞/∞"},
    ],
    "geometry": [
        {"name": "Distance Formula", "formula": "d = √((x2-x1)² + (y2-y1)²)"},
        {"name": "Midpoint Formula", "formula": "M = ((x1+x2)/2, (y1+y2)/2)"},
        {"name": "Slope Formula", "formula": "m = (y2-y1)/(x2-x1)"},
        {"name": "Area of Triangle", "formula": "A = (1/2) * base * height"},
        {"name": "Area of Circle", "formula": "A = πr²"},
        {"name": "Pythagorean Theorem", "formula": "a² + b² = c² for right triangle"},
        {"name": "Law of Cosines", "formula": "c² = a² + b² - 2ab*cos(C)"},
        {"name": "Law of Sines", "formula": "a/sin(A) = b/sin(B) = c/sin(C)"},
    ],
    "sequences_series": [
        {"name": "Arithmetic Sequence", "formula": "a_n = a_1 + (n-1)d"},
        {"name": "Arithmetic Series", "formula": "S_n = n/2 * (a_1 + a_n)"},
        {"name": "Geometric Sequence", "formula": "a_n = a_1 * r^(n-1)"},
        {"name": "Geometric Series", "formula": "S_n = a_1 * (1-r^n)/(1-r) for r≠1"},
        {"name": "Infinite Geometric Series", "formula": "S = a_1/(1-r) for |r|<1"},
        {"name": "Telescoping Series", "formula": "∑(a_n - a_(n+1)) = a_1 - lim a_n"},
    ],
    "probability": [
        {"name": "Conditional Probability", "formula": "P(A|B) = P(A∩B) / P(B)"},
        {"name": "Bayes' Theorem", "formula": "P(A|B) = P(B|A) * P(A) / P(B)"},
        {"name": "Total Probability", "formula": "P(B) = ∑ P(B|A_i) * P(A_i) for partition {A_i}"},
        {"name": "Expected Value", "formula": "E[X] = ∑ x_i * P(X=x_i)"},
        {"name": "Variance", "formula": "Var(X) = E[X²] - (E[X])²"},
        {"name": "Binomial Distribution", "formula": "P(X=k) = C(n,k) * p^k * (1-p)^(n-k)"},
    ],
    "linear_algebra": [
        {"name": "Dot Product", "formula": "a·b = ∑ a_i * b_i = |a||b|cos(θ)"},
        {"name": "Cross Product Magnitude", "formula": "|a×b| = |a||b|sin(θ)"},
        {"name": "Eigenvalue Equation", "formula": "Av = λv"},
        {"name": "Characteristic Polynomial", "formula": "det(A - λI) = 0"},
        {"name": "Trace", "formula": "tr(A) = ∑ a_ii = sum of eigenvalues"},
        {"name": "Determinant Property", "formula": "det(AB) = det(A) * det(B)"},
        {"name": "Rank-Nullity Theorem", "formula": "rank(A) + nullity(A) = n for n×n matrix"},
        {"name": "Cramer's Rule", "formula": "x_i = det(A_i) / det(A) where A_i replaces column i with b"},
        {"name": "Matrix Exponential", "formula": "e^A = ∑ A^n / n! for solving systems of ODEs"},
    ],
    "real_analysis": [
        {"name": "Epsilon-Delta Limit", "formula": "lim_{x→a} f(x) = L iff ∀ε>0 ∃δ>0: 0<|x-a|<δ ⇒ |f(x)-L|<ε"},
        {"name": "Uniform Continuity", "formula": "∀ε>0 ∃δ>0: ∀x,y: |x-y|<δ ⇒ |f(x)-f(y)|<ε"},
        {"name": "Bolzano-Weierstrass", "formula": "Every bounded sequence has a convergent subsequence"},
        {"name": "Intermediate Value Theorem", "formula": "If f continuous on [a,b] and f(a)<c<f(b), ∃x: f(x)=c"},
        {"name": "Mean Value Theorem", "formula": "∃c∈(a,b): f'(c) = (f(b)-f(a))/(b-a) for f differentiable on (a,b)"},
        {"name": "Rolle's Theorem", "formula": "If f(a)=f(b)=0 and f differentiable, ∃c: f'(c)=0"},
        {"name": "Extreme Value Theorem", "formula": "Continuous function on compact set attains max and min"},
        {"name": "Weierstrass Approximation", "formula": "Any continuous function on [a,b] can be uniformly approximated by polynomials"},
        {"name": "Dini's Theorem", "formula": "Monotone sequence of continuous functions converging pointwise to continuous function converges uniformly"},
        {"name": "Arzela-Ascoli", "formula": "A subset of C[X] is relatively compact iff it is equicontinuous and pointwise bounded"},
    ],
    "complex_analysis": [
        {"name": "Cauchy's Integral Formula", "formula": "f(a) = (1/2πi) ∮_γ f(z)/(z-a) dz"},
        {"name": "Residue Theorem", "formula": "∮_γ f(z) dz = 2πi ∑ Res(f, a_k) for poles a_k inside γ"},
        {"name": "Liouville's Theorem", "formula": "Bounded entire function is constant"},
        {"name": "Maximum Modulus Principle", "formula": "|f(z)| attains maximum on boundary of domain for holomorphic f"},
        {"name": "Cauchy-Riemann Equations", "formula": "∂u/∂x = ∂v/∂y and ∂u/∂y = -∂v/∂x for f=u+iv holomorphic"},
        {"name": "Laurent Series", "formula": "f(z) = ∑_{n=-∞}^{∞} a_n (z-a)^n in annulus"},
        {"name": "Argument Principle", "formula": "(1/2πi) ∮ f'(z)/f(z) dz = N - P (zeros minus poles)"},
        {"name": "Rouche's Theorem", "formula": "If |f-g| < |f|+|g| on boundary, f and g have same number of zeros inside"},
    ],
    "differential_equations": [
        {"name": "Separable ODE", "formula": "dy/dx = f(x)g(y) ⇒ ∫dy/g(y) = ∫f(x)dx"},
        {"name": "Linear First-Order ODE", "formula": "y' + P(x)y = Q(x) ⇒ y = e^{-∫P} (∫Qe^{∫P}dx + C)"},
        {"name": "Second-Order Linear ODE", "formula": "ay'' + by' + cy = 0, solve characteristic equation ar²+br+c=0"},
        {"name": "Euler-Cauchy Equation", "formula": "x²y'' + αxy' + βy = 0, try y = x^r"},
        {"name": "Laplace Transform", "formula": "L{f}(s) = ∫_0^∞ e^{-st}f(t)dt"},
        {"name": "Laplace of Derivative", "formula": "L{f'}(s) = sL{f}(s) - f(0)"},
        {"name": "Green's Function", "formula": "Solution to Ly=f via G(x,t) where LG=δ"},
    ],
    "functional_equations": [
        {"name": "Cauchy Functional Equation", "formula": "f(x+y) = f(x) + f(y) ⇒ f(x) = cx if continuous"},
        {"name": "Jensen's Equation", "formula": "f((x+y)/2) = (f(x)+f(y))/2 ⇒ f is affine if continuous"},
        {"name": "Multiplicative Cauchy", "formula": "f(xy) = f(x)f(y) ⇒ f(x) = x^c if continuous and positive"},
        {"name": "d'Alembert's Equation", "formula": "f(x+y) + f(x-y) = 2f(x)f(y) ⇒ f(x) = cosh(cx) or cos(cx)"},
        {"name": "Pexider Equation", "formula": "f(x+y) = g(x) + h(y) ⇒ reduce to Cauchy equation"},
    ],
    "trigonometry": [
        {"name": "Sum of Sines", "formula": "∑_{k=1}^{n} sin(kx) = sin(nx/2)sin((n+1)x/2) / sin(x/2)"},
        {"name": "Sum of Cosines", "formula": "∑_{k=1}^{n} cos(kx) = sin(nx/2)cos((n+1)x/2) / sin(x/2)"},
        {"name": "Product-to-Sum", "formula": "sin A sin B = (cos(A-B) - cos(A+B))/2"},
        {"name": "Half-Angle Formula", "formula": "sin²(x/2) = (1-cos x)/2, cos²(x/2) = (1+cos x)/2"},
        {"name": "Triple Angle", "formula": "sin(3x) = 3sin x - 4sin³x, cos(3x) = 4cos³x - 3cos x"},
    ],
}


def query_formula_sheet(topic: str, query: str = "") -> str:
    """Query the formula sheet for relevant formulas."""
    topic = topic.lower().strip()
    TOPIC_ALIASES = {
        'ode': 'differential_equations', 'odes': 'differential_equations',
        'diff_eq': 'differential_equations', 'de': 'differential_equations',
        'analysis': 'real_analysis', 'complex': 'complex_analysis',
        'functional': 'functional_equations', 'trig': 'trigonometry',
        'nt': 'number_theory', 'comb': 'combinatorics', 'calc': 'calculus',
        'geo': 'geometry', 'sequences': 'sequences_series', 'series': 'sequences_series',
        'prob': 'probability', 'la': 'linear_algebra', 'matrix': 'linear_algebra',
    }
    if topic in FORMULA_SHEET:
        formulas = FORMULA_SHEET[topic]
    elif topic in TOPIC_ALIASES:
        formulas = FORMULA_SHEET[TOPIC_ALIASES[topic]]
    else:
        matched = [k for k in FORMULA_SHEET.keys() if topic in k or k in topic]
        formulas = FORMULA_SHEET[matched[0]] if matched else []
    
    if query:
        q = query.lower()
        formulas = [f for f in formulas if q in f['name'].lower() or q in f['formula'].lower()]
    
    if not formulas:
        return f"No formulas found for topic '{topic}'. Available: {', '.join(FORMULA_SHEET.keys())}"
    
    result = f"Formulas for '{topic}':\n"
    for i, f in enumerate(formulas, 1):
        result += f"{i}. {f['name']}: {f['formula']}\n"
    return result.strip()


# ─── Answer Verification ───────────────────────────────────────────

def verify_answer(answer: str, expected: str) -> str:
    """Verify if an answer matches the expected result.
    
    Args:
        answer: The model's answer
        expected: The expected answer
    
    Returns:
        Verification result with details
    """
    try:
        # Try numeric comparison
        try:
            ans_float = float(answer)
            exp_float = float(expected)
            if abs(ans_float - exp_float) < 1e-6:
                return "✓ Correct (numeric match)"
            else:
                return f"✗ Incorrect: expected {expected}, got {answer}"
        except ValueError:
            pass
        
        # Try sympy comparison
        try:
            ans_expr = sympy.sympify(answer)
            exp_expr = sympy.sympify(expected)
            if sympy.simplify(ans_expr - exp_expr) == 0:
                return "✓ Correct (symbolic match)"
            else:
                return f"✗ Incorrect: expected {expected}, got {answer}"
        except:
            pass
        
        # String comparison
        if answer.strip() == expected.strip():
            return "✓ Correct (exact match)"
        else:
            return f"✗ Incorrect: expected {expected}, got {answer}"
    
    except Exception as e:
        return f"Error in verification: {e}"


# ─── Tool Call Parsing ─────────────────────────────────────────────

TOOL_PATTERN = re.compile(
    r'<tool>(\w+)</tool><code>(.*?)</code>',
    re.DOTALL
)


def parse_tool_calls(text: str) -> List[Tuple[str, str]]:
    """Parse tool calls from model output.
    
    Returns list of (tool_name, code) tuples.
    Handles malformed tags gracefully.
    """
    calls = []
    for match in TOOL_PATTERN.finditer(text):
        tool_name = match.group(1)
        code = match.group(2).strip()
        if code:  # Skip empty code blocks
            calls.append((tool_name, code))
    return calls


def execute_tool(tool_name: str, code: str) -> str:
    """Execute a tool call and return the result.
    
    Fail-safe: never crashes, always returns a string result or error.
    """
    try:
        if tool_name == "calculator":
            return safe_eval(code)
        elif tool_name == "formula_sheet":
            parts = code.split(':', 1)
            topic = parts[0].strip()
            query = parts[1].strip() if len(parts) > 1 else ""
            return query_formula_sheet(topic, query)
        elif tool_name == "verify":
            # Parse answer and expected from code
            parts = code.split(',', 1)
            if len(parts) == 2:
                answer = parts[0].strip()
                expected = parts[1].strip()
                return verify_answer(answer, expected)
            else:
                return "Error: Verify tool requires 'answer, expected' format"
        else:
            return f"Error: Unknown tool '{tool_name}'. Available tools: calculator, formula_sheet, verify"
    except Exception as e:
        return f"Error: Tool execution failed: {type(e).__name__}: {e}"


# ─── System Prompt ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You have access to three tools for solving math problems:

1. **calculator**: Evaluate mathematical expressions
   <tool>calculator</tool><code>pow(7, 4, 13)</code>
   Available: basic math, math module (sin, cos, log, exp, sqrt, factorial, gcd), sympy (solve, integrate, diff, limit, simplify, Matrix, etc.)

2. **formula_sheet**: Query formulas by topic
   <tool>formula_sheet</tool><code>number_theory</code>
   Topics: number_theory, combinatorics, algebra, calculus, geometry, sequences_series, probability, linear_algebra, real_analysis, complex_analysis, differential_equations, functional_equations, trigonometry

3. **verify**: Check if an answer matches expected result
   <tool>verify</tool><code>9, 9</code>

You already know mathematical formulas from your training. The formula sheet is available as a convenience to reduce reasoning burden.

Example:
Q: What is 7^100 mod 13?
A: By Fermat's little theorem, 7^12 ≡ 1 (mod 13). So 7^100 = 7^(12*8+4) ≡ 7^4 (mod 13).
<tool>calculator</tool><code>pow(7, 4, 13)</code>
Result: 9
Therefore, 7^100 mod 13 = 9.

Keep reasoning concise. Use tools for any non-trivial computation."""


# ─── Agentic Loop ──────────────────────────────────────────────────

class AgenticCalculator:
    """Fail-safe agentic loop with calculator, formula sheet, and verify tools.
    
    Safety features:
    - Max turns limit prevents infinite loops
    - Timeout on each tool call
    - Graceful error handling — never crashes
    - Clear final answer extraction
    """
    
    def __init__(self, model_generate, max_turns: int = 10, system_prompt: str = None):
        """
        Args:
            model_generate: Function that takes (messages) and returns generated text.
            max_turns: Maximum tool call turns (prevents infinite loops).
            system_prompt: Optional custom system prompt.
        """
        self.model_generate = model_generate
        self.max_turns = max_turns
        self.system_prompt = system_prompt or SYSTEM_PROMPT
    
    def run(self, question: str) -> dict:
        """Run the agentic loop for a single question.
        
        Returns:
            {
                "final_answer": str,
                "tool_calls": list of (tool_name, code, result) tuples,
                "total_turns": int,
                "full_trace": str,
                "success": bool,
                "error": str or None,
            }
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        
        tool_calls_log = []
        full_trace = []
        turns = 0
        error = None
        
        try:
            # First generation
            response = self.model_generate(messages)
            full_trace.append(f"Model: {response}")
            
            # Process tool calls iteratively
            while turns < self.max_turns:
                calls = parse_tool_calls(response)
                if not calls:
                    # No more tool calls — this is the final answer
                    break
                
                turns += 1
                for tool_name, code in calls:
                    result = execute_tool(tool_name, code)
                    tool_calls_log.append((tool_name, code, result))
                    full_trace.append(f"Tool [{tool_name}]: {code} → {result}")
                    
                    # Replace the tool call in the response with the result
                    tool_tag = f"<tool>{tool_name}</tool><code>{code}</code>"
                    response = response.replace(tool_tag, f"Result: {result}")
                
                # Add model's response with results to history
                messages.append({"role": "assistant", "content": response})
                
                # Prompt for continuation
                messages.append({"role": "user", "content": "Continue reasoning with the result above. If you have the final answer, state it clearly."})
                
                # Next generation
                response = self.model_generate(messages)
                full_trace.append(f"Model: {response}")
            
            success = True
        except Exception as e:
            # Fail-safe: catch any unexpected errors
            success = False
            error = f"Agentic loop failed: {type(e).__name__}: {e}"
            full_trace.append(f"Error: {error}")
            response = f"Error during reasoning: {error}"
        
        return {
            "final_answer": response.strip() if response else "No answer generated",
            "tool_calls": tool_calls_log,
            "total_turns": turns,
            "full_trace": "\n".join(full_trace),
            "success": success,
            "error": error,
        }


# ─── Comprehensive Edge Case Tests ─────────────────────────────────

def run_edge_case_tests():
    """Test all edge cases for the calculator and tools."""
    print("=" * 60)
    print("EDGE CASE TESTS FOR AGENTIC CALCULATOR")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    def test(name: str, actual: str, expected_contains: str):
        nonlocal tests_passed, tests_failed
        if expected_contains in actual:
            print(f"✓ {name}")
            tests_passed += 1
        else:
            print(f"✗ {name}")
            print(f"  Expected to contain: {expected_contains}")
            print(f"  Got: {actual}")
            tests_failed += 1
    
    # ─── Basic Arithmetic ───
    print("\n--- Basic Arithmetic ---")
    test("Addition", safe_eval("2 + 3"), "5")
    test("Subtraction", safe_eval("10 - 3"), "7")
    test("Multiplication", safe_eval("4 * 5"), "20")
    test("Division", safe_eval("10 / 3"), "3.333")
    test("Integer division", safe_eval("10 // 3"), "3")
    test("Modulo", safe_eval("10 % 3"), "1")
    test("Power", safe_eval("2 ** 10"), "1024")
    test("Negative numbers", safe_eval("-5 + 3"), "-2")
    test("Float precision", safe_eval("0.1 + 0.2"), "0.3")
    
    # ─── Modular Arithmetic ───
    print("\n--- Modular Arithmetic ---")
    test("Modular exponentiation", safe_eval("pow(7, 4, 13)"), "9")
    test("Large modular exp", safe_eval("pow(2, 100, 13)"), "3")
    test("Mod with negative", safe_eval("(-5) % 3"), "1")
    
    # ─── Math Module ───
    print("\n--- Math Module ---")
    test("Factorial", safe_eval("factorial(10)"), "3628800")
    test("GCD", safe_eval("gcd(12, 8)"), "4")
    test("LCM", safe_eval("lcm(12, 8)"), "24")
    test("Sqrt", safe_eval("sqrt(144)"), "12")
    test("Pi", safe_eval("str(round(pi, 2))"), "3.14")
    test("E", safe_eval("str(round(E, 2))"), "2.72")
    test("Sin", safe_eval("sin(pi/2)"), "1")
    test("Cos", safe_eval("cos(0)"), "1")
    test("Log", safe_eval("log(exp(1))"), "1")
    test("Exp", safe_eval("exp(0)"), "1")
    
    # ─── Sympy ───
    print("\n--- Sympy ---")
    test("Symbolic integration", safe_eval("sympy.integrate(3*x**2 + 2*x, x)"), "x**2")
    test("Symbolic differentiation", safe_eval("sympy.diff(x**3 + 2*x, x)"), "3*x**2 + 2")
    test("Solve equation", safe_eval("sympy.solve(x**2 - 4, x)"), "[-2, 2]")
    test("Simplify", safe_eval("sympy.simplify((x**2 - 1)/(x - 1))"), "x + 1")
    test("Expand", safe_eval("sympy.expand((x + 1)**2)"), "x**2 + 2*x + 1")
    test("Limit", safe_eval("sympy.limit(sin(x)/x, x, 0)"), "1")
    test("Matrix", safe_eval("sympy.Matrix([[1,2],[3,4]]).det()"), "-2")
    test("Factorial sympy", safe_eval("sympy.factorial(5)"), "120")
    
    # ─── Multi-statement ───
    print("\n--- Multi-statement ---")
    test("Variable assignment", safe_eval("x = 5; x**2"), "25")
    test("Multiple assignments", safe_eval("x = 3; y = 4; x**2 + y**2"), "25")
    test("List comprehension", safe_eval("[i**2 for i in range(5)]"), "[0, 1, 4, 9, 16]")
    
    # ─── Error Handling ───
    print("\n--- Error Handling ---")
    test("Division by zero", safe_eval("1 / 0"), "Error")
    test("Invalid syntax", safe_eval("2 + * 3"), "Error")
    test("Undefined variable", safe_eval("undefined_var"), "Error")
    test("Dangerous pattern", safe_eval("__import__('os')"), "Dangerous pattern")
    test("Import attempt", safe_eval("import os"), "Dangerous pattern")
    test("Exec attempt", safe_eval("exec('print(1)')"), "Dangerous pattern")
    
    # ─── Formula Sheet ───
    print("\n--- Formula Sheet ---")
    test("Number theory formulas", query_formula_sheet("number_theory"), "Fermat")
    test("Combinatorics formulas", query_formula_sheet("combinatorics"), "Binomial")
    test("Algebra formulas", query_formula_sheet("algebra"), "Quadratic")
    test("Calculus formulas", query_formula_sheet("calculus"), "Power Rule")
    test("Geometry formulas", query_formula_sheet("geometry"), "Pythagorean")
    test("Unknown topic", query_formula_sheet("nonexistent_topic_xyz"), "No formulas found")
    test("Fuzzy match", query_formula_sheet("number"), "Fermat")
    
    # ─── Tool Parsing ───
    print("\n--- Tool Parsing ---")
    test_calls = parse_tool_calls("<tool>calculator</tool><code>2 + 2</code>")
    test("Parse tool call", str(test_calls), "calculator")
    
    malformed = parse_tool_calls("<tool>calculator</tool><code></code>")
    test("Empty code block", str(malformed), "[]")
    
    no_tools = parse_tool_calls("No tools here")
    test("No tools in text", str(no_tools), "[]")
    
    # ─── Tool Execution ───
    print("\n--- Tool Execution ---")
    test("Calculator tool", execute_tool("calculator", "2 + 2"), "4")
    test("Unknown tool", execute_tool("unknown", "test"), "Unknown tool")
    test("Formula sheet tool", execute_tool("formula_sheet", "algebra"), "Quadratic")
    
    # ─── Verification ───
    print("\n--- Verification ---")
    test("Correct answer", verify_answer("5", "5"), "Correct")
    test("Incorrect answer", verify_answer("5", "6"), "Incorrect")
    test("Numeric match", verify_answer("3.14159", "3.14159"), "Correct")
    
    # ─── Summary ───
    print("\n" + "=" * 60)
    print(f"TESTS PASSED: {tests_passed}")
    print(f"TESTS FAILED: {tests_failed}")
    print(f"TOTAL: {tests_passed + tests_failed}")
    print("=" * 60)
    
    return tests_failed == 0


# ─── Standalone Test with Mock Model ───────────────────────────────

if __name__ == "__main__":
    # Run edge case tests first
    if not run_edge_case_tests():
        print("\n⚠ Some edge case tests failed!")
    
    print("\n" + "=" * 60)
    print("AGENTIC LOOP TESTS")
    print("=" * 60)
    
    # Mock model that knows how to use tools
    def mock_model(messages):
        last_user = ""
        last_assistant = ""
        for msg in reversed(messages):
            if msg["role"] == "user" and msg["content"] != "Continue reasoning with the result above. If you have the final answer, state it clearly.":
                last_user = msg["content"]
                break
        
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                last_assistant = msg["content"]
                break
        
        # Track conversation state using message count and content
        assistant_count = sum(1 for m in messages if m['role'] == 'assistant')
        
        # Test 1: Modular arithmetic with formula sheet
        if "7^100 mod 13" in last_user:
            if assistant_count == 0:
                return "Let me check number theory formulas first.\n<tool>formula_sheet</tool><code>number_theory</code>"
            elif "Result: 9" in last_assistant:
                # Calculator returned 9, give final answer (check BEFORE Fermat)
                return "Therefore, 7^100 mod 13 = 9."
            elif "Fermat" in last_assistant and "<tool>calculator</tool>" not in last_assistant:
                # Formula sheet was returned, now compute
                return "By Fermat's little theorem, 7^12 ≡ 1 (mod 13). So 7^100 = 7^(12*8+4) ≡ 7^4 (mod 13).\n<tool>calculator</tool><code>pow(7, 4, 13)</code>"
            else:
                return "Therefore, 7^100 mod 13 = 9."
        
        # Test 2: Integration
        elif "integral" in last_user.lower() or "∫" in last_user:
            if assistant_count == 0:
                return "<tool>calculator</tool><code>sympy.integrate(3*x**2 + 2*x, x)</code>"
            else:
                return "The integral of 3x² + 2x with respect to x is x³ + x² + C."
        
        # Test 3: Quadratic equation
        elif "quadratic" in last_user.lower() or "x^2" in last_user or "x²" in last_user:
            if assistant_count == 0:
                return "Let me check algebra formulas.\n<tool>formula_sheet</tool><code>algebra</code>"
            elif "Quadratic" in last_assistant and "<tool>calculator</tool>" not in last_assistant:
                return "Using the quadratic formula for x² - 5x + 6 = 0: a=1, b=-5, c=6\n<tool>calculator</tool><code>((-(-5) + sympy.sqrt((-5)**2 - 4*1*6)) / (2*1), ((-(-5) - sympy.sqrt((-5)**2 - 4*1*6)) / (2*1)))</code>"
            else:
                return "The roots are x = 2 and x = 3."
        
        else:
            return "I don't know how to solve this."
    
    agent = AgenticCalculator(mock_model)
    
    # Test 1: Modular arithmetic with formula sheet
    print("\nTest 1: 7^100 mod 13 (with formula sheet)")
    result = agent.run("What is 7^100 mod 13?")
    print(f"  Success: {result['success']}")
    print(f"  Final answer: {result['final_answer'][:100]}...")
    print(f"  Tool calls: {len(result['tool_calls'])}")
    print(f"  Turns: {result['total_turns']}")
    assert result['success'], "Test 1 should succeed"
    assert len(result['tool_calls']) > 0, "Test 1 should have tool calls"
    
    # Test 2: Integration
    print("\nTest 2: Integral of 3x² + 2x")
    result = agent.run("What is the integral of 3x² + 2x with respect to x?")
    print(f"  Success: {result['success']}")
    print(f"  Final answer: {result['final_answer'][:100]}...")
    print(f"  Tool calls: {len(result['tool_calls'])}")
    assert result['success'], "Test 2 should succeed"
    assert "x³" in result['final_answer'] or "x**3" in result['final_answer'], "Test 2 should mention x³"
    
    # Test 3: Quadratic equation
    print("\nTest 3: Solve x² - 5x + 6 = 0")
    result = agent.run("Solve the quadratic equation x² - 5x + 6 = 0")
    print(f"  Success: {result['success']}")
    print(f"  Final answer: {result['final_answer'][:100]}...")
    print(f"  Tool calls: {len(result['tool_calls'])}")
    assert result['success'], "Test 3 should succeed"
    
    # Test 4: Error handling
    print("\nTest 4: Error handling (malformed model)")
    def bad_model(messages):
        raise Exception("Model crashed")
    
    bad_agent = AgenticCalculator(bad_model)
    result = bad_agent.run("Test question")
    print(f"  Success: {result['success']}")
    print(f"  Error: {result['error']}")
    assert not result['success'], "Test 4 should fail gracefully"
    assert result['error'] is not None, "Test 4 should have error message"
    
    print("\n" + "=" * 60)
    print("ALL AGENTIC LOOP TESTS PASSED")
    print("=" * 60)
