"""
Modal integration: Qwen3.5-2B + Agentic Calculator v2 on T4 GPU.

This connects the calculator harness (with formula sheet support) to a real
model running on Modal. The model can call three tools:
1. calculator — safe Python eval with math/sympy
2. formula_sheet — query formulas by topic
3. verify — check if an answer matches expected result

Why this matters:
- A 2B model can reason about *what* to compute but not *compute* reliably.
- With calculator + formula sheet, it focuses on reasoning structure.
- For Putnam-level math, formulas are provided (open-book exam pattern).

Usage:
    # Quick test (one question):
    modal run code/agentic-calculator/modal_calc.py

    # Interactive chat with calculator:
    modal run code/agentic-calculator/modal_calc.py --interactive
"""
import modal
import time
import sys
import os

app = modal.App("agentic-calculator-v2")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch",
        "accelerate",
        "sentencepiece",
        "sympy",
    )
    .pip_install("git+https://github.com/huggingface/transformers.git")
    .add_local_file("code/agentic-calculator/calculator.py", "/root/calculator.py")
)


@app.cls(
    image=image,
    gpu="T4",  # $0.59/hr, 16GB VRAM — Qwen3.5-2B fits easily
    timeout=600,
    scaledown_window=3600,
)
class CalcModel:
    """Qwen3.5-2B with calculator + formula sheet tools on Modal T4."""

    @modal.enter()
    def load_model(self):
        """Load model once when the container starts."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_id = "Qwen/Qwen3.5-2B"
        print(f"Loading {model_id}... (~30s)")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.device = torch.cuda.get_device_name(0)
        self.vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"Model loaded on {self.device} ({self.vram_gb}GB VRAM)")

        # Import calculator module (copied into image via add_local_file)
        import importlib.util
        spec = importlib.util.spec_from_file_location('calculator', '/root/calculator.py')
        calc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(calc_module)
        self.agent = calc_module.AgenticCalculator(self._generate, max_turns=5)

    def _generate(self, messages: list) -> str:
        """Generate a response from the model (called by agentic loop)."""
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        prompt_len = inputs.input_ids.shape[1]
        generated_ids = out[0][prompt_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()

    @modal.method()
    def agentic_run(self, question: str) -> dict:
        """Run the agentic loop: model + calculator + formula sheet tools."""
        return self.agent.run(question)

    @modal.method()
    def info(self):
        """Return model and GPU info."""
        return {
            "model": "Qwen/Qwen3.5-2B",
            "gpu": self.device,
            "vram_gb": self.vram_gb,
            "tools": ["calculator", "formula_sheet", "verify"],
        }


@app.local_entrypoint()
def main(interactive: bool = False):
    """Entry point. Use --interactive flag for chat mode."""
    model = CalcModel()

    # First call triggers __enter__ which loads the model (~30s cold start)
    info = model.info.remote()
    print(f"Model: {info['model']}")
    print(f"GPU: {info['gpu']} ({info['vram_gb']}GB VRAM)")
    print(f"Tools: {', '.join(info['tools'])}")

    if not interactive:
        # Quick test
        question = "What is 7^100 mod 13?"
        print(f"\nQuestion: {question}")
        print("⏳ Reasoning...")

        start = time.time()
        result = model.agentic_run.remote(question)
        elapsed = time.time() - start

        print(f"\nAnswer: {result['final_answer']}")
        print(f"Tool calls: {len(result['tool_calls'])}")
        print(f"Turns: {result['total_turns']}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Success: {result['success']}")

        if result['tool_calls']:
            print("\nTool call trace:")
            for tool_name, code, tool_result in result['tool_calls']:
                code_short = code[:60] + "..." if len(code) > 60 else code
                result_short = tool_result[:60] + "..." if len(tool_result) > 60 else tool_result
                print(f"  [{tool_name}] {code_short} → {result_short}")
        return

    # Interactive chat mode
    history = []
    print("\n" + "=" * 60)
    print("Qwen3.5-2B + Agentic Calculator (Modal T4 GPU)")
    print("Tools: calculator, formula_sheet, verify")
    print("Model is loaded in VRAM — no restart on every prompt.")
    print("Type 'quit' or 'exit' to stop. Type 'clear' to reset.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            history = []
            print("[History cleared]")
            continue

        print("\n⏳ Reasoning...")
        try:
            start = time.time()
            result = model.agentic_run.remote(user_input)
            elapsed = time.time() - start
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        print(f"\n🤖 Answer: {result['final_answer']}")
        if result['tool_calls']:
            print(f"   ({len(result['tool_calls'])} tool calls, {result['total_turns']} turns, {elapsed:.1f}s)")
        else:
            print(f"   ({elapsed:.1f}s)")

        if not result['success']:
            print(f"   ⚠ Error: {result['error']}")
