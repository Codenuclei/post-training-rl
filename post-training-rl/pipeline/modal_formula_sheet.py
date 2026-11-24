"""
Modal script to generate comprehensive formula sheet using Qwen3.5-9B.

This script uses Qwen3.5-9B on Modal L40S GPU to generate a comprehensive
formula sheet based on the Putnam-AXIOM dataset analysis.

Usage:
    modal run code/putnam-axiom-pipeline/modal_formula_sheet.py
"""
import modal
import json
import time

app = modal.App("putnam-formula-sheet")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch",
        "accelerate",
        "sentencepiece",
        "torchvision",
        "pillow",
    )
    .pip_install("git+https://github.com/huggingface/transformers.git")
)


@app.cls(
    image=image,
    gpu="L40S",
    timeout=600,
    scaledown_window=3600,
)
class FormulaSheetGenerator:
    """Generate formula sheet using Qwen3.5-9B."""

    @modal.enter()
    def load_model(self):
        """Load model once when the container starts."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_id = "Qwen/Qwen3.5-9B"
        print(f"Loading {model_id}... (first time only, ~90s)")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.device = torch.cuda.get_device_name(0)
        self.vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"Model loaded on {self.device} ({self.vram_gb}GB VRAM)")

    @modal.method()
    def generate_formula_sheet(self, topic: str) -> str:
        """Generate formulas for a specific topic."""
        import torch

        prompt = f"""You are a mathematics expert. Generate a comprehensive list of formulas for the topic: {topic}.

Return ONLY valid JSON in this exact format:
[
  {{
    "name": "Formula name",
    "formula": "Mathematical formula in text form",
    "conditions": "When this formula applies"
  }}
]

Focus on formulas relevant to the Putnam competition (undergraduate math competition).
Return ONLY the JSON array, no markdown formatting, no explanations."""

        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        start = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=1.0,
            )
        gen_time = time.time() - start

        prompt_len = inputs.input_ids.shape[1]
        generated_ids = out[0][prompt_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    @modal.method()
    def info(self):
        """Return model and GPU info."""
        return {
            "model": "Qwen/Qwen3.5-9B",
            "gpu": self.device,
            "vram_gb": self.vram_gb,
        }


@app.local_entrypoint()
def main():
    """Generate comprehensive formula sheet for all Putnam topics."""
    generator = FormulaSheetGenerator()

    # Get model info
    info = generator.info.remote()
    print(f"Model: {info['model']}")
    print(f"GPU: {info['gpu']} ({info['vram_gb']}GB VRAM)")

    # Topics to generate formulas for
    topics = [
        "number_theory",
        "combinatorics",
        "algebra",
        "calculus",
        "geometry",
        "sequences_series",
        "probability",
        "linear_algebra",
        "real_analysis",
        "complex_analysis",
    ]

    formula_sheet = {}

    print("\n" + "=" * 60)
    print("Generating formula sheet for Putnam-AXIOM")
    print("=" * 60)

    for topic in topics:
        print(f"\nGenerating formulas for: {topic}...")
        try:
            response = generator.generate_formula_sheet.remote(topic)
            print(f"Response length: {len(response)} characters")

            # Try to parse JSON
            try:
                # Extract JSON from response (remove markdown formatting if present)
                json_str = response
                if '```json' in json_str:
                    json_str = json_str.split('```json')[1].split('```')[0]
                elif '```' in json_str:
                    json_str = json_str.split('```')[1].split('```')[0]

                formulas = json.loads(json_str)
                formula_sheet[topic] = formulas
                print(f"✓ Successfully parsed {len(formulas)} formulas")
            except json.JSONDecodeError as e:
                print(f"✗ JSON parse error: {e}")
                print(f"Response preview: {response[:200]}...")
                formula_sheet[topic] = []

        except Exception as e:
            print(f"✗ Error generating formulas for {topic}: {e}")
            formula_sheet[topic] = []

    # Save formula sheet
    output_path = "code/putnam-axiom-pipeline/generated_formula_sheet.json"
    with open(output_path, 'w') as f:
        json.dump(formula_sheet, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Formula sheet saved to {output_path}")
    print(f"{'=' * 60}")

    # Count total formulas
    total_formulas = sum(len(formulas) for formulas in formula_sheet.values())
    print(f"\nTotal formulas generated: {total_formulas}")

    print("\nNext steps:")
    print("1. Review the generated formula sheet")
    print("2. Update the agentic calculator with the new formulas")
    print("3. Set up RL post-training pipeline")

