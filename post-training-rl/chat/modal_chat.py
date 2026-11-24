"""
Modal chat server for Qwen3.5-9B (BF16) with persistent model in VRAM.

Qwen3.5-9B is a multimodal vision-language model with hybrid Gated DeltaNet + sparse MoE.
We use it text-only via AutoModelForCausalLM. BF16 needs ~18GB VRAM, fits on L40S (48GB).

The model is loaded once in __enter__ and stays in VRAM across all prompts.
No restart on every prompt — only the first call pays the ~90s cold start.

Usage:
    # Quick test (one-shot):
    modal run code/qwen35_chat/modal_chat.py

    # Interactive chat mode (model stays warm):
    modal run code/qwen35_chat/modal_chat.py --interactive
"""
import modal
import time

app = modal.App("qwen35-chat")

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
class QwenChat:
    """Persistent chat class — model loads once and stays in VRAM."""

    @modal.enter()
    def load_model(self):
        """Load model once when the container starts. Stays in VRAM for all calls."""
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
    def generate(self, message: str, history: list = None, max_new_tokens: int = 512):
        """Generate a response. Model is already in VRAM — no reload."""
        import torch

        if history is None:
            history = []

        messages = history + [{"role": "user", "content": message}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        start = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        gen_time = time.time() - start

        prompt_len = inputs.input_ids.shape[1]
        generated_ids = out[0][prompt_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "response": response.strip(),
            "gen_time_sec": round(gen_time, 3),
            "tokens_per_sec": round(len(generated_ids) / gen_time, 2),
        }

    @modal.method()
    def info(self):
        """Return model and GPU info."""
        return {
            "model": "Qwen/Qwen3.5-9B",
            "gpu": self.device,
            "vram_gb": self.vram_gb,
            "params": "9B (BF16)",
        }


@app.local_entrypoint()
def main(interactive: bool = False):
    """Entry point. Use --interactive flag for chat mode."""
    chat = QwenChat()

    # First call triggers __enter__ which loads the model (~90s cold start)
    info = chat.info.remote()
    print(f"Model: {info['model']}")
    print(f"GPU: {info['gpu']} ({info['vram_gb']}GB VRAM)")

    if not interactive:
        result = chat.generate.remote(
            "What is a looped transformer? Explain in 2 sentences.",
            max_new_tokens=128,
        )
        print(f"\nResponse: {result['response']}")
        print(f"Time: {result['gen_time_sec']}s ({result['tokens_per_sec']} tok/s)")
        return

    # Interactive chat mode — model stays in VRAM, no restart per prompt
    history = []
    print("\n" + "=" * 60)
    print("Qwen3.5-9B Interactive Chat (Modal L40S GPU)")
    print("Model is loaded in VRAM — no restart on every prompt.")
    print("Type 'quit' or 'exit' to stop. Type 'clear' to reset history.")
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

        print("\n⏳ Thinking...")
        try:
            result = chat.generate.remote(user_input, history, max_new_tokens=512)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        response = result["response"]
        gen_time = result["gen_time_sec"]
        tok_sec = result["tokens_per_sec"]

        print(f"\n🤖 Qwen3.5-9B: {response}")
        print(f"   ({gen_time}s, {tok_sec} tok/s)")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        if len(history) > 20:
            history = history[-20:]
