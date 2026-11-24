"""
Run Qwen3.5-2B in 4GB VRAM on Modal T4 GPU using llama.cpp + GGUF.

This script proves that extreme quantization (IQ2_M at 0.94GB) lets a capable
2B model fit in 4GB VRAM with room for KV cache and context.

The model is loaded once via @app.cls and stays in VRAM across all prompts.
Uses llama.cpp CUDA backend for GPU acceleration on the cheapest Modal GPU (T4).

VRAM Budget (Qwen3.5-2B IQ2_M):
    Weights:     0.94 GB
    KV Cache:    ~0.5 GB (4K context)
    Overhead:    ~0.3 GB
    Total:       ~1.7 GB / 4 GB target (well under limit)

Usage:
    # Quick test (one-shot):
    modal run code/tiny-4gb-modal/modal_4gb.py

    # Interactive chat (model stays warm):
    modal run code/tiny-4gb-modal/modal_4gb.py --interactive
"""
import modal
import time

app = modal.App("tiny-4gb-chat")

# Build llama.cpp with CUDA from source inside the Modal container
image = (
    modal.Image.debian_slim()
    .apt_install(
        "build-essential",
        "cmake",
        "git",
        "curl",
    )
    .run_commands(
        # Clone and build llama.cpp with CUDA support
        "git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF && "
        "cmake --build build --config Release -j$(nproc)",
        # Verify build
        "ls /opt/llama.cpp/build/bin/llama-server",
    )
    .pip_install("huggingface_hub", "requests")
)


@app.cls(
    image=image,
    gpu="T4",  # Cheapest GPU: $0.59/hr, 16GB VRAM (we only need ~2GB)
    timeout=600,
    scaledown_window=3600,
)
class TinyChat:
    """Persistent chat class — model loads once and stays in VRAM."""

    @modal.enter()
    def load_model(self):
        """Download GGUF model and start llama.cpp server in background."""
        import os
        import subprocess
        import time as _time
        from huggingface_hub import hf_hub_download

        # Download the IQ2_M quantized GGUF (~0.94GB for Qwen3.5-2B)
        model_repo = "bartowski/Qwen_Qwen3.5-2B-GGUF"
        model_file = "Qwen3.5-2B-IQ2_M.gguf"

        print(f"Downloading {model_file} from {model_repo} (~0.94GB)...")
        model_path = hf_hub_download(
            repo_id=model_repo,
            filename=model_file,
            local_dir="/root/models",
        )
        print(f"Model saved to: {model_path}")

        # Start llama.cpp server with CUDA, all layers on GPU
        self.server_proc = subprocess.Popen(
            [
                "/opt/llama.cpp/build/bin/llama-server",
                "-m", model_path,
                "--host", "0.0.0.0",
                "--port", "8080",
                "-ngl", "99",       # All layers to GPU
                "-c", "4096",       # 4K context
                "--threads", "4",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait for server to be ready
        print("Waiting for llama.cpp server to start...")
        for i in range(60):
            try:
                import urllib.request
                urllib.request.urlopen("http://localhost:8080/health", timeout=1)
                print("Server is ready!")
                break
            except Exception:
                _time.sleep(2)
        else:
            raise RuntimeError("llama.cpp server failed to start")

        # Check VRAM usage
        self._check_vram()

    def _check_vram(self):
        """Check GPU VRAM usage via nvidia-smi."""
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits"],
                capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                used, total = lines[1].split(",")
                used_mb = int(used.strip())
                total_mb = int(total.strip())
                print(f"VRAM: {used_mb/1024:.1f} GB / {total_mb/1024:.1f} GB")
                if used_mb / 1024 < 4:
                    print(f"✅ Under 4GB limit! ({used_mb/1024:.1f} GB used)")
                else:
                    print(f"⚠️ Over 4GB: {used_mb/1024:.1f} GB used")
        except Exception as e:
            print(f"Could not check VRAM: {e}")

    @modal.method()
    def generate(self, message: str, history: list = None, max_new_tokens: int = 256):
        """Generate a response via llama.cpp OpenAI-compatible API."""
        import json
        import urllib.request

        if history is None:
            history = []

        messages = history + [{"role": "user", "content": message}]

        payload = json.dumps({
            "model": "Qwen3.5-2B",
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        }).encode()

        req = urllib.request.Request(
            "http://localhost:8080/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.time()
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        gen_time = time.time() - start

        response = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return {
            "response": response.strip(),
            "gen_time_sec": round(gen_time, 3),
            "tokens_per_sec": round(completion_tokens / gen_time, 2) if gen_time > 0 else 0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    @modal.method()
    def info(self):
        """Return model and GPU info."""
        import subprocess
        info = {
            "model": "Qwen3.5-2B",
            "quantization": "IQ2_M (0.94GB)",
            "gpu": "T4 (16GB VRAM, $0.59/hr)",
            "target_vram": "< 4GB",
        }
        # Get actual VRAM usage
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,nounits"],
                capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = [p.strip() for p in lines[1].split(",")]
                info["gpu_name"] = parts[0]
                info["vram_used_gb"] = round(int(parts[1]) / 1024, 1)
                info["vram_total_gb"] = round(int(parts[2]) / 1024, 1)
        except Exception:
            pass
        return info


@app.local_entrypoint()
def main(interactive: bool = False):
    """Entry point. Use --interactive flag for chat mode."""
    chat = TinyChat()

    # First call triggers __enter__ which downloads and loads the model
    info = chat.info.remote()
    print(f"Model: {info['model']} ({info.get('quantization', '')})")
    print(f"GPU: {info.get('gpu_name', info.get('gpu', 'T4'))}")
    vram_used = info.get("vram_used_gb")
    vram_total = info.get("vram_total_gb")
    if vram_used and vram_total:
        print(f"VRAM: {vram_used} GB / {vram_total} GB")
        if vram_used < 4:
            print(f"✅ Under 4GB limit!")

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
    print("Qwen3.5-2B IQ2_M Chat (Modal T4 GPU, < 4GB VRAM)")
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
            result = chat.generate.remote(user_input, history, max_new_tokens=256)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        response = result["response"]
        gen_time = result["gen_time_sec"]
        tok_sec = result["tokens_per_sec"]

        print(f"\n🤖 Qwen3.5-2B: {response}")
        print(f"   ({gen_time}s, {tok_sec} tok/s)")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        if len(history) > 20:
            history = history[-20:]
