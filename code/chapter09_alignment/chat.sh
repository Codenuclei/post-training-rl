#!/usr/bin/env bash
# Chat in the terminal with the DPO-trained Qwen model (LoRA adapter from dpo_qwen_lora.py).
#
# Usage:
#   ./chat.sh                          # adapter qwen-dpo-beta0.1
#   ./chat.sh qwen-dpo-multi-epoch     # any other adapter folder
#
# Commands inside the chat:
#   /dpo    answer with the DPO model (adapter on, default)
#   /ref    answer with the original model (adapter off)
#   /both   answer with both models, one after the other
#   /reset  clear the conversation history
#   /exit   quit (Ctrl+D also works)
set -euo pipefail
cd "$(dirname "$0")"

ADAPTER="${1:-qwen-dpo-beta0.1}"
PY="../../.venv/bin/python"
[ -x "$PY" ] || PY="python3"
[ -d "$ADAPTER" ] || { echo "adapter folder '$ADAPTER' not found in $(pwd)"; exit 1; }

# keep the screen clean: hide library warnings and progress bars
export TRANSFORMERS_VERBOSITY=error HF_HUB_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false

exec "$PY" -c "$(cat <<'EOF'
import contextlib
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
adapter = sys.argv[1]
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"loading {BASE} + {adapter} on {device} ...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
model = PeftModel.from_pretrained(base, adapter).eval()
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)


def reply(history, use_adapter):
    text = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    ids = tok([text], return_tensors="pt").to(device)
    ctx = contextlib.nullcontext() if use_adapter else model.disable_adapter()
    with ctx, torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9,
            repetition_penalty=1.1, streamer=streamer,
        )
    return tok.decode(out[0][ids.input_ids.shape[-1]:], skip_special_tokens=True).strip()


mode, history = "dpo", []
print("ready. commands: /dpo /ref /both /reset /exit\n")
while True:
    try:
        user = input(f"you [{mode}]> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not user:
        continue
    if user in ("/exit", "/quit"):
        break
    if user in ("/dpo", "/ref", "/both"):
        mode = user[1:]
        print(f"(mode: {mode})\n")
        continue
    if user == "/reset":
        history = []
        print("(history cleared)\n")
        continue

    history.append({"role": "user", "content": user})
    answers = {}
    for name in (["ref", "dpo"] if mode == "both" else [mode]):
        print(f"\n{name}> ", end="", flush=True)
        answers[name] = reply(history, use_adapter=(name == "dpo"))
        print()
    # in /both mode the conversation continues with the DPO answer
    history.append({"role": "assistant", "content": answers.get("dpo", answers.get("ref"))})
EOF
)" "$ADAPTER"
