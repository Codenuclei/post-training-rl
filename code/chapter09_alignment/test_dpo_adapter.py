"""
Test a trained DPO LoRA adapter from dpo_qwen_lora.py, without TRL:
  1. side-by-side answers of the reference model (adapter off) and the DPO model (adapter on)
  2. preference accuracy on unseen test pairs, computed by hand from the DPO implicit reward
     r(x, y) = beta * (log pi_theta(y|x) - log pi_ref(y|x))

Usage:
    python test_dpo_adapter.py --adapter qwen-dpo-beta0.1 --num_pairs 200
    python test_dpo_adapter.py --adapter qwen-dpo-multi-epoch --num_pairs 200 --min_score_gap 1
"""
import argparse

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
parser.add_argument("--adapter", default="qwen-dpo-beta0.1")
parser.add_argument("--num_pairs", type=int, default=200)
parser.add_argument("--min_score_gap", type=float, default=0.0)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--show", type=int, default=5, help="number of test pairs to print in full")
parser.add_argument("--skip_generation", action="store_true")
args = parser.parse_args()
BASE, ADAPTER, BETA = args.base, args.adapter, args.beta
MAX_LEN, MAX_PROMPT = 384, 192  # same lengths as dpo_qwen_lora.py

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).cuda()
model = PeftModel.from_pretrained(base, ADAPTER).eval()


# ---------- Test 1: same prompt, reference model vs DPO model ----------
def generate(prompt, max_new_tokens=100):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = tok([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][ids.input_ids.shape[-1]:], skip_special_tokens=True).strip()


prompts = [
    "How can I stay focused while studying for long hours?",
    "Explain what a hash table is to a beginner in 3 sentences.",
    "Write a polite email declining a meeting invitation.",
]
print(f"\n===== TEST 1: answers (adapter = {ADAPTER}) =====")
for p in prompts if not args.skip_generation else []:
    with model.disable_adapter():
        ref = generate(p)
    dpo = generate(p)
    print(f"\nPROMPT: {p}\n--- reference ({len(ref)} chars):\n{ref}\n--- DPO ({len(dpo)} chars):\n{dpo}")


# ---------- Test 2: does the model prefer 'chosen' over 'rejected' on unseen pairs? ----------
def answer_logp(prompt_msgs, answer_msgs):
    """Sum of log-probs of the answer tokens only (exactly what DPO uses)."""
    prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(prompt_msgs + answer_msgs, tokenize=False)
    full = tok(full_text).input_ids
    n_prompt = len(tok(prompt_text).input_ids)
    # same truncation as training: keep the last MAX_PROMPT prompt tokens, then cut the answer to fit MAX_LEN
    p_ids = full[:n_prompt][-MAX_PROMPT:]
    a_ids = full[n_prompt:][: MAX_LEN - len(p_ids)]
    ids = torch.tensor([p_ids + a_ids]).cuda()
    with torch.no_grad():
        logits = model(ids).logits[:, :-1].float()
    logps = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return logps[0, len(p_ids) - 1:].sum().item()


pairs = load_dataset("trl-lib/ultrafeedback_binarized", split="test")
if args.min_score_gap > 0:
    pairs = pairs.filter(lambda x: x["score_chosen"] - x["score_rejected"] >= args.min_score_gap)
pairs = pairs.select(range(min(args.num_pairs, len(pairs))))
N_PAIRS = len(pairs)
wins, margins = 0, []
print(f"\n===== TEST 2: {N_PAIRS} unseen test pairs (score gap >= {args.min_score_gap}), beta = {BETA} =====")
for i, ex in enumerate(pairs):
    # this dataset stores the user turn inside chosen/rejected: split it off as the prompt
    prompt, chosen, rejected = ex["chosen"][:-1], ex["chosen"][-1:], ex["rejected"][-1:]
    pol_c, pol_r = answer_logp(prompt, chosen), answer_logp(prompt, rejected)
    with model.disable_adapter():
        ref_c, ref_r = answer_logp(prompt, chosen), answer_logp(prompt, rejected)
    r_c, r_r = BETA * (pol_c - ref_c), BETA * (pol_r - ref_r)   # implicit rewards
    wins += r_c > r_r
    margins.append(r_c - r_r)
    if i < args.show:
        short = lambda s: s.replace("\n", " ")[:160]
        print(f"\n--- pair {i}  (judge scores: chosen {ex['score_chosen']}, rejected {ex['score_rejected']})")
        print(f"PROMPT   : {short(prompt[-1]['content'])}")
        print(f"CHOSEN   : {short(chosen[0]['content'])}")
        print(f"REJECTED : {short(rejected[0]['content'])}")
        print(f"model reward chosen {r_c:+.3f} | rejected {r_r:+.3f} | {'OK' if r_c > r_r else 'WRONG'}")
print(f"\naccuracy {wins / N_PAIRS:.3f} | mean margin {sum(margins) / N_PAIRS:+.3f}")
