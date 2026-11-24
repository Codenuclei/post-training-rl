"""
GSM8K with DPO + LoRA on a small Qwen model, sized for a 4 GB GPU (RTX 2050) or a Colab T4.

The preference pairs come from the model itself (on-policy, as in Iterative RPO, Pang et al. 2024):
  1. pairs : sample k solutions per GSM8K train question, grade each final answer against the gold answer,
             and pair a correct solution (chosen) with a wrong one (rejected) for the same question
  2. train : DPO + LoRA on those pairs, plus an NLL loss on the chosen solution (rpo_alpha)
  3. eval  : real GSM8K test accuracy (greedy decoding, exact match of the final number)
  4. scale : a direct weight-space edit with no gradients (task arithmetic, Ilharco et al. 2023): the merged
             weights are W0 + lambda * dW, where dW is the LoRA update learned by DPO; lambda is chosen on
             held-out GSM8K *train* questions (--split val), never on the test set
  5. loop  : optional looped transformer at inference (as in Inkling 2 Mini / Universal Transformers): layers
             START..END-1 run N times with shared weights (--loop 8:16:2); Qwen was not trained this way

Usage:
    python gsm8k_dpo.py eval                                   # base model, full test set (1319 questions)
    python gsm8k_dpo.py pairs --num_questions 2000 --k 4       # -> gsm8k_pairs.jsonl
    python gsm8k_dpo.py train --pairs gsm8k_pairs.jsonl --output_dir qwen-gsm8k-dpo
    python gsm8k_dpo.py eval --adapter qwen-gsm8k-dpo          # trained model, same test set
    python gsm8k_dpo.py eval --adapter qwen-gsm8k-dpo --split val --adapter_scale 1.5
    python gsm8k_dpo.py eval --split val --loop 8:16:2
"""

import argparse
import copy
import json
import random
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# the prompt format Qwen2.5 uses for its own math evaluations
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
# the last VAL_SIZE questions of the shuffled train split are never used for pairs; they select lambda
VAL_SIZE = 300


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["eval", "pairs", "train"])
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default=None, help="LoRA adapter folder to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64, help="sequences generated in parallel")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    # eval
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--adapter_scale", type=float, default=1.0, help="lambda in W0 + lambda * dW")
    parser.add_argument("--loop", default=None, help="START:END:N, run layers START..END-1 N times")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N questions")
    parser.add_argument("--show", type=int, default=3, help="test answers to print in full")
    # pairs
    parser.add_argument("--num_questions", type=int, default=2000)
    parser.add_argument("--k", type=int, default=4, help="sampled solutions per question")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--pairs_per_question", type=int, default=2)
    parser.add_argument("--pairs", default="gsm8k_pairs.jsonl")
    # train
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--rpo_alpha", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=float, default=1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--output_dir", default="qwen-gsm8k-dpo")
    return parser.parse_args()


def build_prompt(question):
    return [{"role": "user", "content": question.strip() + SUFFIX}]


def to_number(s):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def gold_answer(solution):
    # GSM8K reference solutions end with "#### <number>"
    return to_number(solution.split("####")[-1])


def extract_answer(text):
    # last \boxed{...}; if the model forgot it, fall back to the last number in the text
    boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    span = boxed[-1] if boxed else text
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", span)
    return to_number(numbers[-1]) if numbers else None


def is_correct(pred, gold):
    return pred is not None and gold is not None and abs(pred - gold) < 1e-4


def shuffled_train(args):
    return load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=args.seed)


def load_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        # LoRA adds dW = scaling * B @ A to each weight; multiplying scaling by lambda gives W0 + lambda * dW
        for module in model.modules():
            if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                for name in module.scaling:
                    module.scaling[name] *= args.adapter_scale
        # merging the LoRA weights makes generation as fast as the base model
        model = model.merge_and_unload()
    if args.loop:
        loop_layers(model, *map(int, args.loop.split(":")))
    return tok, model.eval()


def loop_layers(model, start, end, n):
    """Run decoder layers start..end-1 n times. The repeats share the weights (no extra parameters),
    but each repeat gets its own layer index so it keeps its own KV-cache slot."""
    layers, config = model.model.layers, model.config
    order = list(range(start)) + list(range(start, end)) * n + list(range(end, len(layers)))
    new_layers = []
    for new_idx, old_idx in enumerate(order):
        layer = copy.copy(layers[old_idx])
        layer._modules = dict(layer._modules)  # own child dict, same weight tensors
        layer.self_attn = copy.copy(layer.self_attn)
        layer.self_attn.layer_idx = new_idx
        new_layers.append(layer)
    if getattr(config, "layer_types", None):
        config.layer_types = [config.layer_types[i] for i in order]
    config.num_hidden_layers = len(order)
    model.model.layers = torch.nn.ModuleList(new_layers)
    print(f"looped layers {start}..{end - 1} x{n}: {len(layers)} stored layers, {len(order)} layer passes per token")


@torch.no_grad()
def generate(model, tok, prompts, args, n=1, temperature=0.0):
    """For each prompt, return n (text, finished) tuples; finished=False means it hit max_new_tokens."""
    texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths -> less padding
    eos_ids = torch.tensor(model.generation_config.eos_token_id, device=model.device).flatten()
    if temperature > 0:
        sampling = dict(do_sample=True, temperature=temperature, top_p=0.95, top_k=0)
    else:
        sampling = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
    per_batch = max(1, args.batch_size // n)
    results, start_time = [None] * len(texts), time.time()
    for b, start in enumerate(range(0, len(order), per_batch)):
        idx = order[start : start + per_batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **enc, max_new_tokens=args.max_new_tokens, num_return_sequences=n,
            repetition_penalty=1.0, pad_token_id=tok.pad_token_id, **sampling,
        )
        gen = out[:, enc.input_ids.shape[1] :]
        for j, i in enumerate(idx):
            results[i] = [
                (tok.decode(ids, skip_special_tokens=True).strip(), bool(torch.isin(ids, eos_ids).any()))
                for ids in gen[j * n : (j + 1) * n]
            ]
        done = min(start + per_batch, len(order))
        if b % 5 == 0 or done == len(order):
            print(f"  generated {done}/{len(order)} prompts ({time.time() - start_time:.0f}s)", flush=True)
    return results


def run_eval(args):
    tok, model = load_model(args)
    if args.split == "val":
        train = shuffled_train(args)
        test = train.select(range(len(train) - VAL_SIZE, len(train)))
    else:
        test = load_dataset("openai/gsm8k", "main", split="test")
    if args.limit:
        test = test.select(range(min(args.limit, len(test))))
    name = args.adapter or "base"
    if args.adapter and args.adapter_scale != 1.0:
        name += f"-x{args.adapter_scale:g}"
    if args.loop:
        name += f"-loop{args.loop.replace(':', '_')}"
    print(f"GSM8K {args.split}: {len(test)} questions, model = {args.model_id} + {name}, greedy decoding")
    outs = generate(model, tok, [build_prompt(q) for q in test["question"]], args)

    records, correct = [], 0
    for ex, [(text, finished)] in zip(test, outs):
        gold, pred = gold_answer(ex["answer"]), extract_answer(text)
        ok = is_correct(pred, gold)
        correct += ok
        records.append(dict(question=ex["question"], gold=gold, pred=pred, correct=ok,
                            boxed="\\boxed" in text, finished=finished, response=text))
    out_file = f"gsm8k-eval-{args.split}-{name.strip('/').replace('/', '_')}.jsonl"
    with open(out_file, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for r in records[: args.show]:
        print(f"\nQ: {r['question']}\nA: {r['response']}\n-> pred {r['pred']} | gold {r['gold']} | "
              f"{'OK' if r['correct'] else 'WRONG'}")
    n = len(records)
    print(f"\n[{name} | {args.split}] accuracy {correct / n:.4f} ({correct}/{n}) | "
          f"boxed {sum(r['boxed'] for r in records) / n:.3f} | "
          f"hit token limit {sum(not r['finished'] for r in records) / n:.3f} | "
          f"mean length {sum(len(r['response']) for r in records) / n:.0f} chars | saved {out_file}")


def run_pairs(args):
    tok, model = load_model(args)
    train = shuffled_train(args)
    train = train.select(range(min(args.num_questions, len(train) - VAL_SIZE)))
    print(f"sampling {args.k} solutions for {len(train)} GSM8K train questions (T={args.temperature})")
    outs = generate(model, tok, [build_prompt(q) for q in train["question"]], args,
                    n=args.k, temperature=args.temperature)

    rng = random.Random(args.seed)
    pairs, n_ok, kinds = [], 0, {"all correct": 0, "all wrong": 0, "mixed": 0}
    for ex, samples in zip(train, outs):
        gold = gold_answer(ex["answer"])
        good, bad = [], []
        for text, finished in samples:
            ok = is_correct(extract_answer(text), gold)
            n_ok += ok
            if ok and finished and "\\boxed" in text:
                good.append(text)  # chosen: right number, finished, in the requested format
            elif not ok:
                bad.append(text)
        good, bad = list(dict.fromkeys(good)), list(dict.fromkeys(bad))
        kinds["mixed" if good and bad else "all wrong" if not good else "all correct"] += 1
        rng.shuffle(good)
        rng.shuffle(bad)
        for c, r in list(zip(good, bad))[: args.pairs_per_question]:
            pairs.append({
                "prompt": build_prompt(ex["question"]),
                "chosen": [{"role": "assistant", "content": c}],
                "rejected": [{"role": "assistant", "content": r}],
            })
    with open(args.pairs, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    total = len(train) * args.k
    print(f"sample accuracy {n_ok / total:.3f} ({n_ok}/{total}) | questions: {kinds}")
    print(f"{len(pairs)} preference pairs saved to {args.pairs}")


def run_train(args):
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    # fp16 base weights (~1 GB) on GPU; PEFT keeps the trainable LoRA weights in fp32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.float16 if use_cuda else torch.float32
    )
    train_dataset = load_dataset("json", data_files=args.pairs, split="train").shuffle(seed=args.seed)
    print(f"{len(train_dataset)} preference pairs from {args.pairs}")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    training_args = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        loss_type="sigmoid",
        # NLL on the chosen solution keeps its likelihood from falling together with the rejected one
        rpo_alpha=args.rpo_alpha,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        max_prompt_length=256,
        truncation_mode="keep_start",
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        bf16=False,
        fp16=use_cuda,
        gradient_checkpointing=True,
        # non-reentrant checkpointing lets gradients reach LoRA weights even though the frozen inputs need no grad
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_logits_to_keep=True,
        logging_steps=1 if args.max_steps > 0 else 5,
        save_strategy="no",
        report_to="none",
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    {"eval": run_eval, "pairs": run_pairs, "train": run_train}[args.stage](args)
