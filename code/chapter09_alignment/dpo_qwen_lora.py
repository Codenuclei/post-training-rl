"""
DPO + LoRA on a small Qwen model, sized for a 4 GB GPU (RTX 2050) or a Colab T4.

- fp16 mixed precision instead of bf16 (T4 has no native bf16; this avoids the bf16 error)
- LoRA adapters: the reference model is the same model with the adapter disabled
- conversational preference data, so the Instruct chat template is applied automatically
- optional filtering of near-tie pairs by the UltraFeedback score gap (score_chosen - score_rejected)
- optional per-epoch evaluation of reward accuracy on held-out test pairs and on seen training pairs

Usage:
    python dpo_qwen_lora.py                      # full run (2000 pairs, 1 epoch)
    python dpo_qwen_lora.py --max_steps 5        # quick smoke test
    python dpo_qwen_lora.py --beta 0.5 --output_dir qwen-dpo-beta0.5
    python dpo_qwen_lora.py --num_epochs 4 --learning_rate 5e-5 --lora_r 32 --min_score_gap 1 \
        --lr_scheduler cosine --warmup_ratio 0.05 --eval_samples 200 --output_dir qwen-dpo-multi-epoch
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--max_prompt_length", type=int, default=192)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_epochs", type=float, default=1)
    parser.add_argument("--min_score_gap", type=float, default=0.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lr_scheduler", default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--rpo_alpha", type=float, default=None)
    parser.add_argument("--eval_samples", type=int, default=0)
    parser.add_argument("--output_dir", default="qwen-dpo")
    return parser.parse_args()


def generate(model, tokenizer, prompt, max_new_tokens=80):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)


def main():
    args = parse_args()
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    use_cuda = torch.cuda.is_available()
    # fp16 base weights (~1 GB) on GPU; PEFT keeps the trainable LoRA weights in fp32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16 if use_cuda else torch.float32
    )

    def load_pairs(split, n):
        ds = load_dataset(args.dataset, split=split)
        if args.min_score_gap > 0:
            # pairs whose scores are (nearly) tied carry little or contradictory preference signal
            ds = ds.filter(lambda x: x["score_chosen"] - x["score_rejected"] >= args.min_score_gap)
        return ds.select(range(min(n, len(ds))))

    train_dataset = load_pairs("train", args.num_samples)
    eval_dataset = None
    if args.eval_samples > 0:
        eval_dataset = {
            "test": load_pairs("test", args.eval_samples),
            "train": train_dataset.select(range(min(args.eval_samples, len(train_dataset)))),
        }

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
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        rpo_alpha=args.rpo_alpha,
        eval_strategy="epoch" if eval_dataset else "no",
        per_device_eval_batch_size=1,
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
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")

    # TRL logs each eval set separately; only the loss key carries the dataset name
    for log in trainer.state.log_history:
        for name, label in (("test", "held-out pairs"), ("train", "seen train pairs")):
            if f"eval_{name}_loss" in log:
                print(
                    f"epoch {log['epoch']:.2f} | {label:16s} | acc {log['eval_rewards/accuracies']:.3f} "
                    f"| margin {log['eval_rewards/margins']:.3f} | loss {log[f'eval_{name}_loss']:.4f}"
                )

    prompt = "How can I stay focused while studying for long hours?"
    trainer.model.eval()
    with trainer.model.disable_adapter():
        print("\n[Reference model]\n", generate(trainer.model, tokenizer, prompt))
    print("\n[DPO model]\n", generate(trainer.model, tokenizer, prompt))


if __name__ == "__main__":
    main()
