import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Update a language model using paired preferences

    The request `help topic0` can be followed by `please read topic0 carefully` or by `go away topic0 now`. We first teach a small model to generate both responses, then use paired preferences to increase the relative probability of the first response.

    This CPU experiment trains a tiny autoregressive GRU from scratch and performs actual DPO parameter updates. All data is synthetic. The defaults are 400 SFT steps and 120 DPO steps, with no model download. The book's Qwen 0.5B recipe remains in `code/chapter17_dpo`.
    """)
    return


@app.cell
def _():
    import importlib.util
    import subprocess
    import sys

    packages = {"torch": "torch>=2.5,<3", "numpy": "numpy>=1.26,<3",
                "matplotlib": "matplotlib>=3.8,<4"}
    missing = [spec for name, spec in packages.items() if importlib.util.find_spec(name) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

    import copy
    import json
    import os
    import random
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(7)
    random.seed(7)
    OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "dpo"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SFT_STEPS, DPO_STEPS, BETA = 400, 120, 0.2
    words = ["<pad>", "<sep>", "<eos>", "help", "explain", "please", "read", "carefully",
             "go", "away", "now"] + [f"topic{i}" for i in range(12)]
    vocab = {word: index for index, word in enumerate(words)}
    print({"vocabulary": len(words), "sft_steps": SFT_STEPS, "dpo_steps": DPO_STEPS})
    return (
        BETA,
        DPO_STEPS,
        OUTPUT,
        SFT_STEPS,
        copy,
        json,
        nn,
        np,
        plt,
        random,
        torch,
        vocab,
        words,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Compute completion-only likelihood

    The input contains a prompt, a separator and a response. Each position's logits predict the next word. DPO sums log probabilities over the complete response, excluding prompt predictions and the separator itself and including the response EOS.

    Both responses here have equal lengths and require no padding. A variable-length extension must still sum only the valid response tokens.
    """)
    return


@app.cell
def _(nn, torch, vocab, words):
    def example(topic, verb="help"):
        return {"prompt": f"{verb} topic{topic}",
                "chosen": f"please read topic{topic} carefully <eos>",
                "rejected": f"go away topic{topic} now <eos>"}


    def batch(rows, field):
        inputs, masks = [], []
        for row in rows:
            prompt = row["prompt"].split() + ["<sep>"]
            response = row[field].split()
            inputs.append([vocab[w] for w in prompt + response])
            # After shifting, prediction position len(prompt)-1 targets the first response word.
            masks.append([0.0] * (len(prompt) - 1) + [1.0] * len(response))
        return torch.tensor(inputs), torch.tensor(masks)


    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(len(words), 32)
            self.gru = nn.GRU(32, 64, batch_first=True)
            self.head = nn.Linear(64, len(words))

        def forward(self, tokens):
            hidden, _ = self.gru(self.embedding(tokens))
            return self.head(hidden)


    def sequence_logp(model, tokens, mask):
        logits = model(tokens[:, :-1])
        per_token = logits.log_softmax(-1).gather(-1, tokens[:, 1:, None]).squeeze(-1)
        return (per_token * mask).sum(-1)


    test_tokens, test_mask = batch([example(0)], "chosen")
    assert test_tokens.shape == (1, 8) and test_mask.sum() == 5
    print("Example pair:", example(0))
    print("Tokens:", test_tokens.tolist(), "answer mask:", test_mask.tolist())
    return TinyLM, batch, example, sequence_logp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Train the language model and freeze a reference

    SFT covers 12 topics with equal numbers of both response styles. DPO receives preferences only for topics 0–7; topics 8–11 are held out from preference training. Held-out topics have appeared during SFT, so this split tests preference transfer rather than generalization to entirely unseen topics.

    After SFT, we copy the model to create a reference whose parameters remain frozen during DPO.
    """)
    return


@app.cell
def _(
    BETA,
    SFT_STEPS,
    TinyLM,
    batch,
    copy,
    example,
    random,
    sequence_logp,
    torch,
    vocab,
    words,
):
    all_rows = [example(i, verb) for i in range(12) for verb in ['help', 'explain']]
    train_rows = [row for i in range(8) for row in [example(i), example(i, 'explain')]]
    heldout = [row for i in range(8, 12) for row in [example(i), example(i, 'explain')]]
    model = TinyLM()
    _optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    sft_losses = []
    for _step in range(SFT_STEPS):
        _rows = random.choices(all_rows, k=16)
        _chosen, chosen_mask = batch(_rows, 'chosen')
        _rejected, rejected_mask = batch(_rows, 'rejected')
        _loss = -(sequence_logp(model, _chosen, chosen_mask).mean() + sequence_logp(model, _rejected, rejected_mask).mean()) / 10
        _optimizer.zero_grad()
        _loss.backward()
        _optimizer.step()
        sft_losses.append(float(_loss.detach()))
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    reference_snapshot = {name: value.clone() for name, value in reference.state_dict().items()}

    @torch.no_grad()
    def evaluate(policy, rows):
        c, cm = batch(rows, 'chosen')
        r, rm = batch(rows, 'rejected')
        gap = sequence_logp(policy, c, cm) - sequence_logp(policy, r, rm)
        ref_gap = sequence_logp(reference, c, cm) - sequence_logp(reference, r, rm)
        return {'chosen_pair_rate': float((gap > 0).float().mean()), 'logprob_gap': float(gap.mean()), 'reward_margin': float(BETA * (gap - ref_gap).mean())}

    @torch.no_grad()
    def generate(policy, prompt):
        tokens = [vocab[w] for w in prompt.split() + ['<sep>']]
        output = []
        for _ in range(8):
            token = int(policy(torch.tensor([tokens]))[0, -1].argmax())
            tokens.append(token)
            if words[token] == '<eos>':
                break
            output.append(words[token])
        return ' '.join(output)
    before = evaluate(model, heldout)
    before_text = [generate(model, row['prompt']) for row in heldout]
    print('SFT held-out preference metrics:', before)
    print('Sample before:', before_text[0])
    return (
        before,
        before_text,
        evaluate,
        generate,
        heldout,
        model,
        reference,
        reference_snapshot,
        sft_losses,
        train_rows,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Optimize changes relative to the reference

    Suppose the chosen response gains 2 log-probability units relative to the reference, while the rejected response gains 0.5. Their relative gap is 1.5. After multiplication by β, the negative log-sigmoid loss encourages this gap to grow.

    The reference contributes fixed likelihood values. Both chosen and rejected likelihoods from the current model contribute gradients.
    """)
    return


@app.cell
def _(
    BETA,
    DPO_STEPS,
    batch,
    before,
    before_text,
    evaluate,
    generate,
    heldout,
    model,
    nn,
    np,
    random,
    reference,
    reference_snapshot,
    sequence_logp,
    torch,
    train_rows,
):
    _optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)
    losses, margins = ([], [])
    for _step in range(DPO_STEPS):
        _rows = random.choices(train_rows, k=16)
        _chosen, cm = batch(_rows, 'chosen')
        _rejected, rm = batch(_rows, 'rejected')
        current_gap = sequence_logp(model, _chosen, cm) - sequence_logp(model, _rejected, rm)
        with torch.no_grad():
            ref_gap = sequence_logp(reference, _chosen, cm) - sequence_logp(reference, _rejected, rm)
        margin = BETA * (current_gap - ref_gap)
        _loss = -nn.functional.logsigmoid(margin).mean()
        _optimizer.zero_grad()
        _loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        _optimizer.step()
        losses.append(float(_loss.detach()))
        margins.append(evaluate(model, heldout)['reward_margin'])
    after = evaluate(model, heldout)
    after_text = [generate(model, row['prompt']) for row in heldout]
    topic_before = np.mean([row['prompt'].split()[-1] in text.split() for row, text in zip(heldout, before_text)])
    topic_after = np.mean([row['prompt'].split()[-1] in text.split() for row, text in zip(heldout, after_text)])
    assert all((torch.equal(reference.state_dict()[key], value) for key, value in reference_snapshot.items()))
    assert after['reward_margin'] > 0
    assert any((not torch.equal(value, reference.state_dict()[key]) for key, value in model.state_dict().items()))
    print('Before:', before, '\nAfter:', after)
    print('Requested topic retention before / after:', topic_before, topic_after)
    for row, old, new in zip(heldout[:4], before_text, after_text):
        print(row['prompt'], '\n before:', old, '\n after:', new)
    return after, after_text, losses, margins, topic_after, topic_before


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Evaluate preference and generated content separately

    A growing preference margin establishes that the chosen responses become more preferred relative to the rejected responses. Generated answers may still lose their requested topic or repeat words, so we also inspect response content and topic retention.

    Inspect the generated answers together with preference accuracy and topic retention.
    """)
    return


@app.cell
def _(
    OUTPUT,
    after,
    after_text,
    before,
    before_text,
    heldout,
    json,
    losses,
    margins,
    model,
    plt,
    sft_losses,
    topic_after,
    topic_before,
    torch,
    train_rows,
    words,
):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, data, label in zip(axes, [sft_losses, losses, margins],
                              ["SFT token loss", "DPO loss", "Held-out reward margin"]):
        ax.plot(data)
        ax.set(xlabel="Update", ylabel=label)
    fig.tight_layout()
    fig.savefig(OUTPUT / "training.png", dpi=120)
    plt.show()
    metrics = {"before": before, "after": after,
               "requested_topic_retention": {"before": float(topic_before),
                                             "after": float(topic_after)},
               "examples": [{"prompt": row["prompt"], "before": old, "after": new}
                            for row, old, new in zip(heldout, before_text, after_text)]}
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUTPUT / "preferences.json").write_text(json.dumps(train_rows, indent=2))
    torch.save({"state_dict": model.state_dict(), "vocabulary": words}, OUTPUT / "tiny-dpo.pt")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checklist

    - [ ] Train an autoregressive model and verify that DPO changes its parameters.
    - [ ] Keep the reference frozen and sum likelihoods only over valid response tokens.
    - [ ] Evaluate held-out preference margins separately from generated content.

    ## Exercise

    The chosen and rejected log probabilities change by 2 and 0.5 relative to the reference, with β=0.2. Fill in the reward margin. Hint: subtract first, then multiply by β. Enable `RUN_EXERCISE` after filling `answer`.
    """)
    return


@app.cell
def _():
    RUN_EXERCISE = False
    answer = None
    if RUN_EXERCISE:
        assert abs(answer - 0.3) < 1e-6
    else:
        print("Optional exercise: fill answer and enable RUN_EXERCISE.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [DPO paper](https://arxiv.org/abs/2305.18290): paired response log-probability differences relative to a fixed reference policy.
    - [Companion Qwen DPO experiment](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter17_dpo/dpo-hands-on): use the same objective with a pretrained model and a different data and resource scale.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saved run results

    CPU · Python 3.12.13 · 5.02 s

    [Open the full result page](https://walkinglabs.github.io/hands-on-modern-rl/lab-results/dpo-en.html)
    """)
    return


if __name__ == "__main__":
    app.run()
