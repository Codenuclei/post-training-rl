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
    # Turn a group of responses into inspectable advantages

    Four responses to the same question receive rewards `[0, 0, 1, 1]`. The group mean is 0.5 and the population standard deviation is 0.5, giving advantages close to `[−1, −1, 1, 1]`. A group of entirely incorrect responses has no relative ordering: its advantages should be zero without dividing by zero.

    This CPU mechanism experiment includes a verifier, token-level clipping and an actual gradient update. The policy is a small token-probability table. The DPO and tool-calling Notebooks provide language-model training. All examples are generated locally; companion source: `code/chapter09_grpo_rlvr`.
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

    import json
    import os
    import re
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(7)
    OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "grpo"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rewards = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    advantages = (rewards - rewards.mean(-1, keepdim=True)) / (
        rewards.std(-1, correction=0, keepdim=True) + 1e-4)
    assert torch.allclose(advantages[1], torch.zeros(4))
    assert torch.isfinite(advantages).all()
    print("Rewards:\n", rewards, "\nAdvantages:\n", advantages)
    return OUTPUT, advantages, json, np, plt, re, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Reject ambiguous answers

    The verifier accepts exactly one integer in the form `Answer: 42`. It rejects conflicting answers and extra explanations. The cases below cover correct, incorrect, conflicting and non-integer answers.
    """)
    return


@app.cell
def _(re):
    def verify(response, expected):
        match = re.fullmatch(r"\s*Answer\s*[:：]\s*(-?\d+)\s*", response)
        return float(match is not None and int(match.group(1)) == expected)


    cases = [("Answer: 42", 42, 1), ("Answer: 35", 42, 0),
             ("Answer: 42 Answer: 35", 42, 0), ("Answer: -3", -3, 1),
             ("Answer: 42.5", 42, 0), ("42", 42, 0)]
    for response, expected, score in cases:
        actual = verify(response, expected)
        assert actual == score
        print(repr(response), "→", actual)
    return (cases,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Clip each token before averaging

    Consider token probability ratios 1.5 and 0.5, with clipping bounds `[0.8, 1.2]`. Multiplying the ratios first gives 0.75; clipping that sequence-level number changes the objective.

    We preserve shape `[response, token]`, exclude padding from the mean and broadcast each response's advantage across its tokens.
    """)
    return


@app.cell
def _(advantages, torch):
    ratios = torch.tensor([[1.5, 0.5]])
    positive = torch.ones(1, 1)
    per_token = torch.minimum(ratios * positive, ratios.clamp(0.8, 1.2) * positive).mean()
    sequence = torch.minimum(ratios.prod(-1), ratios.prod(-1).clamp(0.8, 1.2)).mean()
    assert abs(per_token.item() - 0.85) < 1e-6
    assert abs(sequence.item() - 0.75) < 1e-6
    print({"token_surrogate": per_token.item(), "sequence_surrogate": sequence.item()})


    def grpo_loss(logp, old_logp, ref_logp, advantage, mask, beta=0.04):
        ratio = torch.exp(logp - old_logp)
        surrogate = torch.minimum(ratio * advantage[:, None],
                                  ratio.clamp(0.8, 1.2) * advantage[:, None])
        log_ref_ratio = ref_logp - logp
        kl = torch.exp(log_ref_ratio) - log_ref_ratio - 1
        per_response = ((-surrogate + beta * kl) * mask).sum(-1) / mask.sum(-1)
        return per_response.mean()


    logits = torch.nn.Parameter(torch.zeros(4, 3, 2))
    actions = torch.tensor([[0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.float32)
    old_logp = logits.detach().log_softmax(-1).gather(-1, actions[..., None]).squeeze(-1)
    ref_logp = old_logp.clone()
    optimizer = torch.optim.SGD([logits], lr=0.5)
    losses = []
    for _ in range(4):
        logp = logits.log_softmax(-1).gather(-1, actions[..., None]).squeeze(-1)
        loss = grpo_loss(logp, old_logp, ref_logp, advantages[0], mask)
        optimizer.zero_grad()
        loss.backward()
        assert torch.equal(logits.grad[mask == 0], torch.zeros_like(logits.grad[mask == 0]))
        optimizer.step()
        losses.append(float(loss.detach()))
    change = (logits.detach().log_softmax(-1).gather(-1, actions[..., None]).squeeze(-1)
              - old_logp)
    mean_change = (change * mask).sum(-1) / mask.sum(-1)
    assert (mean_change[:2] < 0).all() and (mean_change[2:] > 0).all()
    print("Mean selected-token log-probability change:", mean_change.tolist())
    return mean_change, per_token, sequence


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Inspect gradients and results

    The table update checks three effects: better responses gain probability, worse responses lose probability and padding receives zero gradient.

    The experiment uses population standard deviation, `correction=0`. Frameworks that use sample standard deviation will produce a different advantage scale.
    """)
    return


@app.cell
def _(
    OUTPUT,
    advantages,
    cases,
    json,
    mean_change,
    np,
    per_token,
    plt,
    sequence,
):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].bar(np.arange(4), advantages[0].numpy())
    axes[0].set(xlabel="Response", ylabel="Group-relative advantage")
    axes[1].bar(np.arange(4), mean_change.numpy())
    axes[1].set(xlabel="Response", ylabel="Mean token log-probability change")
    fig.tight_layout()
    fig.savefig(OUTPUT / "mechanism.png", dpi=120)
    plt.show()
    metrics = {"verifier_tests": len(cases), "token_surrogate": per_token.item(),
               "sequence_surrogate": sequence.item(), "logprob_change": mean_change.tolist()}
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checklist

    - [ ] Obtain finite advantages for all-correct and all-incorrect groups, with zero advantages at zero variance.
    - [ ] Retain the token dimension through probability ratios and clipping; normalize response length afterward.
    - [ ] Reject conflicting answers and verify zero gradients at padding positions.

    ## Exercise

    Fill in the mean clipped token surrogate for ratios `[1.5, 0.5]` and positive advantage 1. Hint: compute `min(ratio, clip(ratio))` separately for each token before taking the mean. Enable `RUN_EXERCISE` after filling `answer`.
    """)
    return


@app.cell
def _():
    RUN_EXERCISE = False
    answer = None
    if RUN_EXERCISE:
        assert abs(answer - 0.85) < 1e-6
    else:
        print("Optional exercise: fill answer and enable RUN_EXERCISE.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [DeepSeekMath, Section 4.1](https://arxiv.org/html/2402.03300): token-level probability ratios, response-length averaging and the reference-policy KL term.
    - [Companion RLVR chapter](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter18_grpo/rlvr): connect a verifier to a trainable policy and examine reward-design limits.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saved run results

    CPU · Python 3.12.13 · 2.19 s

    [Open the full result page](https://walkinglabs.github.io/hands-on-modern-rl/lab-results/grpo-en.html)
    """)
    return


if __name__ == "__main__":
    app.run()
