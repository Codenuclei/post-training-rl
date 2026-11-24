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
    # Train a tool policy from mock execution results

    A request asks for AAPL's price on day2. One candidate queries AAPL; another queries MSFT. Both JSON strings may parse successfully. We still need to check the arguments against the request and execute the mock tool to inspect its result.

    We score format, schema, arguments and execution separately, then compare candidates for the same request. The CPU defaults are 500 syntax-pretraining steps and 180 GRPO updates, with eight candidates per group.

    A tiny GRU generates three tool-language tokens and EOS; a deterministic renderer converts them to JSON. The policy learns tool names and arguments from synthetic requests and local mock-tool results.
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
    torch.manual_seed(12)
    random.seed(12)
    OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "financial"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SFT_STEPS, UPDATES, GROUP, BETA = 500, 180, 8, 0.02
    print({"syntax_updates": SFT_STEPS, "grpo_updates": UPDATES, "group_size": GROUP})
    return (
        BETA,
        GROUP,
        OUTPUT,
        SFT_STEPS,
        UPDATES,
        copy,
        json,
        nn,
        np,
        plt,
        random,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Fix the dataset, mock tools and held-out requests

    The split groups requests by tool, entity and date or amount. The same request is not placed on both sides merely by rewording it. All call tokens may appear in syntax pretraining, but target arguments are randomly paired with request arguments during that stage.

    Syntax pretraining supplies an initial policy that can attempt calls. GRPO rewards use training-side requests only.
    """)
    return


@app.cell
def _(json):
    rows = []
    for kind, entities, periods, function in [('price', ['AAPL', 'MSFT', 'TSLA'], ['day1', 'day2'], 'get_stock_price'), ('revenue', ['Apple', 'Microsoft', 'Tesla'], ['2024', '2025'], 'get_revenue'), ('convert', ['100', '200', '300'], ['USD_EUR', 'EUR_USD'], 'convert_currency')]:
        for i, entity in enumerate(entities):
            for j, period in enumerate(periods):
                rows.append({'prompt': [kind, entity, period], 'call': [function, entity, period], 'test': (i + j) % 3 == 0})
    rows += [{'prompt': ['poem', 'none', 'none'], 'call': ['no_call', 'none', 'none'], 'test': False}, {'prompt': ['story', 'none', 'none'], 'call': ['no_call', 'none', 'none'], 'test': True}]
    train_rows = [row for row in rows if not row['test']]
    heldout = [row for row in rows if row['test']]
    assert not {tuple(row['prompt']) for row in train_rows} & {tuple(row['prompt']) for row in heldout}

    def render_call(program):
        if len(program) != 4 or program[-1] != '<eos>':
            return 'invalid program'
        name, first, second, _ = program
        if name == 'get_stock_price':
            args = {'ticker': first, 'date': second}
        elif name == 'get_revenue':
            if second not in ['2024', '2025']:
                return 'invalid year'
            args = {'company': first, 'fiscal_year': int(second)}
        elif name == 'convert_currency':
            if first not in ['100', '200', '300'] or second not in ['USD_EUR', 'EUR_USD']:
                return 'invalid conversion'
            source, destination = second.split('_')
            args = {'amount': int(first), 'from_currency': source, 'to_currency': destination}
        elif name == 'no_call' and first == second == 'none':
            args = {}
        else:
            return 'unknown tool'
        return json.dumps({'name': name, 'arguments': args}, sort_keys=True)

    def execute_tool(call):
        name, args = (call['name'], call['arguments'])
        if name == 'get_stock_price':
            return {('AAPL', 'day1'): 10, ('AAPL', 'day2'): 11, ('MSFT', 'day1'): 20, ('MSFT', 'day2'): 21, ('TSLA', 'day1'): 30, ('TSLA', 'day2'): 31}[args['ticker'], args['date']]
        if name == 'get_revenue':
            return {'Apple': 100, 'Microsoft': 200, 'Tesla': 300}[args['company']] + (args['fiscal_year'] - 2024) * 7
        if name == 'convert_currency':
            rate = {('USD', 'EUR'): 0.9, ('EUR', 'USD'): 1.1}[args['from_currency'], args['to_currency']]
            return round(args['amount'] * rate, 2)
        if name == 'no_call' and args == {}:
            return 'no_call'
        raise ValueError('Unknown mock call')

    def score(program, row):
        text = render_call(program)
        gold = json.loads(render_call(row['call'] + ['<eos>']))
        try:
            call = json.loads(text)
            same_schema = call['name'] == gold['name'] and call['arguments'].keys() == gold['arguments'].keys()
            exact = call == gold
            try:
                executed = exact and execute_tool(call) == execute_tool(gold)
            except (KeyError, ValueError, TypeError):
                executed = False
            return [1.0, float(same_schema), float(exact), float(executed)]
        except (ValueError, TypeError, KeyError):
            return [0.0] * 4
    for _row in rows:
        assert score(_row['call'] + ['<eos>'], _row) == [1.0] * 4
    assert score(['unknown', 'none', 'none', '<eos>'], rows[0]) == [0.0] * 4
    assert score(['get_stock_price', 'MSFT', 'day1', '<eos>'], rows[0])[2:] == [0, 0]
    print({'training_requests': len(train_rows), 'heldout_requests': len(heldout), 'sample_prompt': rows[0]['prompt'], 'gold_call': render_call(rows[0]['call'] + ['<eos>'])})
    return heldout, render_call, rows, score, train_rows


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Learn a small autoregressive calling policy

    The vocabulary contains tool names and available arguments. The GRU generates calls token by token, preserving log-probabilities with shape `[batch, token]`.

    The renderer determines representation without reading the gold answer. Wrong arguments can still produce JSON, and the verifier assigns them a lower reward.
    """)
    return


@app.cell
def _(
    SFT_STEPS,
    copy,
    heldout,
    nn,
    np,
    random,
    render_call,
    rows,
    score,
    torch,
    train_rows,
):
    words = ['<pad>', '<sep>', '<eos>'] + sorted({word for row in rows for word in row['prompt'] + row['call']})
    vocab = {word: index for index, word in enumerate(words)}

    class ToolLM(nn.Module):

        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(len(words), 32)
            self.gru = nn.GRU(32, 64, batch_first=True)
            self.head = nn.Linear(64, len(words))

        def forward(self, tokens):
            hidden, _ = self.gru(self.embedding(tokens))
            return self.head(hidden)

    def encode_prompts(batch):
        return torch.tensor([[vocab[w] for w in row['prompt'] + ['<sep>']] for row in batch])

    def token_logp(policy, prompts, responses):
        tokens = torch.cat([prompts, responses], -1)
        logits = policy(tokens[:, :-1])[:, prompts.shape[1] - 1:]
        return logits.log_softmax(-1).gather(-1, responses[..., None]).squeeze(-1)

    @torch.no_grad()  # The prompt has four tokens; prediction position 3 targets the first response word.
    def sample(policy, prompts, greedy=False):
        tokens, responses = (prompts.clone(), [])
        for _ in range(4):
            logits = policy(tokens)[:, -1]
            next_token = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
            responses.append(next_token)
            tokens = torch.cat([tokens, next_token[:, None]], -1)
        return torch.stack(responses, -1)

    def decode(response):
        return [words[i] for i in response.tolist()]
    model = ToolLM()
    _optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(SFT_STEPS):
        _batch = random.choices(train_rows, k=24)
        random_calls = [random.choice([r for r in rows if r['call'][0] == row['call'][0]])['call'] for row in _batch]
        _responses = torch.tensor([[vocab[w] for w in call + ['<eos>']] for call in random_calls])
        _loss = -token_logp(model, encode_prompts(_batch), _responses).mean()
        _optimizer.zero_grad()
        _loss.backward()
        _optimizer.step()
    reference = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def evaluate(policy, dataset=None):
        dataset = heldout if dataset is None else dataset
        response = sample(policy, encode_prompts(dataset), greedy=True)
        details = [score(decode(item), row) for item, row in zip(response, dataset)]
        names = ['rendered_json_rate', 'schema_rate', 'argument_accuracy', 'execution_success']
        metrics = dict(zip(names, np.array(details).mean(0).tolist()))
        return (metrics, [render_call(decode(item)) for item in response])
    before, before_text = evaluate(model)
    before_train = evaluate(model, train_rows)[0]
    print('Before GRPO:', before)
    return (
        before,
        before_text,
        before_train,
        decode,
        encode_prompts,
        evaluate,
        model,
        reference,
        sample,
        token_logp,
        words,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Sample a group per request and update token probabilities

    The reward weights are 0.2 for format, 0.2 for schema, 0.3 for arguments and 0.3 for successful execution. Rewards are standardized separately within each request's group. Equal-reward groups have zero advantages.

    We save the old policy's sampled token probabilities, then perform two clipped updates. The reference policy stays fixed. KL is calculated per token before response-length averaging. Responses have exactly four tokens, so the mask is all ones; a variable-length implementation must exclude padding after EOS.
    """)
    return


@app.cell
def _(
    BETA,
    GROUP,
    UPDATES,
    before,
    before_text,
    before_train,
    decode,
    encode_prompts,
    evaluate,
    heldout,
    model,
    nn,
    np,
    random,
    reference,
    sample,
    score,
    token_logp,
    torch,
    train_rows,
):
    _optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    training_rewards, heldout_curve = ([], [])
    for update in range(UPDATES):
        _batch = random.choices(train_rows, k=8)
        expanded = [row for row in _batch for _ in range(GROUP)]
        prompts = encode_prompts(expanded)
        _responses = sample(model, prompts)
        scores = torch.tensor([score(decode(item), row) for item, row in zip(_responses, expanded)])
        rewards = (scores * torch.tensor([0.2, 0.2, 0.3, 0.3])).sum(-1).reshape(-1, GROUP)
        advantage = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, correction=0, keepdim=True) + 0.0001)
        advantage = advantage.flatten().detach()
        with torch.no_grad():
            old_logp = token_logp(model, prompts, _responses)
            ref_logp = token_logp(reference, prompts, _responses)
        for _ in range(2):
            logp = token_logp(model, prompts, _responses)
            ratio = torch.exp(logp - old_logp)
            clipped = ratio.clamp(0.8, 1.2)
            surrogate = torch.minimum(ratio * advantage[:, None], clipped * advantage[:, None])
            log_ref_ratio = ref_logp - logp
            kl = torch.exp(log_ref_ratio) - log_ref_ratio - 1
            _loss = (-surrogate + BETA * kl).mean(-1).mean()
            _optimizer.zero_grad()
            _loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            _optimizer.step()
        training_rewards.append(float(rewards.mean()))
        if update % 15 == 0:
            heldout_curve.append((update, evaluate(model)[0]['execution_success']))
    after, after_text = evaluate(model)
    after_train = evaluate(model, train_rows)[0]
    assert any((not torch.equal(value, reference.state_dict()[key]) for key, value in model.state_dict().items()))
    assert np.isfinite(training_rewards).all()
    print('Held-out before:', before, '\nHeld-out after:', after)
    print('Training requests before:', before_train, '\nTraining requests after:', after_train)
    for _row, old, new in zip(heldout[:4], before_text, after_text):
        print(_row['prompt'], '\n before:', old, '\n after:', new)
    return after, after_text, after_train, heldout_curve, training_rewards


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Report formatting and execution separately

    A training-reward increase can come from valid formatting alone. Complete held-out execution success requires the correct tool, arguments and result. Incorrect entities remain visible in the examples.

    If training success improves but held-out success does not, the experiment exposes failure to generalize across argument combinations. Two possible explanations are insufficient request coverage and failure to copy request arguments. Adding training-side combinations and adding supervised argument-copying pretraining give separate follow-up comparisons.
    """)
    return


@app.cell
def _(
    OUTPUT,
    after,
    after_text,
    after_train,
    before,
    before_text,
    before_train,
    heldout,
    heldout_curve,
    json,
    model,
    np,
    plt,
    rows,
    torch,
    training_rewards,
    words,
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(training_rewards)
    axes[0].set(xlabel="GRPO update", ylabel="Sampled training reward")
    curve = np.array(heldout_curve)
    axes[1].plot(curve[:, 0], curve[:, 1], marker="o")
    axes[1].set(xlabel="GRPO update", ylabel="Held-out execution success", ylim=(-0.05, 1.05))
    fig.tight_layout()
    fig.savefig(OUTPUT / "training.png", dpi=120)
    plt.show()
    metrics = {"before": before, "after": after,
               "training_before": before_train, "training_after": after_train,
               "examples": [{"prompt": row["prompt"], "before": old, "after": new}
                            for row, old, new in zip(heldout, before_text, after_text)]}
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUTPUT / "dataset.json").write_text(json.dumps(rows, indent=2))
    torch.save({"state_dict": model.state_dict(), "vocabulary": words}, OUTPUT / "tiny-tool-grpo.pt")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checklist

    - [ ] Track format, schema, arguments and execution separately; wrong arguments receive no execution reward.
    - [ ] Perform actual policy updates using old-policy token probabilities and a frozen reference.
    - [ ] Isolate held-out semantic combinations and retain failed calls.

    ## Exercise

    A call parses and has the correct schema, but its arguments and result are wrong. Fill in the total reward. Hint: add only the first two reward weights. Enable `RUN_EXERCISE` after filling `answer`.
    """)
    return


@app.cell
def _():
    RUN_EXERCISE = False
    answer = None
    if RUN_EXERCISE:
        assert abs(answer - 0.4) < 1e-6
    else:
        print("Optional exercise: fill answer and enable RUN_EXERCISE.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [DeepSeekMath](https://arxiv.org/html/2402.03300): the order of group-relative advantages, token-level ratios, clipping and KL aggregation.
    - [Companion financial tool-calling chapter](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter18_grpo/financial-tool-calling-grpo): connect the verifier to a Qwen language model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saved run results

    CPU · Python 3.12.13 · 4.22 s

    [Open the full result page](https://walkinglabs.github.io/hands-on-modern-rl/lab-results/financial-en.html)
    """)
    return


if __name__ == "__main__":
    app.run()
