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
    # Exploration and regret

    The first machine pays one point with probability 0.45; the second pays with probability 0.55. On the first attempt, the first machine may win while the second loses. Always choosing the machine with the highest observed average can therefore lock us into the worse choice.

    We compare random selection, greedy selection, ε-greedy and UCB, then swap the machines' probabilities to examine the limits of old observations. The default CPU experiment uses 50 seeds and 2,000 steps per seed. All data is generated locally. Companion source: `code/chapter03_mdp/two_armed_bandit.py`.
    """)
    return


@app.cell
def _():
    import importlib.util
    import subprocess
    import sys

    packages = {"numpy": "numpy>=1.26,<3", "matplotlib": "matplotlib>=3.8,<4"}
    missing = [spec for name, spec in packages.items() if importlib.util.find_spec(name) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

    import json
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "bandit"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SEEDS, STEPS = 50, 2000
    print({"numpy": np.__version__, "seeds": SEEDS, "steps": STEPS, "device": "CPU"})
    return OUTPUT, SEEDS, STEPS, json, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Inspect one decision

    Only the environment and evaluator know the true payout probabilities. The policy sees the rewards it has collected. **Pseudo-regret** subtracts the chosen machine's expected reward from the best machine's expected reward. It separates the cost of a decision from random payout noise.
    """)
    return


@app.cell
def _(STEPS, np):
    def run_bandit(method, seed, switch=False):
        # Fix potential rewards for every step and arm to compare matching seeds fairly.
        rng = np.random.default_rng(seed)
        potential_rewards = rng.random((STEPS, 2))
        choices = np.random.default_rng(seed + 10000)
        counts, values = np.zeros(2), np.zeros(2)
        rewards, regret, actions = [], [], []
        for step in range(STEPS):
            probabilities = np.array([0.45, 0.55])
            if switch and step >= STEPS // 2:
                probabilities = probabilities[::-1]
            if step < 2:
                action = step
            elif method == "random":
                action = int(choices.integers(2))
            elif method == "epsilon" and choices.random() < 0.1:
                action = int(choices.integers(2))
            else:
                scores = values.copy()
                if method == "ucb":
                    scores += np.sqrt(2 * np.log(step + 1) / counts)
                best = np.flatnonzero(scores == scores.max())
                action = int(choices.choice(best))
            reward = float(potential_rewards[step, action] < probabilities[action])
            counts[action] += 1
            values[action] += (reward - values[action]) / counts[action]
            rewards.append(reward)
            regret.append(probabilities.max() - probabilities[action])
            actions.append(action)
        assert counts.sum() == STEPS
        return np.array(rewards), np.array(regret), np.array(actions)

    sample = run_bandit("greedy", 0)
    assert set(np.unique(sample[0])).issubset({0, 1})
    assert np.all(sample[1] >= 0)
    print("First 12 actions:", sample[2][:12].tolist())
    print("First 12 rewards:", sample[0][:12].tolist())
    return (run_bandit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Compare multiple seeds

    ε-greedy reserves 10% of its decisions for further exploration. UCB adds an uncertainty bonus to actions with fewer observations. We plot mean cumulative reward and mean cumulative pseudo-regret; the shaded region is the standard deviation across seeds.
    """)
    return


@app.cell
def _(OUTPUT, SEEDS, STEPS, json, np, plt, run_bandit):
    methods = ['random', 'greedy', 'epsilon', 'ucb']
    runs = {name: [run_bandit(name, seed) for seed in range(SEEDS)] for name in methods}
    _fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    summary = {}
    for _name, samples in runs.items():
        totals = np.cumsum(np.stack([run[0] for run in samples]), axis=1)
        regrets = np.cumsum(np.stack([run[1] for run in samples]), axis=1)
        axes[0].plot(totals.mean(0), label=_name)
        mean, std = (regrets.mean(0), regrets.std(0))
        axes[1].plot(mean, label=_name)
        axes[1].fill_between(np.arange(STEPS), mean - std, mean + std, alpha=0.12)
        summary[_name] = {'reward': float(totals[:, -1].mean()), 'regret': float(regrets[:, -1].mean())}
    for _ax, title in zip(axes, ['Cumulative reward', 'Pseudo-regret (mean ± seed std)']):
        _ax.set(title=title, xlabel='Step')
        _ax.legend()
    _fig.tight_layout()
    _fig.savefig(OUTPUT / 'stationary.png', dpi=120)
    plt.show()
    print(json.dumps(summary, indent=2))
    return methods, summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Change the environment

    After step 1,000, the machines exchange payout probabilities. Greedy selection and this UCB implementation use all past observations. Their regret after the switch shows how old evidence can slow adaptation.

    A follow-up experiment can keep only a recent observation window. Use the same seeds and change point when comparing it with the original policy.
    """)
    return


@app.cell
def _(OUTPUT, SEEDS, STEPS, json, methods, np, plt, run_bandit, summary):
    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    for _name in methods:
        regret = np.stack([run_bandit(_name, seed, switch=True)[1] for seed in range(SEEDS)])
        _ax.plot(np.cumsum(regret, axis=1).mean(0), label=_name)
    _ax.axvline(STEPS // 2, ls='--', color='grey', label='Probabilities switch')
    _ax.set(xlabel='Step', ylabel='Cumulative pseudo-regret')
    _ax.legend()
    _fig.tight_layout()
    _fig.savefig(OUTPUT / 'nonstationary.png', dpi=120)
    plt.show()
    (OUTPUT / 'metrics.json').write_text(json.dumps(summary, indent=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checklist

    - [ ] Distinguish a single payout, average reward and expected regret.
    - [ ] Explain why greedy policies can behave differently across seeds.
    - [ ] Identify the limitation of a full-history average after a distribution change.

    ## Exercise

    Fill `answer` with the expected total rewards from choosing a machine with probability 0.55 ten times and a machine with probability 0.45 ten times. Hint: multiply the number of selections by the payout probability. Set `RUN_EXERCISE=True` after filling the answer.
    """)
    return


@app.cell
def _(np):
    RUN_EXERCISE = False
    answer = None
    if RUN_EXERCISE:
        assert np.allclose(answer, [5.5, 4.5])
    else:
        print("Exercise is optional; set RUN_EXERCISE=True after filling answer.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Sutton & Barto, Chapter 2](http://incompleteideas.net/book/the-book-2nd.html): the reasoning behind ε-greedy, optimistic initialization and uncertainty-based exploration.
    - [Companion bandit chapter](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter03_mdp/bandit): connect single-action rewards to the later policy-gradient experiments.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saved run results

    CPU · Python 3.12.13 · 5.97 s

    [Open the full result page](https://walkinglabs.github.io/hands-on-modern-rl/lab-results/bandit-en.html)
    """)
    return


if __name__ == "__main__":
    app.run()
