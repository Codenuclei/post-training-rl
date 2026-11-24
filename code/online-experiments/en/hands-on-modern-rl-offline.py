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
    # Evaluate actions beyond a fixed dataset

    Suppose the dataset contains only actions near zero for a particular state. A value network may still assign a high value to action 1. Action 1 lies outside the observed data coverage.

    We fix one dataset and compare direct Q maximization, a conservative penalty and a behavior constraint. The CPU defaults are three seeds, 600 updates per Q network and 400 updates per policy. All data is synthetic.

    This is a single-step contextual bandit. We compare a CQL-style penalty and a behavior constraint. Companion source: `minimal_offline_rl_contextual_bandit.py`.
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
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn

    torch.set_num_threads(1)
    OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "offline"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SEEDS, Q_STEPS, POLICY_STEPS = [0, 1, 2], 600, 400


    def target_action(state):
        return 0.35 * torch.sin(2.5 * state)


    def behavior(state):
        return 0.2 * target_action(state)


    def reward_fn(state, action):
        return -(action - target_action(state)).square()


    torch.manual_seed(2)
    states = torch.empty(6000, 1).uniform_(-1, 1)
    actions = (behavior(states) + 0.04 * torch.randn_like(states)).clamp(-1, 1)
    rewards = reward_fn(states, actions)
    # After fixing the data, training uses these arrays without querying the reward function.
    train = (states[:4800], actions[:4800], rewards[:4800])
    test = (states[4800:], actions[4800:], rewards[4800:])
    print({"train_samples": len(train[0]), "test_samples": len(test[0]),
           "observed_action_range": [float(actions.min()), float(actions.max())]})
    return (
        OUTPUT,
        POLICY_STEPS,
        Q_STEPS,
        SEEDS,
        actions,
        behavior,
        json,
        nn,
        np,
        plt,
        reward_fn,
        states,
        target_action,
        test,
        torch,
        train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Inspect coverage before fitting values

    States cover −1 to 1, while actions occupy a narrow band. The conservative penalty compares Q values for random actions with Q values for observed actions, discouraging unsupported high estimates.

    For each seed, both Q networks start from the same parameters and use the same training-sample indices.
    """)
    return


@app.cell
def _(POLICY_STEPS, Q_STEPS, SEEDS, nn, np, test, torch, train):
    class QNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64),
                                     nn.ReLU(), nn.Linear(64, 1))

        def forward(self, state, action):
            return self.net(torch.cat([state, action], -1))


    def train_q(seed, alpha):
        torch.manual_seed(seed)
        q = QNetwork()
        optimizer = torch.optim.Adam(q.parameters(), lr=5e-4)
        generator = torch.Generator().manual_seed(2000 + seed)
        logs = []
        for step in range(Q_STEPS):
            index = torch.randint(len(train[0]), (128,), generator=generator)
            state, action, reward = [item[index] for item in train]
            q_data = q(state, action)
            fit_loss = (q_data - reward).square().mean()
            # Uniform-action log-mean-exp; the constant has no effect on Q gradients.
            candidates = torch.rand(128, 10, 1, generator=generator) * 2 - 1
            repeated = state[:, None].expand(-1, 10, -1).reshape(-1, 1)
            q_random = q(repeated, candidates.reshape(-1, 1)).reshape(128, 10)
            penalty = (torch.logsumexp(q_random, -1) - np.log(10)).mean() - q_data.mean()
            loss = fit_loss + alpha * penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 30 == 0:
                with torch.no_grad():
                    mse = (q(test[0], test[1]) - test[2]).square().mean().item()
                logs.append((step, mse))
        return q.requires_grad_(False), logs


    def train_actor(q, seed, bc_weight):
        torch.manual_seed(seed + 100)
        actor = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 1), nn.Tanh())
        optimizer = torch.optim.Adam(actor.parameters(), lr=5e-4)
        generator = torch.Generator().manual_seed(seed + 3000)
        for _ in range(POLICY_STEPS):
            index = torch.randint(len(train[0]), (128,), generator=generator)
            state, data_action = train[0][index], train[1][index]
            chosen = actor(state)
            loss = -q(state, chosen).mean() + bc_weight * (chosen - data_action).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return actor


    runs = []
    for seed in SEEDS:
        naive_q, naive_log = train_q(seed, 0.0)
        conservative_q, conservative_log = train_q(seed, 0.2)
        runs.append({"naive_q": naive_q, "conservative_q": conservative_q,
                     "naive_log": naive_log, "conservative_log": conservative_log,
                     "Naive Q": train_actor(naive_q, seed, 0),
                     "Conservative": train_actor(conservative_q, seed, 0),
                     "Q + BC": train_actor(naive_q, seed, 3)})
    assert len(runs) == 3
    print("Trained three seeds without collecting additional environment rewards.")
    return (runs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Separate fitting error from actual policy return

    Held-out data MSE still measures only the behavior policy's neighborhood. We now freeze the models and use the known simulator to evaluate new policy actions. Evaluation values never feed back into the optimizer.

    The oracle provides an upper bound only during evaluation; its actions are not supplied to the learner. Conservative methods can also be too restrictive, so we report both distance from data support and actual return.
    """)
    return


@app.cell
def _(behavior, json, np, reward_fn, runs, torch):
    grid = torch.linspace(-1, 1, 1000)[:, None]
    metrics = {}
    with torch.no_grad():
        for _name in ['Naive Q', 'Conservative', 'Q + BC']:
            policy_actions = [run[_name](grid) for run in runs]
            returns = [float(reward_fn(grid, action).mean()) for action in policy_actions]
            gap = [float((action - behavior(grid)).abs().mean()) for action in policy_actions]
            metrics[_name] = {'true_returns': returns, 'support_gaps': gap}
        metrics['Behavior'] = {'true_return': float(reward_fn(grid, behavior(grid)).mean())}
        metrics['Oracle'] = {'true_return': 0.0}
    print(json.dumps(metrics, indent=2))
    assert np.isfinite([value for name in ['Naive Q', 'Conservative', 'Q + BC'] for value in metrics[name]['true_returns']]).all()
    return grid, metrics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Inspect out-of-distribution estimates

    The first plot shows dataset coverage. The second fixes a state and compares true rewards with both Q estimates. The third shows learned policies. Plots use seed 0, while the table retains all three seeds.

    True reward curves are final diagnostics, never training inputs. Change data noise or initialization to compare the resulting policies.
    """)
    return


@app.cell
def _(
    OUTPUT,
    actions,
    behavior,
    grid,
    json,
    metrics,
    np,
    plt,
    reward_fn,
    runs,
    states,
    target_action,
    torch,
):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].scatter(states[::10], actions[::10], s=3, alpha=0.25, label='Fixed dataset')
    axes[0].plot(grid, target_action(grid), color='black', label='Oracle')
    axes[0].set(xlabel='State', ylabel='Action', ylim=(-1, 1))
    candidate_actions = torch.linspace(-1, 1, 300)[:, None]
    probe_state = torch.full_like(candidate_actions, 0.5)
    with torch.no_grad():
        for key, label in [('naive_q', 'Naive Q'), ('conservative_q', 'Conservative Q')]:
            axes[1].plot(candidate_actions, runs[0][key](probe_state, candidate_actions), label=label)
        axes[1].plot(candidate_actions, reward_fn(probe_state, candidate_actions), 'k--', label='True')
        for _name in ['Naive Q', 'Conservative', 'Q + BC']:
            axes[2].plot(grid, runs[0][_name](grid), label=_name)
    axes[1].set(xlabel='Action at state=0.5', ylabel='Estimated Q / actual reward')
    axes[2].plot(grid, behavior(grid), 'k--', label='Behavior mean')
    axes[2].set(xlabel='State', ylabel='Policy action', ylim=(-1, 1))
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTPUT / 'coverage-and-policy.png', dpi=120)
    plt.show()
    fig, ax = plt.subplots(figsize=(6, 3))
    for key in ['naive_log', 'conservative_log']:
        logs = np.array([run[key] for run in runs])
        ax.semilogy(logs[0, :, 0], logs[:, :, 1].mean(0), label=key)
    ax.set(xlabel='Update', ylabel='Held-out behavior-data MSE')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / 'fit-errors.png', dpi=120)
    plt.show()
    (OUTPUT / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    torch.save({name: runs[0][name].state_dict() for name in ['Naive Q', 'Conservative', 'Q + BC']}, OUTPUT / 'policies-seed0.pt')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checklist

    - [ ] Avoid querying rewards for new actions after fixing the training dataset.
    - [ ] Report behavior-data fitting error, support distance and independent policy return.
    - [ ] Explain how conservative penalties and behavior constraints change the policy.

    ## Exercise

    Observed actions are close to zero. Fill in whether these data directly validate `Q(state, action=1)`. Hint: producing a number does not establish observational support for it. Enable `RUN_EXERCISE` after filling `answer`.
    """)
    return


@app.cell
def _():
    RUN_EXERCISE = False
    answer = None
    if RUN_EXERCISE:
        assert answer is False
    else:
        print("Optional exercise: fill answer and enable RUN_EXERCISE.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Conservative Q-Learning](https://arxiv.org/abs/2006.04779): why suppressing unsupported high values can help offline policy learning.
    - [TD3+BC](https://arxiv.org/abs/2106.06860): behavior constraints limit policy deviation from data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saved run results

    CPU · Python 3.12.13 · 6.64 s

    [Open the full result page](https://walkinglabs.github.io/hands-on-modern-rl/lab-results/offline-en.html)
    """)
    return


if __name__ == "__main__":
    app.run()
