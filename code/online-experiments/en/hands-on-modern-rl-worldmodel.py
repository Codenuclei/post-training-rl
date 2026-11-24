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
    # Learn action consequences, then plan

    A cart starts at position 2.5 with zero velocity. Applying force to the left moves it toward the origin. Continuing that force after reaching the origin sends it past the target.

    We collect random-action trajectories, learn how state and force determine the next step, then compare candidate action sequences. The CPU defaults are 180 trajectories of 30 steps, 900 model updates and 256 candidate sequences with planning horizon 15. No external data is needed. Dynamics and rewards match the book's `minimal_mbrl_point_mass.py`; this experiment adds trajectory-level hold-out and drift checks.
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
    torch.manual_seed(7)
    OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "worldmodel"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    UPDATES, HORIZON, CANDIDATES = 900, 15, 256


    def env_step(state, action):
        force = action[..., 0].clamp(-1, 1)
        position, velocity = state[..., 0], state[..., 1]
        next_velocity = 0.95 * velocity + 0.15 * torch.tanh(force)
        next_position = position + next_velocity
        nxt = torch.stack([next_position, next_velocity], -1)
        reward = -(next_position.square() + 0.1 * next_velocity.square() + 0.001 * force.square())
        return nxt, reward


    episodes = []
    for _ in range(180):
        state = torch.randn(2) * torch.tensor([2.0, 0.5])
        states, actions, targets = [], [], []
        for _ in range(30):
            action = torch.empty(1).uniform_(-1, 1)
            nxt, reward = env_step(state, action)
            states.append(state)
            actions.append(action)
            targets.append(torch.cat([nxt - state, reward[None]]))
            state = nxt
        episodes.append((torch.stack(states), torch.stack(actions), torch.stack(targets)))
    train = [torch.cat([episode[i] for episode in episodes[:140]]) for i in range(3)]
    test = [torch.cat([episode[i] for episode in episodes[140:]]) for i in range(3)]
    print({"train_shapes": [list(x.shape) for x in train],
           "test_shapes": [list(x.shape) for x in test], "split": "whole trajectories"})
    assert train[0].shape == (4200, 2) and test[0].shape == (1200, 2)
    return (
        CANDIDATES,
        HORIZON,
        OUTPUT,
        UPDATES,
        env_step,
        episodes,
        json,
        nn,
        np,
        plt,
        test,
        torch,
        train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Compare a zero-change baseline with a learned model

    The first two target columns are changes in position and velocity; the third is reward. Splitting complete trajectories keeps adjacent frames out of opposite sides of the split and avoids an overly optimistic test error.

    Only training trajectories update the model. Test trajectories are used exclusively to measure prediction error.
    """)
    return


@app.cell
def _(UPDATES, nn, test, torch, train):
    class Dynamics(nn.Module):

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3))

        def forward(self, state, action):
            return self.net(torch.cat([state, action], -1))
    model = Dynamics()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    logs = []
    for update in range(UPDATES):
        _index = torch.randint(len(train[0]), (256,))
        pred = model(train[0][_index], train[1][_index])
        loss = (pred - train[2][_index]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update % 30 == 0:
            with torch.no_grad():
                test_error = (model(test[0], test[1]) - test[2]).square().mean(0)
            logs.append([update, float(loss.detach()), *test_error.tolist()])
    with torch.no_grad():
        one_step_mse = (model(test[0], test[1])[:, :2] - test[2][:, :2]).square().mean().item()
    zero_delta_mse = test[2][:, :2].square().mean().item()
    print({'zero_delta_state_mse': zero_delta_mse, 'learned_state_mse': one_step_mse})
    assert one_step_mse < zero_delta_mse
    return logs, model, one_step_mse, zero_delta_mse


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Inspect open-loop prediction drift

    We keep the actual actions from a test trajectory and repeatedly feed predicted states back into the model. No new true position is supplied during this rollout, exposing how errors accumulate.

    The planner in the next section receives a true observation after every executed action. That feedback makes it different from this open-loop prediction test.
    """)
    return


@app.cell
def _(OUTPUT, episodes, logs, model, np, plt, torch):
    @torch.no_grad()
    def drift(episode):
        states, actions, _ = episode
        imagined = states[0].clone()
        errors = []
        for step in range(len(actions) - 1):
            imagined = imagined + model(imagined, actions[step])[:2]
            errors.append(float((imagined - states[step + 1]).square().mean()))
        return errors
    drift_errors = np.array([drift(episode) for episode in episodes[140:]])
    _fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    logs_1 = np.array(logs)
    axes[0].semilogy(logs_1[:, 0], logs_1[:, 2:4].mean(-1), label='Held-out state MSE')
    axes[0].set(xlabel='Update', ylabel='One-step state MSE')
    axes[1].plot(np.arange(1, 30), drift_errors.mean(0))
    axes[1].set(xlabel='Open-loop prediction horizon', ylabel='Held-out state MSE')
    _fig.tight_layout()
    _fig.savefig(OUTPUT / 'model-errors.png', dpi=120)
    plt.show()
    return (drift_errors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Plan inside the model and execute one real step

    The planner accumulates predicted rewards for each candidate action sequence and executes only the first action of the highest-scoring sequence. It searches again at the next step.

    Random control, zero torque and learned-model planning are evaluated from the same initial states. All reported returns come from the true dynamics. Planning can exploit model errors, so we retain the result for every initial state.
    """)
    return


@app.cell
def _(
    CANDIDATES,
    HORIZON,
    OUTPUT,
    drift_errors,
    env_step,
    json,
    model,
    np,
    one_step_mse,
    plt,
    torch,
    zero_delta_mse,
):
    @torch.no_grad()
    def plan(state):
        sequences = torch.empty(CANDIDATES, HORIZON, 1).uniform_(-1, 1)
        imagined = state[None].expand(CANDIDATES, -1).clone()
        scores = torch.zeros(CANDIDATES)
        for step in range(HORIZON):
            pred = model(imagined, sequences[:, step])
            imagined = imagined + pred[:, :2]
            scores = scores + 0.97 ** step * pred[:, 2]
        return sequences[scores.argmax(), 0]

    @torch.no_grad()
    def rollout(policy, start):
        state, total, path = (start.clone(), 0.0, [start.tolist()])
        for _ in range(40):
            state, reward = env_step(state, policy(state))
            total = total + reward.item()
            path.append(state.tolist())
        return (total, path)
    starts = [torch.tensor([position, speed]) for position, speed in [(2.5, 0), (-2.5, 0), (1.5, 0.3), (-1.5, -0.3), (0.5, 0.2), (-0.5, -0.2)]]
    results = {}
    _fig, ax = plt.subplots(figsize=(7, 3.5))
    for name, policy in [('Random', lambda s: torch.empty(1).uniform_(-1, 1)), ('Zero torque', lambda s: torch.zeros(1)), ('Learned MPC', plan)]:
        samples = []
        for _index, start in enumerate(starts):
            torch.manual_seed(1000 + _index)
            samples.append(rollout(policy, start))
        results[name] = [sample[0] for sample in samples]
        ax.plot(np.array(samples[0][1])[:, 0], label=name)
        print(name, 'true returns:', np.round(results[name], 1).tolist())
    ax.axhline(0, color='grey', ls='--')
    ax.set(xlabel='Real environment step', ylabel='Position / fixed start [2.5, 0]')
    ax.legend()
    _fig.tight_layout()
    _fig.savefig(OUTPUT / 'planning.png', dpi=120)
    plt.show()
    assert np.isfinite([value for values in results.values() for value in values]).all()
    metrics = {'test_state_mse': one_step_mse, 'zero_delta_state_mse': zero_delta_mse, 'open_loop_mse_at_29': float(drift_errors[:, -1].mean()), 'true_returns': results}
    (OUTPUT / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    torch.save(model.state_dict(), OUTPUT / 'dynamics.pt')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checklist

    - [ ] Split complete trajectories and report independent one-step error and multi-step drift.
    - [ ] Distinguish the dynamics model, planner and real environment.
    - [ ] Compare policies from matched initial states and measure returns with the true dynamics.

    ## Exercise

    Fill in the number of actions executed after each planning search. Hint: execute one action and receive a new true state before planning again. Enable `RUN_EXERCISE` after filling `answer`.
    """)
    return


@app.cell
def _():
    RUN_EXERCISE = False
    answer = None
    if RUN_EXERCISE:
        assert answer == 1
    else:
        print("Optional exercise: fill answer and enable RUN_EXERCISE.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Companion Model-Based RL chapter](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter28_vla/embodied-intelligence/model-based-rl/): the roles of learned dynamics, model error and model predictive control.
    - [PETS](https://arxiv.org/abs/1805.12114): further reading on dynamics uncertainty; this experiment uses one deterministic model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saved run results

    CPU · Python 3.12.13 · 4.23 s

    [Open the full result page](https://walkinglabs.github.io/hands-on-modern-rl/lab-results/worldmodel-en.html)
    """)
    return


if __name__ == "__main__":
    app.run()
