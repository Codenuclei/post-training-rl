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
    # Separate action selection from action evaluation

    Two actions both have a true value of zero, but their estimates contain noise. Taking the larger estimate on every trial can produce a positive average. We first reproduce this effect, then compare DQN and Double DQN in CartPole.

    The default CPU budget is 15,000 steps for each of two seeds per algorithm. Increase `STEPS` to 80,000 and use five seeds for a longer comparison. Companion source: `code/chapter04_dqn/double_dqn_cartpole.py`.
    """)
    return


@app.cell
def _():
    import importlib.util
    import subprocess
    import sys

    packages = {"numpy": "numpy>=1.26,<3", "matplotlib": "matplotlib>=3.8,<4",
                "torch": "torch>=2.5,<3", "gymnasium": "gymnasium[classic-control]>=1.0,<2",
                "pygame": "pygame>=2.5,<3", "PIL": "pillow>=10,<13"}
    missing = [spec for name, spec in packages.items() if importlib.util.find_spec(name) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

    import collections
    import copy
    import json
    import os
    import random
    from pathlib import Path

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import gymnasium as gym
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn

    torch.set_num_threads(1)
    OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "dqn"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    STEPS, SEEDS, GAMMA = 15000, [0, 1], 0.99
    rng = np.random.default_rng(0)
    selection, evaluation = rng.normal(size=(2, 10000, 2))
    same = selection.max(1).mean()
    separate = evaluation[np.arange(10000), selection.argmax(1)].mean()
    print({"noisy_max": float(same), "separate_evaluation": float(separate)})
    assert same > 0.4 and abs(separate) < 0.05
    return (
        GAMMA,
        OUTPUT,
        SEEDS,
        STEPS,
        collections,
        copy,
        gym,
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
    ## 1. Inspect the bootstrap target

    The network maps observations of shape `[batch, 4]` to action values of shape `[batch, 2]`. DQN uses the target network both to select and to evaluate an action. Double DQN selects with the online network and evaluates that action with the target network.

    `terminated` removes the successor value. A time-limit `truncated` signal ends the sampled episode while retaining the bootstrap term.
    """)
    return


@app.cell
def _(nn, torch):
    def network():
        return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64),
                             nn.ReLU(), nn.Linear(64, 2))


    @torch.no_grad()
    def bootstrap(online_values, target_values, double):
        if double:
            actions = online_values.argmax(-1, keepdim=True)
            return target_values.gather(1, actions).squeeze(1)
        return target_values.max(1).values


    online_test = torch.tensor([[3.0, 1.0]])
    target_test = torch.tensor([[1.0, 4.0]])
    assert bootstrap(online_test, target_test, False).item() == 4
    assert bootstrap(online_test, target_test, True).item() == 1
    print("Toy target test passed: DQN=4, Double DQN=1.")
    return bootstrap, network


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Use matching budgets and independent evaluation

    Both algorithms use the same network size, exploration schedule and evaluation seeds. The replay buffer stores actual transitions. The target network is synchronized every 250 steps.

    Every 1,500 steps, we disable exploration and evaluate in a separate environment. Evaluation episodes never enter the training buffer.
    """)
    return


@app.cell
def _(
    GAMMA,
    SEEDS,
    STEPS,
    bootstrap,
    collections,
    copy,
    gym,
    network,
    nn,
    np,
    random,
    torch,
):
    @torch.no_grad()
    def evaluate(model, seeds=range(1000, 1010)):
        env = gym.make('CartPole-v1')
        returns, estimates = ([], [])
        for seed in seeds:
            state, _ = env.reset(seed=seed)
            estimates.append(float(model(torch.tensor(state)).max()))
            total = 0
            while True:
                action = int(model(torch.tensor(state)).argmax())
                state, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
        env.close()
        return (float(np.mean(returns)), float(np.mean(estimates)))

    def train(double, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        online = network()
        target = copy.deepcopy(online).requires_grad_(False)
        optimizer = torch.optim.Adam(online.parameters(), lr=0.0005)
        buffer = collections.deque(maxlen=20000)
        env = gym.make('CartPole-v1')
        env.action_space.seed(seed)
        state, _ = env.reset(seed=seed)
        log = [(0, *evaluate(online))]
        for step in range(1, STEPS + 1):
            epsilon = max(0.05, 1 - step / (0.6 * STEPS))
            with torch.no_grad():
                action = int(online(torch.tensor(state)).argmax())
            if random.random() < epsilon:
                action = env.action_space.sample()
            nxt, reward, terminated, truncated, _ = env.step(action)
            buffer.append((state, action, reward, nxt, float(terminated)))
            state = nxt
            if terminated or truncated:
                state, _ = env.reset()
            if len(buffer) >= 256 and step % 4 == 0:
                s, a, r, ns, terminal = zip(*random.sample(buffer, 64))
                s, ns = (torch.tensor(np.array(s)), torch.tensor(np.array(ns)))
                q = online(s).gather(1, torch.tensor(a).view(-1, 1)).squeeze(1)
                with torch.no_grad():
                    next_value = bootstrap(online(ns), target(ns), double)
                    y = torch.tensor(r) + GAMMA * (1 - torch.tensor(terminal)) * next_value
                loss = nn.functional.smooth_l1_loss(q, y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10)
                optimizer.step()
            if step % 250 == 0:
                target.load_state_dict(online.state_dict())
            if step % 1500 == 0:
                log.append((step, *evaluate(online)))
        env.close()
        return (online, np.array(log))
    runs = {}
    for _name, double in [('DQN', False), ('Double DQN', True)]:
        runs[_name] = [train(double, seed) for seed in SEEDS]
        print(_name, 'held-out returns:', [float(log[-1, 1]) for _, log in runs[_name]])
    return (runs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Inspect returns, values and an actual replay

    A Q value predicts discounted return. Subtracting it from the undiscounted 500-point episode limit would not measure overestimation. The figures show evaluation rewards, Q estimates and a fixed-seed policy replay.

    The replay always uses Double DQN with seed 0. We do not select the best-looking run from the comparison.
    """)
    return


@app.cell
def _(OUTPUT, gym, json, np, plt, runs, torch):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    metrics = {}
    for _name, samples in runs.items():
        logs = np.stack([log for _, log in samples])
        for column, ax in [(1, axes[0]), (2, axes[1])]:
            mean, std = (logs[:, :, column].mean(0), logs[:, :, column].std(0))
            ax.plot(logs[0, :, 0], mean, label=_name)
            ax.fill_between(logs[0, :, 0], mean - std, mean + std, alpha=0.15)
        metrics[_name] = {'evaluation_returns': logs[:, -1, 1].tolist(), 'initial_state_q': logs[:, -1, 2].tolist()}
    for ax, label in zip(axes, ['Held-out return', 'Initial-state max Q estimate']):
        ax.set(xlabel='Environment steps', ylabel=label)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / 'comparison.png', dpi=120)
    plt.show()
    from IPython.display import Image, display
    from PIL import Image as PILImage
    model = runs['Double DQN'][0][0]
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    state, _ = env.reset(seed=2000)
    frames = []
    for step in range(500):
        if step % 2 == 0:
            frames.append(PILImage.fromarray(env.render()).resize((360, 240)))
        with torch.no_grad():
            action = int(model(torch.tensor(state)).argmax())
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()
    frames[0].save(OUTPUT / 'policy.gif', save_all=True, append_images=frames[1:], duration=40, loop=0)
    display(Image(filename=str(OUTPUT / 'policy.gif')))
    torch.save(model.state_dict(), OUTPUT / 'double-dqn-seed0.pt')
    assert np.isfinite([v for item in metrics.values() for v in item['evaluation_returns']]).all()
    (OUTPUT / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checklist

    - [ ] Locate action selection and evaluation in the two target calculations.
    - [ ] Explain the bootstrap mask for task termination and time limits.
    - [ ] Save independent evaluation curves and a fixed-seed replay, including weak policies.

    ## Exercise

    The online network outputs `[3, 1]`, and the target network outputs `[1, 4]`. Fill in the Double DQN successor value. Hint: first find the action selected by the online network. Enable `RUN_EXERCISE` after filling `answer`.
    """)
    return


@app.cell
def _():
    RUN_EXERCISE = False
    answer = None
    if RUN_EXERCISE:
        assert answer == 1.0
    else:
        print("Optional exercise: fill answer and enable RUN_EXERCISE.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461): the separation of action selection and evaluation and the scope of the original experiments.
    - [Gymnasium Env](https://gymnasium.farama.org/api/env/): the meanings of `terminated` and `truncated`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saved run results

    CPU · Python 3.12.13 · 13.6 s

    [Open the full result page](https://walkinglabs.github.io/hands-on-modern-rl/lab-results/dqn-en.html)
    """)
    return


if __name__ == "__main__":
    app.run()
