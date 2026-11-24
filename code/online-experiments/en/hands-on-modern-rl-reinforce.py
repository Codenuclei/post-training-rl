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
    # Subtract a state baseline from the return

    Receiving 100 points after an action is not enough to decide whether that action deserves encouragement. If the same state normally leads to 120 points, the outcome is below expectation. If it normally leads to 60 points, the outcome is above expectation.

    We learn this state-dependent expectation with a value network and compare updates based on raw returns with updates based on return minus expectation. The default CPU budget is two seeds and 192 episodes per method. Increase it to five seeds and 960 episodes for a longer experiment. Companion source: `code/chapter05_policy_gradient/reinforce_with_baseline.py`.
    """)
    return


@app.cell
def _():
    import importlib.util
    import subprocess
    import sys

    packages = {"torch": "torch>=2.5,<3", "numpy": "numpy>=1.26,<3",
                "matplotlib": "matplotlib>=3.8,<4", "PIL": "pillow>=10,<13",
                "gymnasium": "gymnasium[classic-control]>=1.0,<2", "pygame": "pygame>=2.5,<3"}
    missing = [spec for name, spec in packages.items() if importlib.util.find_spec(name) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

    import json
    import os
    from pathlib import Path

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import gymnasium as gym
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn

    torch.set_num_threads(1)
    OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "reinforce"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    EPISODES, SEEDS, BATCH_EPISODES = 192, [0, 1], 8
    print({"episodes": EPISODES, "seeds": SEEDS, "batch_episodes": BATCH_EPISODES})
    return (
        BATCH_EPISODES,
        EPISODES,
        OUTPUT,
        SEEDS,
        gym,
        json,
        nn,
        np,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Check the expected gradient and its variance

    The baseline must not depend on the action sampled at the current state. At a fixed state, the policy score has zero expectation over actions. Subtracting a state baseline therefore leaves the expected gradient unchanged.

    A useful baseline can reduce variance; an arbitrary baseline need not. We enumerate a two-action example to compare both the mean and variance of the gradient.
    """)
    return


@app.cell
def _(np):
    p = 0.3
    probabilities = np.array([1 - p, p])
    scores = np.array([-p, 1 - p])
    action_returns = np.array([4.0, 6.0])
    variance = {}
    for baseline in [0.0, 5.0, 20.0]:
        gradients = scores * (action_returns - baseline)
        mean = np.sum(probabilities * gradients)
        var = np.sum(probabilities * (gradients - mean) ** 2)
        variance[baseline] = float(var)
        print({"baseline": baseline, "expected_gradient": float(mean), "variance": float(var)})
        assert abs(mean - 0.42) < 1e-10
    assert variance[5.0] < variance[0.0] < variance[20.0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Learn from complete episodes

    The policy outputs logits for left and right, while the value network outputs one state value. Both methods divide returns by 100 to use the same scale. We update after eight complete episodes, summing the policy loss within each episode and averaging across episodes.

    `detach()` prevents the policy loss from updating the critic through the advantage. The critic has its own regression objective.
    """)
    return


@app.cell
def _(BATCH_EPISODES, EPISODES, SEEDS, gym, nn, np, torch):
    def make_network(outputs):
        return nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, outputs))

    def collect(policy, env):
        state, _ = env.reset()
        states, actions, rewards = ([], [], [])
        while True:
            with torch.no_grad():
                action = torch.distributions.Categorical(logits=policy(torch.tensor(state))).sample()
            states.append(state)
            actions.append(int(action))
            state, reward, terminated, truncated, _ = env.step(int(action))
            rewards.append(reward)
            if terminated or truncated:
                break
        returns, running = ([], 0.0)
        for reward in reversed(rewards):
            running = reward + 0.99 * running
            returns.insert(0, running / 100)
        tensors = (torch.tensor(np.array(states)), torch.tensor(actions), torch.tensor(returns))
        return (*tensors, sum(rewards))

    @torch.no_grad()
    def evaluate(policy):
        env = gym.make('CartPole-v1')
        result = []
        for seed in range(1000, 1010):
            state, _ = env.reset(seed=seed)
            total = 0
            while True:
                action = int(policy(torch.tensor(state)).argmax())
                state, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            result.append(total)
        env.close()
        return float(np.mean(result))

    def train(use_baseline, seed):
        torch.manual_seed(seed)
        policy, critic = (make_network(2), make_network(1))
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.003)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.003)
        env = gym.make('CartPole-v1')
        env.reset(seed=seed)
        returns, logs, gradient_norms = ([], [(0, evaluate(policy))], [])
        for first in range(0, EPISODES, BATCH_EPISODES):
            batch = [collect(policy, env) for _ in range(BATCH_EPISODES)]
            losses = []
            for states, actions, targets, total in batch:
                baseline = critic(states).squeeze(1) if use_baseline else torch.zeros_like(targets)
                advantage = (targets - baseline).detach()
                logp = torch.distributions.Categorical(logits=policy(states)).log_prob(actions)
                losses.append(-(logp * advantage).sum())
                returns.append(total)
            optimizer.zero_grad()
            torch.stack(losses).mean().backward()
            norm = nn.utils.clip_grad_norm_(policy.parameters(), 10)
            gradient_norms.append(float(norm))
            optimizer.step()
            if use_baseline:
                states = torch.cat([row[0] for row in batch])
                targets = torch.cat([row[2] for row in batch])
                for _ in range(5):
                    critic_optimizer.zero_grad()
                    loss = nn.functional.mse_loss(critic(states).squeeze(1), targets)
                    loss.backward()
                    critic_optimizer.step()
            if (first + BATCH_EPISODES) % 48 == 0:
                logs.append((first + BATCH_EPISODES, evaluate(policy)))
        env.close()
        return (policy, np.array(returns), np.array(logs), gradient_norms)
    runs = {name: [train(enabled, seed) for seed in SEEDS] for name, enabled in [('REINFORCE', False), ('With baseline', True)]}
    for _name, _samples in runs.items():
        print(_name, 'held-out return:', [float(row[2][-1, 1]) for row in _samples])
    return (runs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Compare curves and policy replays

    During training, the policy, visited-state distribution and returns all change. Fluctuating gradient norms therefore do not isolate gradient variance under a fixed policy. The earlier enumerated example isolates the baseline's effect.

    We retain both seeds' independent evaluation results even when a short training run shows no clear advantage.
    """)
    return


@app.cell
def _(OUTPUT, gym, json, np, plt, runs, torch):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    metrics = {}
    for _name, _samples in runs.items():
        logs = np.stack([sample[2] for sample in _samples])
        axes[0].plot(logs[0, :, 0], logs[:, :, 1].mean(0), label=_name)
        axes[0].fill_between(logs[0, :, 0], logs[:, :, 1].min(0), logs[:, :, 1].max(0), alpha=0.15)
        norms = np.array([sample[3] for sample in _samples])
        axes[1].plot(norms.mean(0), label=_name)
        metrics[_name] = {'heldout_returns': logs[:, -1, 1].tolist()}
    axes[0].set(xlabel='Training episodes', ylabel='Held-out return (seed range)')
    axes[1].set(xlabel='Update', ylabel='Gradient norm before clipping')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / 'comparison.png', dpi=120)
    plt.show()
    from IPython.display import Image, display
    from PIL import Image as PILImage
    policy = runs['With baseline'][0][0]
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    state, _ = env.reset(seed=2000)
    frames = []
    for step in range(500):
        if step % 2 == 0:
            frames.append(PILImage.fromarray(env.render()).resize((360, 240)))
        with torch.no_grad():
            action = int(policy(torch.tensor(state)).argmax())
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()
    frames[0].save(OUTPUT / 'policy.gif', save_all=True, append_images=frames[1:], duration=40, loop=0)
    display(Image(filename=str(OUTPUT / 'policy.gif')))
    torch.save(policy.state_dict(), OUTPUT / 'policy-seed0.pt')
    assert all((np.isfinite(row[3]).all() for samples in runs.values() for row in samples))
    (OUTPUT / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checklist

    - [ ] Explain by enumeration why a state baseline preserves the expected gradient.
    - [ ] Explain why an excessively large baseline can increase variance.
    - [ ] Distinguish fixed-policy variance from changing gradient norms during training.

    ## Exercise

    The return is 100 and the state baseline is 120. Fill in the unscaled advantage. Hint: subtract the baseline from the return. Enable `RUN_EXERCISE` after filling `answer`.
    """)
    return


@app.cell
def _():
    RUN_EXERCISE = False
    answer = None
    if RUN_EXERCISE:
        assert answer == -20
    else:
        print("Optional exercise: fill answer and enable RUN_EXERCISE.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Sutton & Barto, Chapter 13](http://incompleteideas.net/book/the-book-2nd.html): episodic REINFORCE updates and action-independent state baselines.
    - [Companion baseline experiment](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter08_policy_gradient/baseline-experiment): examine how value-fitting errors affect policy updates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saved run results

    CPU · Python 3.12.13 · 5.78 s

    [Open the full result page](https://walkinglabs.github.io/hands-on-modern-rl/lab-results/reinforce-en.html)
    """)
    return


if __name__ == "__main__":
    app.run()
