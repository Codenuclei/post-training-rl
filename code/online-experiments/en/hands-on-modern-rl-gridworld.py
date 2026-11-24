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
    # From a value table to a path

    We start in the top-left corner. Entering the bottom-right corner earns 1 point; entering the trap at (1, 1) earns −1 point and ends the episode. Every ordinary step costs 0.01 points. Moving into a wall leaves the agent in the same cell, so unnecessary movement has a cost.

    First compute values with known transition rules, then let Q-learning learn through actual transitions. The default CPU experiment uses 30 seeds and 500 episodes per seed. Data is generated locally. The map, rewards and terminal-state convention match `code/chapter03_mdp/gridworld_q_learning.py`.
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

    OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "gridworld"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    START, TRAP, GOAL = (0, 0), (1, 1), (3, 3)
    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    GAMMA, ALPHA, EPISODES, SEEDS = 0.95, 0.15, 500, 30
    print({"start": START, "trap": TRAP, "goal": GOAL, "episodes": EPISODES})
    return (
        ALPHA,
        EPISODES,
        GAMMA,
        GOAL,
        MOVES,
        OUTPUT,
        SEEDS,
        START,
        TRAP,
        json,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Attach the reward to a transition

    The trap or goal reward is paid on entry. A terminal transition has no successor value. For example, moving right from (3, 2) has a target of 1; adding a goal-state value would count the terminal reward twice.
    """)
    return


@app.cell
def _(GAMMA, GOAL, MOVES, START, TRAP, np):
    def transition(state, action):
        if state in (TRAP, GOAL):
            return (state, 0.0, True)
        dr, dc = MOVES[action]
        nxt = (state[0] + dr, state[1] + dc)
        if not (0 <= nxt[0] < 4 and 0 <= nxt[1] < 4):
            nxt = state
        reward = 1.0 if nxt == GOAL else -1.0 if nxt == TRAP else -0.01
        return (nxt, reward, nxt in (TRAP, GOAL))

    def action_values(values, state):
        result = []
        for action in range(4):
            nxt, reward, done = transition(state, action)
            result.append(reward + GAMMA * values[nxt] * (not done))
        return np.array(result)
    values, history = (np.zeros((4, 4)), [])
    for sweep in range(1000):
        history.append(values.copy())
        updated = values.copy()
        for _state in np.ndindex(4, 4):
            if _state not in (TRAP, GOAL):
                updated[_state] = action_values(values, _state).max()
        delta = np.abs(updated - values).max()
        values = updated
        if delta < 1e-12:
            break
    assert transition((3, 2), 3) == (GOAL, 1.0, True)
    assert abs(values[START] - 0.728537125) < 1e-09
    print('Optimal state values:\n', np.round(values, 3))
    return transition, values


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Learn without receiving the solution

    The Q table has shape `[4, 4, 4]`. The last dimension represents up, down, left and right. Each step updates only the state and action just visited. The exploration rate decreases from 1 to 0.05, gradually replacing early exploration with exploitation.
    """)
    return


@app.cell
def _(ALPHA, EPISODES, GAMMA, GOAL, SEEDS, START, np, transition):
    def train(seed):
        rng = np.random.default_rng(seed)
        q = np.zeros((4, 4, 4))
        returns = []
        for episode in range(EPISODES):
            state, total = START, 0.0
            epsilon = 1 - 0.95 * episode / (EPISODES - 1)
            for _ in range(100):
                candidates = np.flatnonzero(np.isclose(q[state], q[state].max()))
                action = int(rng.integers(4) if rng.random() < epsilon else rng.choice(candidates))
                nxt, reward, done = transition(state, action)
                bootstrap = 0.0 if done else q[nxt].max()
                q[state][action] += ALPHA * (reward + GAMMA * bootstrap - q[state][action])
                total += reward
                state = nxt
                if done:
                    break
            returns.append(total)
        return q, returns


    def evaluate(q):
        state, path, total = START, [START], 0.0
        for _ in range(100):
            state, reward, done = transition(state, int(q[state].argmax()))
            path.append(state)
            total += reward
            if done:
                break
        return {"success": state == GOAL, "steps": len(path) - 1, "return": total, "path": path}


    runs = [train(seed) for seed in range(SEEDS)]
    evaluations = [evaluate(q) for q, _ in runs]
    metrics = {"success_rate": float(np.mean([r["success"] for r in evaluations])),
               "mean_steps": float(np.mean([r["steps"] for r in evaluations]))}
    print(metrics)
    print("Seed 0 greedy path:", evaluations[0]["path"])
    assert metrics["success_rate"] == 1.0
    assert metrics["mean_steps"] == 6.0
    return evaluations, metrics, runs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Inspect the policy and learning curve

    The training curve includes exploratory actions. The displayed path comes from a separate greedy evaluation with exploration disabled, so the two returns need not match. The table reports success and evaluation return across 30 seeds.
    """)
    return


@app.cell
def _(
    EPISODES,
    GOAL,
    OUTPUT,
    TRAP,
    evaluations,
    json,
    metrics,
    np,
    plt,
    runs,
    values,
):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].imshow(values, cmap='Blues')
    q = runs[0][0]
    axes[1].imshow(q.max(-1), cmap='Blues')
    arrows = ['↑', '↓', '←', '→']
    for _state in np.ndindex(4, 4):
        row, col = _state
        axes[0].text(col, row, f'{values[_state]:.2f}', ha='center', va='center')
        label = 'X' if _state == TRAP else 'G' if _state == GOAL else arrows[q[_state].argmax()]
        axes[1].text(col, row, label, ha='center', va='center', fontsize=15)
    path = np.array(evaluations[0]['path'])
    axes[1].plot(path[:, 1], path[:, 0], color='orange', alpha=0.6)
    rewards = np.stack([reward for _, reward in runs])
    window = np.ones(20) / 20
    smooth = np.stack([np.convolve(row, window, mode='valid') for row in rewards])
    x = np.arange(19, EPISODES)
    axes[2].plot(x, smooth.mean(0))
    axes[2].fill_between(x, smooth.mean(0) - smooth.std(0), smooth.mean(0) + smooth.std(0), alpha=0.2)
    for ax, title in zip(axes, ['Value iteration', 'Q-learning policy / seed 0', 'Training return']):
        ax.set_title(title)
    axes[2].set(xlabel='Episode', ylabel='Return (20-episode mean)')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'gridworld.png', dpi=120)
    plt.show()
    np.save(OUTPUT / 'q-table.npy', q)
    (OUTPUT / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checklist

    - [ ] Explain why terminal transitions have zero successor value.
    - [ ] Read a successful path from the learned Q table.
    - [ ] Distinguish exploratory training return from greedy evaluation return.

    ## Exercise

    Fill in the TD target when moving right from (3, 2) into the goal. Hint: this transition terminates the episode. Enable `RUN_EXERCISE` after filling `answer`.
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

    - [Companion value experiment](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter03_mdp/value-experiment): the same map and reward convention.
    - [Sutton & Barto, Chapters 4 and 6](http://incompleteideas.net/book/the-book-2nd.html): value iteration with a known model and learning through sampled transitions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saved run results

    CPU · Python 3.12.13 · 2.78 s

    [Open the full result page](https://walkinglabs.github.io/hands-on-modern-rl/lab-results/gridworld-en.html)
    """)
    return


if __name__ == "__main__":
    app.run()
