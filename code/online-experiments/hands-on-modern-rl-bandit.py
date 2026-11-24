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
    # 从两台老虎机开始 / Exploration and regret

    第一台机器每次有 0.45 的概率给出 1 分，第二台有 0.55 的概率给出 1 分。
    只试一次时，第一台可能恰好中奖，第二台可能恰好落空。一直选择眼前平均分最高的机器，
    就可能错过长期更好的选择。

    我们先比较随机、贪心、ε-greedy 和 UCB，再把两台机器的胜率调换，检查旧经验的边界。
    数据由代码生成，默认使用 CPU，50 个种子，每个种子 2,000 步；完整运行无需外部数据。
    配套源码：`code/chapter03_mdp/two_armed_bandit.py`。
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
    ## 1. 先检查一次选择 / One decision

    机器的真实胜率只交给环境和评估器，策略只能看到已经获得的奖励。
    **遗憾**衡量的是“最佳机器的期望奖励减去本次所选机器的期望奖励”，不把中奖噪声算成策略错误。
    """)
    return


@app.cell
def _(STEPS, np):
    def run_bandit(method, seed, switch=False):
        # 固定每一步每台机器的潜在奖励，便于同种子公平比较。
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
    ## 2. 一次幸运不能代表策略可靠 / Multiple seeds

    ε-greedy 留出 10% 的机会继续尝试。UCB 则给试得少的动作加上一个不确定性奖励。
    我们同时画平均累计收益和平均累计遗憾，阴影表示种子之间的标准差。
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
    ## 3. 环境改变后，旧平均值会拖慢适应 / A changing environment

    第 1,000 步以后，两台机器的胜率互换。这里的 UCB 和贪心使用全部历史，
    我们可以从分段遗憾看到“更多数据”也会保留过时经验。
    如要改进，可以只统计最近窗口的数据；比较时仍要使用相同的种子和变化时刻。
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
    ## 完成检查 / Checklist
    - [ ] 能区分一次中奖、平均收益和期望遗憾。
    - [ ] 能解释不同随机种子下贪心策略可能出现的分歧。
    - [ ] 能指出全历史平均值在变化环境中的限制。

    ## 练习 / Exercise
    将 `answer` 改为两个数，分别表示胜率 0.55 和 0.45 的机器各选择 10 次时的期望总收益。
    提示：把选择次数乘以胜率；填写后再将 `RUN_EXERCISE` 改为 `True`。
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
    ## 参考资料 / References
    - [Sutton & Barto, Chapter 2](http://incompleteideas.net/book/the-book-2nd.html)：
      比较 ε-greedy、乐观初始化和不确定性探索的出发点。
    - [本书多臂老虎机章节](https://walkinglabs.github.io/hands-on-modern-rl/chapter03_mdp/bandit)：
      将一次动作的收益连接到后续策略梯度实验。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 本次运行结果

    CPU · Python 3.12.13 · 5.13 s

    [查看完整结果页](https://walkinglabs.github.io/hands-on-modern-rl/lab-results/bandit-zh.html)
    """)
    return


if __name__ == "__main__":
    app.run()
