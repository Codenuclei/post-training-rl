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
    # 查表、计算、再提交答案 / SQL and calculator Agent

    请求要求计算某家公司的费用率或收入增长率。我们创建真正的 SQLite 表，
    让策略查看表结构、查询数据、调用计算器，最后接受数值验证。
    每次工具调用都会增加一步成本，查询不到足够数据就无法计算。

    默认 CPU、两个种子、160 次更新。公司数据为合成数据，动作和 SQL 模板固定。
    这个实验训练有限工具路由策略；教材的 rLLM FinQA 实验继续讨论语言模型生成参数与回答。
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
    import numpy as np
    import torch

    torch.set_num_threads(1)
    OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "agentic-finqa"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    import sqlite3


    def make_task(index):
        previous = 80 + 3 * index
        return {"kind": index % 2, "company": f"company-{index}",
                "previous": previous, "current": previous + 10 + index,
                "expense": 20 + index}


    training = [make_task(i) for i in range(24)]
    heldout = [make_task(i) for i in range(32, 48)]
    assert {t["company"] for t in training}.isdisjoint(t["company"] for t in heldout)
    print(training[0], heldout[0])
    return OUTPUT, heldout, np, sqlite3, torch, training


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 工具读到什么 / Inspect the tool data

    每个任务持有独立的内存数据库，包含同一家公司的两年数据。
    训练集与评估集的公司编号互不相交。问题类型公开可见，标准答案保留在终局验证器中。
    费用率为当年费用除以当年收入，增长率为收入差除以前一年收入。
    """)
    return


@app.cell
def _(sqlite3, training):
    def open_database(task):
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE accounts(company TEXT, year INT, revenue REAL, expense REAL)")
        rows = [(task["company"], 2024, task["previous"], task["expense"] - 2),
                (task["company"], 2025, task["current"], task["expense"])]
        connection.executemany("INSERT INTO accounts VALUES (?, ?, ?, ?)", rows)
        return connection


    with open_database(training[0]) as connection:
        print(connection.execute("PRAGMA table_info(accounts)").fetchall())
        print(connection.execute("SELECT * FROM accounts").fetchall())
    connection.close()
    return (open_database,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 数据进入下一轮观测 / Multi-turn observations

    合法动作只限制工具协议：先发现表和字段，拿到对应数据后才能计算。
    策略仍要选择查询哪一年、执行哪个公式、何时提交。查询使用绑定参数，
    返回值进入观测；只给数据库名称而不给查询结果，策略就无法完成这里的工具轨迹。
    """)
    return


@app.cell
def _(np, open_database, training):
    class FinanceEnvironment:
        actions = ["list_tables", "schema", "query_current", "query_previous",
                   "calculate_ratio", "calculate_growth", "submit"]
        max_turns = 8

        def __init__(self, task):
            self.task, self.connection = task, open_database(task)
            self.stage, self.current, self.previous, self.result = 0, None, None, None
            self.success = False

        def close(self):
            self.connection.close()

        def observe(self):
            state = [float(self.task["kind"] == k) for k in range(2)]
            state += [float(self.stage == k) for k in range(3)]
            state += [float(self.current is not None), float(self.previous is not None),
                      float(self.result is not None)]
            state += list(self.current or [0, 0]) + [self.previous or 0, self.result or 0]
            state[-4:] = [float(value) / 1000 for value in state[-4:]]
            legal = [False] * len(self.actions)
            if self.stage < 2:
                legal[self.stage] = True
            else:
                legal[2], legal[3] = self.current is None, self.previous is None
                legal[4] = self.current is not None
                legal[5] = self.current is not None and self.previous is not None
                legal[6] = self.result is not None
            return state, legal

        def step(self, action):
            assert self.observe()[1][action]
            if action == 0:
                self.stage = 1
                tables = self.connection.execute("SELECT name FROM sqlite_master").fetchall()
                response = {"tables": tables}
            elif action == 1:
                self.stage = 2
                response = {"schema": self.connection.execute("PRAGMA table_info(accounts)").fetchall()}
            elif action in [2, 3]:
                query = "SELECT revenue, expense FROM accounts WHERE company = ? AND year = ?"
                parameters = (self.task["company"], 2025 if action == 2 else 2024)
                row = self.connection.execute(query, parameters).fetchone()
                if action == 2:
                    self.current = row
                else:
                    self.previous = row[0]
                response = {"sql": query, "parameters": parameters, "row": row}
            elif action in [4, 5]:
                revenue, expense = self.current
                self.result = expense / revenue if action == 4 else (revenue / self.previous - 1)
                response = {"calculator_result": self.result}
            else:
                task = self.task
                expected = (task["expense"] / task["current"] if task["kind"] == 0
                            else task["current"] / task["previous"] - 1)
                self.success = bool(np.isclose(self.result, expected, rtol=1e-6, atol=1e-9))
                return float(self.success), True, {"answer": self.result}
            return -0.01, False, response


    demo = FinanceEnvironment(training[1])
    for action in [0, 1, 2, 3, 5, 6]:
        print(demo.actions[action], demo.step(action))
    assert demo.success
    demo.close()
    return (FinanceEnvironment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 固定预算训练与独立评估 / Train and evaluate

    训练每轮采样四个请求，每个请求四条轨迹。REINFORCE 更新小型神经网络，
    终局奖励检查数值答案，工具调用每步扣 0.01。前后评估都采用贪心动作，
    保存成功率、平均步数和逐步 SQL/计算记录。预设接口顺序已经提供一部分流程知识。
    """)
    return


@app.cell
def _(np, torch):
    import random
    from torch import nn

    class AgentPolicy(nn.Module):

        def __init__(self, dimensions, actions):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dimensions, 32), nn.Tanh(), nn.Linear(32, actions))

        def forward(self, observation, legal):
            logits = self.net(torch.tensor(observation, dtype=torch.float32))
            return logits.masked_fill(~torch.tensor(legal, dtype=torch.bool), -1000000000.0)

    def rollout(policy, task, factory, greedy=False, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        env = factory(task)
        logprobs, entropies, trace, rewards = ([], [], [], [])
        for turn in range(env.max_turns):
            observation, legal = env.observe()
            distribution = torch.distributions.Categorical(logits=policy(observation, legal))
            action = distribution.logits.argmax() if greedy else distribution.sample()
            logprobs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            reward, done, response = env.step(int(action))
            rewards.append(reward)
            trace.append({'turn': turn, 'action': env.actions[int(action)], 'observation': response, 'reward': reward})
            if done:
                break
        success = bool(env.success)
        if hasattr(env, 'close'):
            env.close()
        return {'return': float(sum(rewards)), 'success': success, 'trace': trace, 'logprob': torch.stack(logprobs).sum(), 'entropy': torch.stack(entropies).sum()}

    @torch.no_grad()
    def evaluate_agent(policy, tasks, factory, greedy=True):
        samples = [rollout(policy, task, factory, greedy=greedy, seed=10000 + i) for i, task in enumerate(tasks)]
        return {'success_rate': float(np.mean([item['success'] for item in samples])), 'return': float(np.mean([item['return'] for item in samples])), 'mean_turns': float(np.mean([len(item['trace']) for item in samples]))}

    def train_agent(factory, tasks, heldout, seed, updates=160):
        torch.manual_seed(seed)
        rng = random.Random(seed)
        probe = factory(tasks[0])
        policy = AgentPolicy(len(probe.observe()[0]), len(probe.actions))
        if hasattr(probe, 'close'):
            probe.close()
        before = evaluate_agent(policy, heldout, factory)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.003)
        history = []
        for update in range(updates):
            losses = []
            for task in rng.choices(tasks, k=4):
                group = [rollout(policy, task, factory) for _ in range(4)]
                total = sum((item['return'] for item in group))
                for item in group:
                    baseline = (total - item['return']) / (len(group) - 1)
                    advantage = item['return'] - baseline
                    losses.append(-item['logprob'] * advantage - 0.002 * item['entropy'])
            optimizer.zero_grad()
            loss = torch.stack(losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 2)
            optimizer.step()
            if update % 20 == 0 or update == updates - 1:
                history.append({'update': update, 'loss': float(loss.detach()), **evaluate_agent(policy, heldout, factory)})
        after = evaluate_agent(policy, heldout, factory)
        return (policy, before, after, history)

    def run_comparison(factory, training, heldout, output, updates=160):
        results, policies, curves = ([], [], [])
        for seed in [3, 7]:
            policy, before, after, history = train_agent(factory, training, heldout, seed, updates=updates)
            results.append({'seed': seed, 'before': before, 'after': after})
            policies.append(policy)
            curves.append(history)
        for row in results:
            print(row)
        import json
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for row, curve in zip(results, curves):
            ax.plot([point['update'] for point in curve], [point['success_rate'] for point in curve], label=f"seed {row['seed']}")
        ax.set(xlabel='Policy update', ylabel='Held-out greedy success', ylim=(-0.05, 1.05))
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / 'training.png', dpi=120)
        plt.show()
        trace = rollout(policies[0], heldout[0], factory, greedy=True)['trace']
        print(json.dumps(trace, indent=2))
        torch.save(policies[0].state_dict(), output / 'policy-seed3.pt')
        metrics = {'runs': results, 'fixed_heldout_trace': trace, 'train_tasks': len(training), 'heldout_tasks': len(heldout), 'updates': updates, 'algorithm': 'REINFORCE with leave-one-out baseline'}
        assert all((np.isfinite(row['after']['return']) for row in results))
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        return (policies, metrics)
    print('Agent contract: observe → action → tool observation → next decision → verifier')
    return (run_comparison,)


@app.cell
def _(FinanceEnvironment, OUTPUT, heldout, run_comparison, training):
    policies, metrics = run_comparison(FinanceEnvironment, training, heldout, OUTPUT)
    assert any("sql" in step["observation"] for step in metrics["fixed_heldout_trace"])
    print("SQL observations are included in the saved held-out trajectory.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 完成检查 / Checklist
    - [ ] 找到真实 SQL 查询、参数与返回行。
    - [ ] 能区分费用率与收入增长率需要的数据和公式。
    - [ ] 检查独立公司上的成功率与平均工具步数。

    ## 练习 / Exercise
    前一年收入为 100，当年收入为 125。填入增长率。提示：先求差，再除以前一年收入。
    """)
    return


@app.cell
def _():
    RUN_EXERCISE = False
    answer = None
    if RUN_EXERCISE:
        assert abs(answer - 0.25) < 1e-6
    else:
        print("Fill answer, then enable RUN_EXERCISE.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 参考资料 / References
    - [rLLM FinQA 实验](https://walkinglabs.github.io/hands-on-modern-rl/chapter22_agentic/rllm-finqa-lab)：带金融工具的语言模型训练。
    - [Python sqlite3](https://docs.python.org/3/library/sqlite3.html)：本实验实际使用的数据库接口。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 本次运行结果

    CPU · Python 3.12.13 · 6.9 s

    [查看完整结果页](https://walkinglabs.github.io/hands-on-modern-rl/lab-results/agentic-finqa-zh.html)
    """)
    return


if __name__ == "__main__":
    app.run()
