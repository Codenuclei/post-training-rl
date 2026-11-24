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
    # Repair code, then verify it with tests

    We repair subtraction in `solve(x, y)` into addition or multiplication. The policy reads the specification, selects a patch, invokes Python to run public tests, then submits or revises. Submission triggers separate tests, so memorizing one public example cannot earn full reward.

    The default uses a CPU, two seeds and 160 updates. The policy selects among three trusted patch templates to study multi-turn test feedback. The book's rLLM recipe covers freely generated code and language-model training.
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
    OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "agentic-code"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    from functools import lru_cache
    import tempfile

    training = [{"kind": i % 2, "x": i + 2, "y": i % 5 + 2} for i in range(24)]
    heldout = [{"kind": i % 2, "x": i + 40, "y": i % 7 + 2} for i in range(16)]
    print({"train_tasks": len(training), "heldout_tasks": len(heldout)})
    return (
        OUTPUT,
        Path,
        heldout,
        json,
        lru_cache,
        np,
        subprocess,
        sys,
        tempfile,
        torch,
        training,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Execute the tests

    The tool writes a candidate function into a temporary file and runs assertions in a separate Python process. Identical source/test pairs are cached to avoid repeated process startup; first results always come from execution.

    Only our fixed trusted templates are executed here. Candidate code comes from three templates and runs with a two-second timeout.
    """)
    return


@app.cell
def _(Path, lru_cache, subprocess, sys, tempfile, training):
    @lru_cache(maxsize=4096)
    def python_tests(source, cases):
        assertions = [f"assert solve({x}, {y}) == {answer}" for x, y, answer in cases]
        program = source + "\n" + "\n".join(assertions) + "\nprint('passed')\n"
        with tempfile.TemporaryDirectory(prefix="agent-test-") as folder:
            path = Path(folder) / "candidate.py"
            path.write_text(program)
            try:
                result = subprocess.run([sys.executable, "-I", "-S", str(path)],
                                        capture_output=True, text=True, timeout=2)
                return {"passed": result.returncode == 0, "exit_code": result.returncode,
                        "output": (result.stdout + result.stderr)[-300:]}
            except subprocess.TimeoutExpired:
                return {"passed": False, "exit_code": -1, "output": "timeout"}


    def expected(task, x, y):
        return x + y if task["kind"] == 0 else x * y


    def patch_source(task, action):
        expression = {1: "x + y", 2: "x * y", 3: str(expected(task, task["x"], task["y"]))}
        if action == 3:
            expression[3] += f" if x == {task['x']} else 0"
        return "def solve(x, y):\n    return " + expression[action] + "\n"


    demo_task = training[0]
    overfit = patch_source(demo_task, 3)
    public_case = ((demo_task["x"], demo_task["y"], expected(demo_task, 2, 2)),)
    private_case = ((5, 3, expected(demo_task, 5, 3)),)
    print(overfit, python_tests(overfit, public_case), python_tests(overfit, private_case))
    assert python_tests(overfit, public_case)["passed"]
    assert not python_tests(overfit, private_case)["passed"]
    return expected, patch_source, python_tests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Public and private tests

    The specification exposes whether the function should add or multiply. Public-test feedback enters the next observation; private answers remain in the terminal verifier. Legal actions provide the inspect, patch, test and submit protocol.

    Training and evaluation use different numeric inputs. Evaluate the sum and product specifications separately.
    """)
    return


@app.cell
def _(expected, heldout, patch_source, python_tests, training):
    class CodeEnvironment:
        actions = ["inspect", "patch_add", "patch_multiply", "patch_overfit", "test", "submit"]
        max_turns = 7

        def __init__(self, task):
            self.task, self.stage = task, 0
            self.source = "def solve(x, y):\n    return x - y\n"
            self.feedback, self.success = False, False

        def observe(self):
            state = [float(self.stage > 0 and self.task["kind"] == k) for k in range(2)]
            state += [float(self.stage == k) for k in range(4)] + [float(self.feedback)]
            legal = [False] * len(self.actions)
            for action in [[0], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 5]][self.stage]:
                legal[action] = True
            return state, legal

        def step(self, action):
            assert self.observe()[1][action]
            task = self.task
            if action == 0:
                self.stage = 1
                return -0.01, False, {"spec": "sum" if task["kind"] == 0 else "product",
                                      "source": self.source,
                                      "public_case": [task["x"], task["y"],
                                          expected(task, task["x"], task["y"])]}
            if action in [1, 2, 3]:
                self.source, self.stage, self.feedback = patch_source(task, action), 2, False
                return -0.01, False, {"source": self.source}
            if action == 4:
                cases = ((task["x"], task["y"], expected(task, task["x"], task["y"])),)
                result = python_tests(self.source, cases)
                self.feedback, self.stage = result["passed"], 3
                return -0.01, False, result
            inputs = [(task["x"] + 1, task["y"] + 2), (-task["x"], task["y"]), (0, 3)]
            cases = tuple((x, y, expected(task, x, y)) for x, y in inputs)
            result = python_tests(self.source, cases)
            self.success = result["passed"]
            return float(self.success), True, {"private_tests": result}


    assert {t["x"] for t in training}.isdisjoint(t["x"] for t in heldout)
    print(CodeEnvironment.actions)
    return (CodeEnvironment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Train the patch policy

    We sample four trajectories per request and use the other trajectories' mean return as a baseline. Patch and submission decisions contribute to trajectory loss. Test responses become inputs to the next decision.

    Both before and after evaluation use greedy decisions on the same held-out requests. The budget is fixed and the held-out curve is used for reporting, without checkpoint selection.
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
def _(
    CodeEnvironment,
    OUTPUT,
    heldout,
    json,
    python_tests,
    run_comparison,
    training,
):
    policies, metrics = run_comparison(CodeEnvironment, training, heldout, OUTPUT)
    metrics["python_test_cache"] = python_tests.cache_info()._asdict()
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    assert metrics["python_test_cache"]["misses"] > 0
    print("Actual Python test invocations:", metrics["python_test_cache"]["misses"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checklist

    - [ ] Locate actual subprocess test inputs, exit codes and output.
    - [ ] Explain why a patch that passes public tests can fail private tests.
    - [ ] Save training curves, patch trajectories, parameters and invocation counts.

    ## Exercise

    Fill in whether the example-memorizing patch passes the private case. Hint: it returns zero for other inputs.
    """)
    return


@app.cell
def _():
    RUN_EXERCISE = False
    answer = None
    if RUN_EXERCISE:
        assert answer is False
    else:
        print("Fill answer, then enable RUN_EXERCISE.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [rLLM DeepCoder lab](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter22_agentic/rllm-deepcoder-lab): the language-model code-agent training route.
    - [Agentic training system](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter22_agentic/build-agentic-training-system): environment, trajectory and model-update interfaces.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saved run results

    CPU · Python 3.12.13 · 10.99 s

    [Open the full result page](https://walkinglabs.github.io/hands-on-modern-rl/lab-results/agentic-code-en.html)
    """)
    return


if __name__ == "__main__":
    app.run()
