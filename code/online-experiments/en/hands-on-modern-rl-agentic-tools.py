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
    # Decide again after the tool responds

    A request asks the agent to look up two numbers, add them and double the result. The initial lookup contributes to the final answer: subsequent calculations need its returned numbers. We implement that dependency as an actual environment and update the tool-selection policy using trajectory returns.

    The default uses a CPU, two seeds and 160 policy updates, with no external dataset. A small neural network selects finite tool actions under an environment-provided protocol. This experiment tests multi-turn credit assignment and tool routing.
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
    OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "agentic-tools"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    training = [{"kind": i % 2, "x": i + 2, "y": i % 5 + 1} for i in range(24)]
    heldout = [{"kind": i % 2, "x": i + 32, "y": i % 7 + 1} for i in range(16)]
    assert {t["x"] for t in training}.isdisjoint(t["x"] for t in heldout)
    print({"training_tasks": len(training), "heldout_tasks": len(heldout), "seeds": [3, 7]})
    return OUTPUT, heldout, np, torch, training


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Observations from tools

    One request type requires addition followed by doubling; the other requires subtraction followed by squaring. The request type is public. The lookup reveals the actual numbers. Legal-action constraints provide the interface order, while arithmetic choices remain policy decisions. Gold answers never enter the observation.
    """)
    return


@app.cell
def _(training):
    class ToolEnvironment:
        actions = ["lookup", "add", "subtract", "double", "square", "finish"]
        max_turns = 5

        def __init__(self, task):
            self.task, self.stage, self.value = task, 0, None
            self.success, self.numbers = False, None

        def observe(self):
            state = [float(self.task["kind"] == k) for k in range(2)]
            state += [float(self.stage == k) for k in range(4)]
            state += [value / 100 for value in (self.numbers or [0, 0])]
            state += [0.0 if self.value is None else self.value / 10000]
            legal = [False] * len(self.actions)
            for action in [[0], [1, 2], [3, 4], [5]][self.stage]:
                legal[action] = True
            return state, legal

        def step(self, action):
            assert self.observe()[1][action]
            if action == 0:
                x, y = self.task["x"], self.task["y"]
                self.numbers, self.stage = (x, y), 1
                return -0.01, False, {"x": x, "y": y}
            x, y = self.numbers
            if action in [1, 2]:
                self.value = x + y if action == 1 else x - y
                self.stage = 2
            elif action in [3, 4]:
                self.value = 2 * self.value if action == 3 else self.value ** 2
                self.stage = 3
            else:
                expected = 2 * (x + y) if self.task["kind"] == 0 else (x - y) ** 2
                self.success = self.value == expected
                return float(self.success), True, {"answer": self.value}
            return -0.01, False, {"calculator_result": self.value}


    demo = ToolEnvironment(training[0])
    for action in [0, 1, 3, 5]:
        print(demo.actions[action], demo.step(action))
    assert demo.success
    return (ToolEnvironment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A terminal reward can update earlier actions

    With rewards `[0, 0, 0, 1]` and discount 0.9, the first action's return is 0.729. Early actions retain a learning signal; sparse rewards create exploration and variance problems. The optimizer uses REINFORCE with the mean return of the other trajectories for the same request as a baseline. Tool observations are inputs to the next decision.

    Lookup and submission are fixed by the protocol; training mainly changes the two intermediate arithmetic choices. The discount above is a numerical example; training below uses the undiscounted trajectory return.

    The gradient example applies a mask to selected action log probabilities. Observation positions have no direct policy-loss term. It illustrates loss masking, rather than a complete language-model training step.
    """)
    return


@app.cell
def _(torch):
    returns = [0.9 ** 3, 0.9 ** 2, 0.9, 1.0]
    assert abs(returns[0] - 0.729) < 1e-10
    logits = torch.zeros(8, 2, requires_grad=True)
    action_mask = torch.tensor([1, 0, 0, 1, 0, 1, 0, 1], dtype=torch.float32)
    selected_logprobs = logits.log_softmax(dim=-1)[:, 0]
    loss = -(selected_logprobs * action_mask * 0.729).sum()
    loss.backward()
    assert (logits.grad[action_mask == 0] == 0).all()
    assert logits.grad[0].abs().sum() > 0
    print("Discounted returns:", returns, "\nAction/observation gradient:", logits.grad.tolist())
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
    return rollout, run_comparison


@app.cell
def _(OUTPUT, ToolEnvironment, heldout, run_comparison, training):
    policies, metrics = run_comparison(ToolEnvironment, training, heldout, OUTPUT)
    return metrics, policies


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Inspect the trajectory

    The curves evaluate unseen number combinations. Both before and after evaluation use greedy decisions, identical seeds and a fixed budget. Saved trajectories record lookup results, arithmetic calls and final answers.

    The environment supplies legal tool order. Removing the constraints is a separate difficulty experiment. Evaluation is logged without selecting checkpoints or changing the training budget.
    """)
    return


@app.cell
def _(ToolEnvironment, heldout, metrics, policies, rollout):
    assert len(metrics["fixed_heldout_trace"]) == 4
    for task in heldout:
        episode = rollout(policies[0], task, ToolEnvironment, greedy=True)
        assert episode["trace"][0]["observation"]["x"] == task["x"]
    print("Verified that each held-out trajectory receives its actual lookup result.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checklist

    - [ ] Compute the nonzero return of an early action from terminal reward.
    - [ ] Distinguish policy choices, legal-action constraints and tool observations.
    - [ ] Save held-out curves, complete trajectories and policy parameters.

    ## Exercise

    Fill in the first return for four actions and discount 0.9. Hint: discount the final unit reward three times.
    """)
    return


@app.cell
def _():
    RUN_EXERCISE = False
    answer = None
    if RUN_EXERCISE:
        assert abs(answer - 0.729) < 1e-6
    else:
        print("Fill answer, then enable RUN_EXERCISE.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Multi-turn RL](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter22_agentic/formulation): actions, tool observations, trajectories and action masks.
    - [Credit assignment](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter22_agentic/credit-assignment): how terminal rewards affect multi-turn policies.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saved run results

    CPU · Python 3.12.13 · 6.37 s

    [Open the full result page](https://walkinglabs.github.io/hands-on-modern-rl/lab-results/agentic-tools-en.html)
    """)
    return


if __name__ == "__main__":
    app.run()
