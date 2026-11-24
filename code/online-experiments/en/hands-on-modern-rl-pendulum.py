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
    # Continuous torque and value estimates

    When a pendulum leans to the right, the action includes the magnitude of the applied torque. Pendulum accepts a single number between −2 and 2. We first measure a zero-torque policy, then train a Gaussian policy and a value network.

    This experiment uses the book's A2C and VecNormalize recipe: CPU, eight environments and 100,000 steps by default. Set `TOTAL_STEPS=300000` for a longer budget. Companion source: `code/chapter06_actor_critic/actor_critic_pendulum.py`.
    """)
    return


@app.cell
def _():
    import importlib.util
    import subprocess
    import sys
    packages = {'torch': 'torch>=2.5,<3', 'numpy': 'numpy>=1.26,<3', 'matplotlib': 'matplotlib>=3.8,<4', 'PIL': 'pillow>=10,<13', 'gymnasium': 'gymnasium[classic-control]>=1.0,<2', 'pygame': 'pygame>=2.5,<3', 'stable_baselines3': 'stable-baselines3>=2.4,<3'}
    missing = [spec for name, spec in packages.items() if importlib.util.find_spec(name) is None]
    if missing:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
    import json
    import os
    from pathlib import Path
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    import gymnasium as gym
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from stable_baselines3 import A2C
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    torch.set_num_threads(1)
    OUTPUT = Path(os.getenv('HANDS_ON_LAB_OUTPUT', 'lab-output')) / 'pendulum'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TOTAL_STEPS, SEED = (100000, 42)
    probe = gym.make('Pendulum-v1')
    _obs, _ = probe.reset(seed=SEED)
    print({'observation': _obs.tolist(), 'torque_range': [-2, 2], 'steps': TOTAL_STEPS})
    assert _obs.shape == (3,) and probe.action_space.shape == (1,)
    probe.close()
    return (
        A2C,
        BaseCallback,
        DummyVecEnv,
        Monitor,
        OUTPUT,
        SEED,
        TOTAL_STEPS,
        VecNormalize,
        gym,
        json,
        np,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Preserve observation normalization during evaluation

    Observations contain the cosine and sine of the angle and the angular velocity. Training normalizes observations and rewards, so the critic predicts normalized returns.

    Evaluation uses the training observation statistics without updating them and accumulates raw environment rewards. Evaluation seeds 1,000–1,009 provide initial states that are excluded from training.
    """)
    return


@app.cell
def _(
    A2C,
    BaseCallback,
    DummyVecEnv,
    Monitor,
    SEED,
    TOTAL_STEPS,
    VecNormalize,
    gym,
    np,
    torch,
):
    def make_env(rank):
        def create():
            env = Monitor(gym.make("Pendulum-v1"))
            env.reset(seed=SEED + rank)
            return env
        return create


    train_env = VecNormalize(DummyVecEnv([make_env(i) for i in range(8)]),
                             norm_obs=True, norm_reward=True)
    model = A2C("MlpPolicy", train_env, n_steps=32, learning_rate=7e-4,
                gae_lambda=0.95, gamma=0.99, seed=SEED, device="cpu", verbose=0)


    @torch.no_grad()
    def evaluate(policy=None):
        env = gym.make("Pendulum-v1")
        totals = []
        for seed in range(1000, 1010):
            obs, _ = env.reset(seed=seed)
            total = 0.0
            for _ in range(200):
                action = np.zeros(1) if policy is None else policy.predict(
                    train_env.normalize_obs(obs[None]), deterministic=True)[0][0]
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            totals.append(total)
        env.close()
        return totals


    zero_return, before = evaluate(), evaluate(model)
    print("Zero torque / untrained mean:", np.mean(zero_return), np.mean(before))


    class Record(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rewards, self.curve = [], [(0, float(np.mean(before)))]
            self.losses = []

        def _on_step(self):
            self.rewards.extend(info["episode"]["r"] for info in self.locals["infos"]
                                if "episode" in info)
            if self.num_timesteps % 20000 == 0:
                self.curve.append((self.num_timesteps, float(np.mean(evaluate(self.model)))))
                self.losses.append(float(self.model.logger.name_to_value.get("train/value_loss", 0)))
            return True


    record = Record()
    model.learn(total_timesteps=TOTAL_STEPS, callback=record)
    after = evaluate(model)
    print("Held-out return before / after:", np.mean(before), np.mean(after))
    assert np.isfinite(after).all() and len(record.rewards) > 0
    return after, before, make_env, model, record, train_env, zero_return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Save and reload the policy and statistics

    Saving only the network loses the way its inputs were normalized. We save both the model and the statistics, reload them and check that the same observation produces the same action.

    The plots retain the actual short-run performance. If returns remain low, extend the training budget and compare against the same evaluation seeds.
    """)
    return


@app.cell
def _(
    A2C,
    DummyVecEnv,
    OUTPUT,
    VecNormalize,
    make_env,
    model,
    np,
    plt,
    record,
    train_env,
    zero_return,
):
    model.save(OUTPUT / 'a2c')
    train_env.save(OUTPUT / 'normalization.pkl')
    reloaded_env = VecNormalize.load(OUTPUT / 'normalization.pkl', DummyVecEnv([make_env(100)]))
    reloaded_env.training, reloaded_env.norm_reward = (False, False)
    reloaded = A2C.load(OUTPUT / 'a2c', env=reloaded_env, device='cpu')
    _obs = np.array([[1.0, 0.0, 0.3]], dtype=np.float32)
    assert np.allclose(model.predict(train_env.normalize_obs(_obs), deterministic=True)[0], reloaded.predict(reloaded_env.normalize_obs(_obs), deterministic=True)[0])
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.5))
    curve = np.array(record.curve)
    _axes[0].plot(curve[:, 0], curve[:, 1], marker='o')
    _axes[0].axhline(np.mean(zero_return), ls='--', color='grey', label='Zero torque')
    _axes[0].set(xlabel='Environment steps', ylabel='Held-out raw return')
    _axes[0].legend()
    _axes[1].plot(curve[1:, 0], record.losses)
    _axes[1].set(xlabel='Environment steps', ylabel='Value loss (normalized rewards)')
    _fig.tight_layout()
    _fig.savefig(OUTPUT / 'training.png', dpi=120)
    plt.show()
    return reloaded, reloaded_env


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Inspect a fixed-seed replay

    The replay uses a new seed, 2,000. Observations pass through frozen normalization statistics before reaching the policy. Torque, angular velocity and value predictions come from this saved policy. The critic values and raw environment rewards use different units.
    """)
    return


@app.cell
def _(
    OUTPUT,
    after,
    before,
    gym,
    json,
    np,
    plt,
    reloaded,
    reloaded_env,
    torch,
    train_env,
    zero_return,
):
    from IPython.display import Image, display
    from PIL import Image as PILImage
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    _obs, _ = env.reset(seed=2000)
    frames, actions, speeds, values = ([], [], [], [])
    for step in range(200):
        normalized = reloaded_env.normalize_obs(_obs[None])
        action, _ = reloaded.predict(normalized, deterministic=True)
        with torch.no_grad():
            value = reloaded.policy.predict_values(torch.tensor(normalized)).item()
        actions.append(float(action[0, 0]))
        speeds.append(float(_obs[2]))
        values.append(value)
        if step % 2 == 0:
            frames.append(PILImage.fromarray(env.render()).resize((240, 240)))
        _obs, _, terminated, truncated, _ = env.step(action[0])
        if terminated or truncated:
            break
    env.close()
    assert np.max(np.abs(actions)) <= 2
    frames[0].save(OUTPUT / 'policy.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)
    display(Image(filename=str(OUTPUT / 'policy.gif')))
    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, data, label in zip(_axes, [actions, speeds, values], ['Torque', 'Angular velocity', 'Normalized value estimate']):
        ax.plot(data)
        ax.set(xlabel='Step', ylabel=label)
    _fig.tight_layout()
    _fig.savefig(OUTPUT / 'replay-signals.png', dpi=120)
    plt.show()
    metrics = {'zero_torque_returns': zero_return, 'before_returns': before, 'after_returns': after}
    (OUTPUT / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    train_env.close()
    reloaded_env.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checklist

    - [ ] Explain the output difference between Gaussian and discrete-action policies.
    - [ ] Reload the model and observation statistics and pass the action-consistency check.
    - [ ] Evaluate using raw environment rewards and retain unsuccessful control behavior.

    ## Exercise

    Fill in whether observation statistics should be updated during evaluation. Hint: evaluation uses fixed training statistics. Enable `RUN_EXERCISE` after filling `answer`.
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

    - [SB3 A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html): the implementation interface and continuous-policy configuration.
    - [SB3 VecNormalize](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize): save, load and freeze normalization statistics.
    - [Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/): observations, torque bounds and rewards.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saved run results

    CPU · Python 3.12.13 · 10.16 s

    [Open the full result page](https://walkinglabs.github.io/hands-on-modern-rl/lab-results/pendulum-en.html)
    """)
    return


if __name__ == "__main__":
    app.run()
