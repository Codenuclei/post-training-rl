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
    # 从 CartPole 开始：30 秒训练一个 PPO 智能体

    CartPole 的动作只有两个：向左或向右推动小车。下面的实验使用 CPU 训练 PPO，并依次展示奖励曲线和训练后的策略动画。无需 GPU，也无需提前掌握 PPO 的数学推导。

    运行方式：依次执行下面三个代码单元。训练步数默认为 30,000；如果在线环境较慢，可先改成 10,000 验证流程。
    """)
    return


@app.cell
def _():
    # '%pip install -q "gymnasium[classic-control]>=1.0,<2" "stable-baselines3>=2.4,<3" "imageio>=2.34,<3"' command supported automatically in marimo
    return


@app.cell
def _():
    from pathlib import Path
    import os
    import time

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import gymnasium as gym
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",  # macOS
        "Microsoft YaHei",   # Windows
        "SimHei",            # Windows fallback
        "Noto Sans CJK SC",  # Linux
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    import numpy as np
    from IPython.display import Image, display
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy

    SEED = 42
    TOTAL_TIMESTEPS = 30_000
    OUTPUT_DIR = Path("cartpole-artifacts")
    OUTPUT_DIR.mkdir(exist_ok=True)

    def evaluate(model, episodes=5):
        env = gym.make("CartPole-v1")
        episode_rewards, _ = evaluate_policy(
            model, env, n_eval_episodes=episodes, deterministic=True,
            return_episode_rewards=True, warn=False
        )
        env.close()
        return float(np.mean(episode_rewards)), float(np.std(episode_rewards))

    return (
        Image,
        OUTPUT_DIR,
        PPO,
        SEED,
        TOTAL_TIMESTEPS,
        display,
        evaluate,
        gym,
        imageio,
        plt,
        time,
    )


@app.cell
def _(OUTPUT_DIR, PPO, SEED, TOTAL_TIMESTEPS, evaluate, gym, time):
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)
    model = PPO(
        "MlpPolicy", env, seed=SEED, verbose=0, device="cpu",
        n_steps=1024, batch_size=64, learning_rate=3e-4
    )

    steps = [0]
    initial_mean, initial_std = evaluate(model)
    rewards = [initial_mean]
    print(f"初始策略：{initial_mean:.1f} ± {initial_std:.1f}")

    started_at = time.perf_counter()
    trained = 0
    while trained < TOTAL_TIMESTEPS:
        chunk = min(2_000, TOTAL_TIMESTEPS - trained)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        trained += chunk
        mean_reward, std_reward = evaluate(model)
        steps.append(trained)
        rewards.append(mean_reward)
        print(f"{trained:>6,}/{TOTAL_TIMESTEPS:,} 步 | 奖励 {mean_reward:>6.1f} ± {std_reward:>5.1f}")

    model.save(OUTPUT_DIR / "ppo-cartpole")
    elapsed = time.perf_counter() - started_at
    final_mean, final_std = evaluate(model, episodes=10)
    print(f"\n训练耗时：{elapsed:.1f} 秒")
    print(f"10 回合平均奖励：{final_mean:.1f} ± {final_std:.1f}")
    return env, model, rewards, steps


@app.cell
def _(
    Image,
    OUTPUT_DIR,
    SEED,
    display,
    env,
    gym,
    imageio,
    model,
    plt,
    rewards,
    steps,
):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, rewards, marker="o", linewidth=2, color="#4f46e5")
    ax.axhline(475, linestyle="--", color="#16a34a", label="解决阈值 475")
    ax.set(xlabel="训练步数", ylabel="平均奖励", ylim=(0, 510), title="PPO 在 CartPole-v1 上的训练结果")
    ax.grid(alpha=0.22)
    ax.legend()
    plt.show()

    render_env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = render_env.reset(seed=SEED + 1)
    frames, score = [], 0.0
    for _ in range(500):
        frames.append(render_env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = render_env.step(action)
        score += float(reward)
        if terminated or truncated:
            break
    render_env.close()
    env.close()

    gif_path = OUTPUT_DIR / "cartpole-trained-policy.gif"
    imageio.mimsave(gif_path, frames, duration=1 / 30, loop=0)
    print(f"动画回合得分：{score:.0f}/500")
    display(Image(filename=str(gif_path)))
    return


if __name__ == "__main__":
    app.run()
