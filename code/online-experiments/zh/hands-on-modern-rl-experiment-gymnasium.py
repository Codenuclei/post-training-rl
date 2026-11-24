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
    # 尝试不同的控制环境

    WalkingLab × Hands-On Modern RL 配套实验。

    选择代表环境，比较状态、动作与训练反馈。

    - 运行资源：CPU
    - 默认快速预算：2,000 个训练单位；具体单位由任务决定。
    - 完整预算建议：使用各任务在 EXPERIMENTS 中配置的完整默认预算。
    - [在线 Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment-gymnasium)
    - [实验源码](https://github.com/walkinglabs/hands-on-modern-rl/tree/main/modelscope-space/hands-on-modern-rl-experiment-gymnasium)
    - [配套章节](https://walkinglabs.github.io/hands-on-modern-rl/chapter01_cartpole/principles)

    Notebook 导入 Studio 使用的同一套训练实现。按顺序运行，比较检查点曲线和最终策略回放。
    首次安装需要下载原生环境或模拟器资源；后续运行会复用工作目录缓存。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 实验目标与训练安排

    我们观察训练预算增加后，策略在任务评估指标上的变化。先用快速预算检查环境和日志，
    确认结果单元格能产生曲线或结果文件，再延长训练。

    使用普通 CPU Notebook 实例即可。

    观察各轮检查点的评估曲线与策略回放，再增加训练步数进行对照。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 安装配套 Studio 运行环境

    安装当前实验所需的依赖，并加载与 Studio 相同的训练代码。
    """)
    return


@app.cell
def _():
    from __future__ import annotations

    import hashlib
    import importlib.util
    import os
    import subprocess
    import sys
    from pathlib import Path

    REPOSITORY_URL = "https://github.com/walkinglabs/hands-on-modern-rl.git"
    SPACE_SLUG = "hands-on-modern-rl-experiment-gymnasium"
    INSTALL_DEPENDENCIES = os.getenv("HANDS_ON_LAB_PREINSTALLED") != "1"

    def locate_or_clone_repo() -> Path:
        here = Path.cwd().resolve()
        for candidate in (here, *here.parents):
            if (candidate / "modelscope-space" / SPACE_SLUG).is_dir():
                return candidate
        workspace = Path("/mnt/workspace") if Path("/mnt/workspace").is_dir() else Path.cwd()
        target = workspace / "hands-on-modern-rl-notebooks" / "source"
        target.parent.mkdir(parents=True, exist_ok=True)
        if (target / ".git").is_dir():
            subprocess.run(["git", "-C", str(target), "pull", "--ff-only"], check=True)
        else:
            subprocess.run(["git", "clone", "--depth", "1", REPOSITORY_URL, str(target)], check=True)
        return target

    REPO_ROOT = locate_or_clone_repo()
    SPACE_DIR = REPO_ROOT / "modelscope-space" / SPACE_SLUG
    requirements = SPACE_DIR / "requirements.txt"
    packages = SPACE_DIR / "packages.txt"
    cache_root = Path("/mnt/workspace/hands-on-modern-rl-notebooks") if Path("/mnt/workspace").is_dir() else REPO_ROOT / ".cache" / "online-experiments"
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(requirements.read_bytes() + (packages.read_bytes() if packages.exists() else b"")).hexdigest()[:12]
    marker = cache_root / f"{SPACE_SLUG}-{digest}.ready"

    if INSTALL_DEPENDENCIES and not marker.exists():
        if packages.exists() and sys.platform.startswith("linux") and hasattr(os, "geteuid") and os.geteuid() == 0:
            system_packages = [line.strip() for line in packages.read_text().splitlines() if line.strip() and not line.startswith("#")]
            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(["apt-get", "install", "-y", "--no-install-recommends", *system_packages], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)], check=True)
        marker.touch()
    else:
        print(f"Dependency cache ready: {marker}")

    if importlib.util.find_spec("ipywidgets") is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "ipywidgets>=8,<9"],
            check=True,
        )

    os.chdir(SPACE_DIR)
    if str(SPACE_DIR) not in sys.path:
        sys.path.insert(0, str(SPACE_DIR))
    print(f"代码仓库： {REPO_ROOT}")
    print(f"实验运行目录： {SPACE_DIR}")
    return Path, SPACE_SLUG, cache_root, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 设置训练轮次

    选择任务、每轮步数和总轮次。每一轮都保留独立的模型与评估结果。
    """)
    return


@app.cell
def _():
    import app as playground

    EXPERIMENT = "Bandit · ε-greedy"
    STEPS_PER_EPOCH = 1000
    EPOCHS = 2
    LEARNING_RATE = 0.1
    GAMMA = 0.99
    EPSILON = 0.10
    SEED = 42

    print("可选实验：")
    for name in playground.EXPERIMENTS:
        print(" ", name)
    if EXPERIMENT not in playground.EXPERIMENTS:
        raise ValueError(f"请从以下列表选择 EXPERIMENT：{list(playground.EXPERIMENTS)}")
    print(f"轮次安排：{EPOCHS} 轮 × {STEPS_PER_EPOCH:,} steps")
    return (
        EPOCHS,
        EPSILON,
        EXPERIMENT,
        GAMMA,
        LEARNING_RATE,
        SEED,
        STEPS_PER_EPOCH,
        playground,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 训练并保存每轮模型

    观察实际训练日志，等待模型索引、评估结果和结果文件写入工作目录。
    """)
    return


@app.cell
def _(
    EPOCHS,
    EPSILON,
    EXPERIMENT,
    GAMMA,
    LEARNING_RATE,
    SEED,
    STEPS_PER_EPOCH,
    playground,
):
    import html as html_module
    import re
    from IPython.display import Image as NotebookImage, display
    results = []
    for result in playground.train(EXPERIMENT, STEPS_PER_EPOCH, EPOCHS, LEARNING_RATE, GAMMA, EPSILON, SEED, '中文'):
        results.append(result)
        console = result[-1]
        value = getattr(console, 'value', str(console))
        text = html_module.unescape(re.sub('<[^>]+>', '', str(value)))
        print(text[-1800:], flush=True)
    if not results:
        raise RuntimeError('训练没有返回结果，请检查前面的日志。')
    status, metric, curve, _preview, artifact, console = results[-1]
    display(curve)
    print('结果文件：', getattr(artifact, 'value', artifact))
    print(f'已保存 {{len(playground.load_models(EXPERIMENT))}} 个此任务可选择的轮次模型。')
    return NotebookImage, display


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 加载已保存模型并回放

    选择器重新读取磁盘中的模型索引。只要 Notebook 工作目录仍保留，内核重启后也能重新加载。
    Studio 与 Notebook 使用独立容器，因此这里列出当前 Notebook 工作目录产生的模型。
    选择一个轮次，再点击“回放所选策略”，检查该模型的实际行为。
    """)
    return


@app.cell
def _(EXPERIMENT, NotebookImage, Path, display, playground):
    from IPython.display import clear_output
    import ipywidgets as widgets
    trained_models = playground.load_models(EXPERIMENT)
    if not trained_models:
        raise RuntimeError('未找到已保存的轮次模型，请先运行上面的训练单元格。')

    def model_label(record):
        return f"轮次 {record['epoch']}/{record['epochs']} · {record['step']:,}/{record['total']:,} 步 · score {record['score']:.2f} · {record['model_id']}"
    model_by_id = {record['model_id']: record for record in trained_models}
    model_picker = widgets.Dropdown(options=[(model_label(record), record['model_id']) for record in trained_models], value=trained_models[0]['model_id'], description='训练轮次模型：', layout=widgets.Layout(width='95%'), style={'description_width': '110px'})
    visualize_button = widgets.Button(description='回放所选策略', button_style='primary', icon='play', layout=widgets.Layout(width='240px'))
    inference_output = widgets.Output()

    def visualize_selected(_=None):
        record = model_by_id[model_picker.value]
        with inference_output:
            clear_output(wait=True)
            print('加载已保存模型：', record['model'])
            preview = Path(str(record['preview']))
            if preview.is_file():
                display(NotebookImage(filename=str(preview)))
            else:
                print('所选策略没有已保存的回放文件。')
            print('轮次：', f"{record['epoch']}/{record['epochs']}")
            print('训练步数：', f"{record['step']:,}")
            print('评估得分：', record['score'])
    visualize_button.on_click(visualize_selected)
    display(widgets.VBox([model_picker, visualize_button, inference_output]))
    print('默认选择最新轮次；也可以选择其他已保存模型，然后点击 回放所选策略.')
    return (trained_models,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 先检查结果，再延长预算

    比较第一个和最后一个检查点，再观察策略回放。曲线上升而回放行为异常时，
    需要检查奖励、评估方式和渲染是否对应同一策略。快速预算下的平坦曲线也不足以下结论。

    延长训练可参考：使用各任务在 EXPERIMENTS 中配置的完整默认预算。比较方法时固定预算，并至少检查三个随机种子的结果。
    """)
    return


@app.cell
def _(
    EPOCHS,
    NotebookImage,
    Path,
    SPACE_SLUG,
    cache_root,
    display,
    epoch_models,
    os,
    trained_models,
):
    import json
    records = epoch_models if 'epoch_models' in globals() else trained_models
    assert records, 'Training must produce a saved checkpoint'
    latest = max(records, key=lambda record: (record.get('created_ns', 0), record.get('step', 0)))
    _preview = Path(str(latest.get('preview', '')))
    assert _preview.is_file(), 'Training must produce a replay or result image'
    display(NotebookImage(filename=str(_preview)))
    output_id = os.getenv('HANDS_ON_LAB_ID', SPACE_SLUG)
    output = Path(os.getenv('HANDS_ON_LAB_OUTPUT', str(cache_root / 'results'))) / output_id
    output.mkdir(parents=True, exist_ok=True)
    metrics = {'score': latest.get('score'), 'steps': latest.get('step'), 'epochs': EPOCHS, 'checkpoints': len(records), 'preview': _preview.name}
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 本次运行结果

    CPU · Python 3.12.13 · 3.08 s

    [查看完整结果页](https://walkinglabs.github.io/hands-on-modern-rl/lab-results/gymnasium-zh.html)
    """)
    return


if __name__ == "__main__":
    app.run()
