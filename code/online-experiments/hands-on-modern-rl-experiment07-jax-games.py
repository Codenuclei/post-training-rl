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
    # Experiment 07 · JAX MinAtar Games

    **WalkingLab × Hands-On Modern RL companion experiment notebook**

    JIT-compile a policy-gradient update on CPU and train inside a compact MinAtar game.

    - Resource profile: **CPU**
    - Quick run in this notebook: **100** training units
    - Full experiment: 1,000+ episodes after the first JAX compilation
    - [Live ModelScope Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment07-jax-games)
    - [Experiment source](https://github.com/walkinglabs/hands-on-modern-rl/tree/main/modelscope-space/hands-on-modern-rl-experiment07-jax-games)
    - [Hands-On Modern RL](https://github.com/walkinglabs/hands-on-modern-rl) · [WalkingLab](https://modelscope.cn/organization/walkinglab)

    The notebook imports the exact runtime used by the Studio. Change the parameters below, run the cells in order,
    and compare the checkpoint curve with the final policy GIF or result image. The first setup can take longer because
    native environments and simulator assets are cached; later runs reuse `/mnt/workspace/hands-on-modern-rl-notebooks`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Experiment and training schedule

    This experiment asks whether the selected policy improves on the task's evaluation metric as its training budget
    increases. Start with the quick budget to verify the environment and logs. Then increase the budget only after the
    complete result cell produces a curve and an artifact.

    A normal ModelScope **CPU Notebook** is sufficient; no GPU is required.

    Compare the evaluation curve and replay at each checkpoint. Increase the budget above for a longer training run.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Prepare the matching Studio runtime
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
    SPACE_SLUG = "hands-on-modern-rl-experiment07-jax-games"
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
    print(f"Repository: {REPO_ROOT}")
    print(f"Experiment runtime: {SPACE_DIR}")
    return (
        Path,
        SPACE_DIR,
        SPACE_SLUG,
        cache_root,
        importlib,
        os,
        subprocess,
        sys,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Configure the epoch schedule
    """)
    return


@app.cell
def _(SPACE_SLUG, importlib, subprocess, sys):
    if SPACE_SLUG.endswith('experiment10-minestudio'):
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-deps', 'minestudio==1.1.6'], check=True)
    if SPACE_SLUG.endswith('experiment11-unity-mlagents'):
        from bootstrap_mlagents import ensure_mlagents
        ensure_mlagents()
    runtime = importlib.import_module('space_runtime')
    tasks = {item['key']: item for item in runtime.TASKS}
    print('Available tasks:')
    for key, item in tasks.items():
        title = item.get('title', {})
        print(f"  {key:20s} {title.get('en', title)} · {item.get('environment', 'environment provided by runtime')}")
    TASK_KEY = 'breakout'
    STEPS_PER_EPOCH = 50
    EPOCHS = 2
    TRAINING_BUDGET = STEPS_PER_EPOCH * EPOCHS
    LEARNING_RATE = 0.0003
    GAMMA = 0.99
    EPSILON = 0.1
    SEED = 42
    if TASK_KEY not in tasks:
        raise ValueError(f'Unknown TASK_KEY={TASK_KEY!r}. Choose one of {list(tasks)}')
    selected_task = tasks[TASK_KEY]
    print('\nSelected:', selected_task.get('title', {}).get('en', TASK_KEY))
    print('Algorithm:', selected_task.get('algorithm'))
    print(f'Epoch schedule: {EPOCHS} epochs × {STEPS_PER_EPOCH:,} steps = {TRAINING_BUDGET:,} total steps')
    print('Every epoch is evaluated and saved as a separately selectable model.')
    return (
        EPOCHS,
        EPSILON,
        GAMMA,
        LEARNING_RATE,
        SEED,
        TASK_KEY,
        TRAINING_BUDGET,
        runtime,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Train and save one model per epoch
    """)
    return


@app.cell
def _(
    EPOCHS,
    EPSILON,
    GAMMA,
    LEARNING_RATE,
    Path,
    SEED,
    SPACE_DIR,
    TASK_KEY,
    TRAINING_BUDGET,
    runtime,
):
    from IPython.display import display
    import json
    import matplotlib.pyplot as plt
    import time
    print('Device: CPU')
    print('Starting the same training generator used by the live Studio...\n')
    events = []
    epoch_models = []
    _MODEL_INDEX = SPACE_DIR / 'artifacts' / 'notebook-models.json'
    _MODEL_INDEX.parent.mkdir(parents=True, exist_ok=True)

    def persist_epoch_models():
        try:
            saved_index = json.loads(_MODEL_INDEX.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            saved_index = {'models': []}
        previous = [item for item in saved_index.get('models', []) if isinstance(item, dict)]
        new_ids = {item['model_id'] for item in epoch_models}
        merged = epoch_models + [item for item in previous if item.get('model_id') not in new_ids]
        _MODEL_INDEX.write_text(json.dumps({'models': merged}, indent=2, ensure_ascii=False), encoding='utf-8')
    for event in runtime.run(TASK_KEY, TRAINING_BUDGET, LEARNING_RATE, GAMMA, EPSILON, SEED, checkpoints=EPOCHS):
        events.append(dict(event))
        if event.get('model') and event.get('checkpoint_index'):
            model_path = str(event['model'])
            model_file = Path(model_path)
            try:
                relative_model_id = model_file.relative_to(SPACE_DIR / 'artifacts').as_posix()
            except ValueError:
                relative_model_id = model_file.name
            epoch_models.append({'model_id': str(event.get('model_id') or relative_model_id), 'task_key': TASK_KEY, 'epoch': int(event['checkpoint_index']), 'epochs': int(event.get('checkpoint_count') or EPOCHS), 'step': int(event.get('step') or 0), 'score': event.get('score'), 'model': model_path, 'preview': str(event.get('preview') or ''), 'created_ns': time.time_ns()})
            persist_epoch_models()
        message = event.get('log') or event.get('detail')
        if message:
            print(message, flush=True)
    if not events:
        raise RuntimeError('The runtime returned no training events')
    final_event = events[-1]
    print('\nFinal phase:', final_event.get('phase', 'complete'))
    print('Final score:', final_event.get('score', 'reported in the log'))
    print(f'Saved {len(epoch_models)} selectable epoch models in this run.')
    print('Model index:', _MODEL_INDEX)
    return display, epoch_models, final_event, json, plt


@app.cell
def _(TASK_KEY, final_event, plt):
    x = final_event.get("x", [])
    y = final_event.get("y", [])
    if x and y:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(x, y, marker="o", color="#5b5ce2", linewidth=2)
        ax.set_title(f"{TASK_KEY} · checkpoint evaluation")
        ax.set_xlabel("Training progress")
        ax.set_ylabel("Evaluation score")
        ax.grid(alpha=0.25)
        plt.show()
    else:
        print("This task reports its result through the artifact rather than a scalar learning curve.")

    print("Run the inference cell below to choose an epoch model and visualize its exact learned policy.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Load a saved epoch model and run inference

    This cell reloads the on-disk model index. It therefore works after training and after a kernel restart, as long as the ModelScope Notebook workspace is still available. Studio and Notebook run in separate containers, so this selector lists models produced in the current Notebook workspace.
    """)
    return


@app.cell
def _(Path, SPACE_DIR, TASK_KEY, display, json, runtime):
    from IPython.display import Image as NotebookImage, clear_output
    import ipywidgets as widgets
    _MODEL_INDEX = SPACE_DIR / 'artifacts' / 'notebook-models.json'

    def load_notebook_models():
        try:
            payload = json.loads(_MODEL_INDEX.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []
        records = [item for item in payload.get('models', []) if isinstance(item, dict) and item.get('model_id') and (item.get('task_key') == TASK_KEY)]
        return sorted(records, key=lambda item: (int(item.get('created_ns', 0)), int(item.get('step', 0)), int(item.get('epoch', 0))), reverse=True)
    trained_models = load_notebook_models()
    if not trained_models:
        raise RuntimeError('No saved epoch model was found. Run the training cell above first.')

    def model_label(record):
        score = record.get('score')
        score_text = '—' if score is None else f'{float(score):.2f}'
        return f"Epoch {record.get('epoch')}/{record.get('epochs')} · {int(record.get('step', 0)):,} steps · score {score_text} · {record['model_id']}"
    model_by_id = {record['model_id']: record for record in trained_models}
    model_picker = widgets.Dropdown(options=[(model_label(record), record['model_id']) for record in trained_models], value=trained_models[0]['model_id'], description='Epoch model:', layout=widgets.Layout(width='95%'), style={'description_width': '110px'})
    visualize_button = widgets.Button(description='Visualize selected policy', button_style='primary', icon='play', layout=widgets.Layout(width='240px'))
    inference_output = widgets.Output()

    def visualize_selected(_=None):
        record = model_by_id[model_picker.value]
        with inference_output:
            clear_output(wait=True)
            print('Loading saved model:', record['model'])
            preview = record.get('preview')
            if hasattr(runtime, 'render_preview'):
                try:
                    rendered = runtime.render_preview(record['model_id'])
                    preview = rendered.get('preview', preview)
                    print('Generated a fresh deterministic rollout.')
                except Exception as exc:
                    print(f'Fresh rollout unavailable ({type(exc).__name__}); using the saved epoch rollout.')
            elif hasattr(runtime, 'render_saved_model'):
                try:
                    rendered = runtime.render_saved_model(TASK_KEY, record['model'], record['model_id'])
                    preview = rendered.get('preview', preview)
                    print('Generated a fresh rollout from the selected saved policy.')
                except Exception as exc:
                    print(f'Fresh rollout unavailable ({type(exc).__name__}); using the saved epoch rollout.')
            preview_path = Path(str(preview)) if preview else None
            if preview_path and preview_path.is_file():
                display(NotebookImage(filename=str(preview_path)))
            else:
                print('No replay file is available for this checkpoint.')
            print('Model:', record['model'])
            print('Epoch:', f"{record.get('epoch')}/{record.get('epochs')}")
            print('Training step:', f"{int(record.get('step', 0)):,}")
            print('Evaluation score:', record.get('score', '—'))
    visualize_button.on_click(visualize_selected)
    display(widgets.VBox([model_picker, visualize_button, inference_output]))
    print('The newest epoch is selected by default. Choose another saved model, then click Visualize selected policy.')
    return NotebookImage, trained_models


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Read the result before increasing the budget

    Compare the first and last checkpoint values, then inspect the replay. A rising curve with an implausible replay can
    help locate differences in reward, evaluation and rendering. Continue with **1,000+ episodes after the first JAX compilation**, keep the seed fixed,
    and compare at least three seeds before drawing a conclusion.
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
    json,
    os,
    trained_models,
):
    records = epoch_models if 'epoch_models' in globals() else trained_models
    assert records, 'Training must produce a saved checkpoint'
    latest = max(records, key=lambda record: (record.get('created_ns', 0), record.get('step', 0)))
    preview = Path(str(latest.get('preview', '')))
    assert preview.is_file(), 'Training must produce a replay or result image'
    display(NotebookImage(filename=str(preview)))
    output_id = os.getenv('HANDS_ON_LAB_ID', SPACE_SLUG)
    output = Path(os.getenv('HANDS_ON_LAB_OUTPUT', str(cache_root / 'results'))) / output_id
    output.mkdir(parents=True, exist_ok=True)
    metrics = {'score': latest.get('score'), 'steps': latest.get('step'), 'epochs': EPOCHS, 'checkpoints': len(records), 'preview': preview.name}
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saved run results

    CPU · Python 3.12.13 · 16.38 s

    [Open the full result page](https://walkinglabs.github.io/hands-on-modern-rl/lab-results/jax-en.html)
    """)
    return


if __name__ == "__main__":
    app.run()
