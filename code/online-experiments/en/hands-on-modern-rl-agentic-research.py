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
    # Retrieve, choose evidence and verify citations

    The agent answers questions from the book's fixed teaching corpus. It selects a query, retrieves documents, chooses evidence and extracts an answer. Answer correctness and citation correctness are scored separately. We reuse the book's runnable benchmark and add matched before/after held-out evaluation.

    The default uses a CPU, three seeds and 220 epochs. Queries are supplied candidates, answers come from document fields, and training updates a shared-feature linear policy.
    """)
    return


@app.cell
def _():
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("matplotlib") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib>=3.8,<4"])

    import copy
    import hashlib
    import inspect
    import json
    import os
    from pathlib import Path
    import random
    import urllib.request
    import matplotlib.pyplot as plt

    OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "agentic-research"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("A fixed local corpus; no external search API or model download.")
    return (
        OUTPUT,
        copy,
        hashlib,
        importlib,
        inspect,
        json,
        plt,
        random,
        sys,
        urllib,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Pin source and data

    The benchmark download is pinned to a repository commit and verified with SHA-256. Its first run requires the network to fetch one Python file. Subsequent runs reuse the verified local cache.

    Each corpus document contains a title, text, answer and document identifier.
    """)
    return


@app.cell
def _(OUTPUT, hashlib, importlib, inspect, sys, urllib):
    COMMIT = "4bce68267c0e17107ab3b42b35efb2dffa3f777d"
    SOURCE = "docs/chapter22_agentic/code/deep_research_rl_benchmark.py"
    EXPECTED_SHA256 = "22a043be7e04532629b6f89f8ba3fecc3accfce744b1ee4fba1212b5813e2a02"
    cache = OUTPUT / "benchmark.py"
    if not cache.exists():
        url = f"https://raw.githubusercontent.com/walkinglabs/hands-on-modern-rl/{COMMIT}/{SOURCE}"
        cache.write_bytes(urllib.request.urlopen(url, timeout=30).read())
    assert hashlib.sha256(cache.read_bytes()).hexdigest() == EXPECTED_SHA256
    spec = importlib.util.spec_from_file_location("research_benchmark", cache)
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)
    print("Verified source:", EXPECTED_SHA256)
    print(inspect.getsource(benchmark.run_episode))
    return EXPECTED_SHA256, benchmark


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Hold out questions

    The first seven questions train the policy; the last three evaluate it. Their target document identifiers are disjoint. The retrieval corpus is shared because evaluation requires access to evidence for new questions.

    Held-out labels do not enter training updates. Evaluation retains matching retrieval perturbation seeds, temperatures and action-selection modes.
    """)
    return


@app.cell
def _(benchmark, copy, random):
    training, heldout = benchmark.TASKS[:7], benchmark.TASKS[7:]
    assert {t.support_doc_id for t in training}.isdisjoint(t.support_doc_id for t in heldout)
    print({"train_questions": len(training), "heldout_questions": len(heldout),
           "corpus_documents": len(benchmark.DOCS)})


    def matched_evaluation(policy, greedy):
        scores, exact, citations = [], [], []
        for repeat in range(20):
            sampled_policy = copy.deepcopy(policy)
            sampled_policy.rng = random.Random(1000 + repeat)
            for index, task in enumerate(heldout):
                score, trace = benchmark.run_episode(sampled_policy, task,
                    random.Random(2000 + 100 * repeat + index), temperature=0.7, greedy=greedy)
                scores.append(score)
                exact.append(trace["answer"].lower() == task.gold_answer.lower())
                citations.append(trace["doc"].doc_id == task.support_doc_id)
        return {"reward": sum(scores) / len(scores), "answer_exact": sum(exact) / len(exact),
                "citation_support": sum(citations) / len(citations)}


    print("Matched evaluator ready: same seeds, temperature and action mode before/after.")
    return heldout, matched_evaluation, training


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Matched evaluation

    Training changes linear weights for query and evidence selection. We report greedy evaluation and repeated sampled evaluation separately, each using identical settings before and after training.

    The curve shows training-question rewards. Held-out metrics are saved separately. The table separately reports answer accuracy, citation support and total reward.
    """)
    return


@app.cell
def _(
    EXPECTED_SHA256,
    OUTPUT,
    benchmark,
    heldout,
    json,
    matched_evaluation,
    plt,
    random,
    training,
):
    runs, histories, policies = [], [], []
    for seed in [3, 7, 11]:
        policy = benchmark.LinearSoftmaxPolicy(random.Random(seed))
        before = {mode: matched_evaluation(policy, mode == "greedy")
                  for mode in ["greedy", "sampled"]}
        history = benchmark.train(policy, training, epochs=220)
        after = {mode: matched_evaluation(policy, mode == "greedy")
                 for mode in ["greedy", "sampled"]}
        runs.append({"seed": seed, "before": before, "after": after})
        histories.append(history)
        policies.append(policy)
    print(json.dumps(runs, indent=2))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for seed, history in zip([3, 7, 11], histories):
        smoothed = [sum(history[max(0, i - 19):i + 1]) / min(i + 1, 20)
                    for i in range(len(history))]
        ax.plot(smoothed, label=f"seed {seed}")
    ax.set(xlabel="Training epoch", ylabel="Training reward (20-epoch mean)", ylim=(-0.05, 1.05))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "training.png", dpi=120)
    plt.show()

    score, trace = benchmark.run_episode(policies[0], heldout[0], random.Random(2000), greedy=True)
    retrieved = benchmark.search(trace["query"], random.Random(2000))
    trajectory = {"question": heldout[0].question, "query": trace["query"],
                  "retrieved": [doc.doc_id for doc in retrieved], "chosen_doc": trace["doc"].doc_id,
                  "answer": trace["answer"], "reward": score}
    weights = [{"query": policy.query_weights, "document": policy.doc_weights} for policy in policies]
    (OUTPUT / "policy-weights.json").write_text(json.dumps(weights, indent=2))
    metrics = {"runs": runs, "fixed_heldout_trace": trajectory, "source_sha256": EXPECTED_SHA256,
               "train_questions": len(training), "heldout_questions": len(heldout), "epochs": 220,
               "algorithm": "Feature policy with REINFORCE-style baseline updates"}
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    assert trajectory["chosen_doc"] in trajectory["retrieved"]
    print(json.dumps(trajectory, indent=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checklist

    - [ ] Verify the source hash and train/evaluation split.
    - [ ] Inspect answer accuracy, citation support and total reward separately.
    - [ ] Save queries, retrieved documents, selected evidence and policy weights.

    ## Exercise

    Fill in the reward when the answer is incorrect but the citation is correct. Hint: answers contribute 0.75 and citations contribute 0.25.
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
    ## References

    - [DeepResearch evaluation](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter24_deep_research/deep-research-eval): answers, evidence, process and cost.
    - [Browser research trajectories](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter24_deep_research/browser-rl-harness): continue from a fixed corpus to web tools.
    - [Pinned book benchmark](https://github.com/walkinglabs/hands-on-modern-rl/blob/4bce68267c0e17107ab3b42b35efb2dffa3f777d/docs/chapter22_agentic/code/deep_research_rl_benchmark.py): the exact source used here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saved run results

    CPU · Python 3.12.13 · 3.28 s

    [Open the full result page](https://walkinglabs.github.io/hands-on-modern-rl/lab-results/agentic-research-en.html)
    """)
    return


if __name__ == "__main__":
    app.run()
