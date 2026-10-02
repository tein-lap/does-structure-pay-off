"""End-to-end check of score.py and loro.py on fake runs."""

import json
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PATCH = "--- a/pkg/client.py\n+++ b/pkg/client.py\n@@ -12,1 +12,1 @@\n-x\n+y\n"


def write_run(root, config, budget, seed, task, resolved, events):
    d = root / config / f"b{budget}" / f"s{seed}" / task
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"resolved": resolved}))
    (d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def test_score_and_loro(tmp_path, graph):
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    for name in ("pkg", "other"):
        with open(graphs / f"{name}.pkl", "wb") as fh:
            pickle.dump(graph, fh)
    tasks = [{"instance_id": f"{repo}-{i}", "repo": f"org/{repo}", "patch": PATCH}
             for repo in ("pkg", "other") for i in range(2)]
    (tmp_path / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))

    runs = tmp_path / "runs"
    read = {"tool": "read_file", "args": {"path": "pkg/client.py"}}
    grep = {"tool": "grep", "args": {"pattern": "x"}}
    for t in tasks:
        write_run(runs, "A", 100, 0, t["instance_id"], False, [grep] * 60 + [read])
        write_run(runs, "D", 100, 0, t["instance_id"], True, [read, {"tool": "get_code_neighbors", "args": {}}])

    out = tmp_path / "results"
    subprocess.run([sys.executable, str(ROOT / "experiments/score.py"), "--runs", str(runs),
                    "--tasks", str(tmp_path / "tasks.jsonl"), "--graphs", str(graphs), "--out", str(out)],
                   check=True, capture_output=True, text=True)
    resolution = (out / "resolution.md").read_text()
    assert "| A | 0.0 | 4 |" in resolution and "| D | 100.0 | 4 |" in resolution
    per_task = (out / "per_task.csv").read_text()
    assert "file_recall@25" in per_task

    loro = subprocess.run([sys.executable, str(ROOT / "experiments/loro.py"), "--per-task", str(out / "per_task.csv")],
                          check=True, capture_output=True, text=True).stdout
    assert "| org/pkg | D | 100.0 | 100.0 | +0.0 |" in loro


def test_bridge_eval_random_baseline_is_exact():
    sys.path.insert(0, str(ROOT / "experiments"))
    from bridge_eval import random_hit, tokens
    assert abs(random_hit(10, 1, 1) - 0.1) < 1e-12
    assert abs(random_hit(10, 2, 5) - (1 - 56 / 252)) < 1e-12   # 1 - C(8,5)/C(10,5)
    assert random_hit(5, 0, 3) == 0.0
    assert "content" in tokens("prepare_content_length") and "length" in tokens("ContentLength")
