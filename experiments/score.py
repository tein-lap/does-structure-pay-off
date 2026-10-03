"""Score agent runs: resolution, localization recall and budget use.

Expected layout (one folder per run):

    runs/<config>/b<budget>/s<seed>/<task_id>/events.jsonl   one tool call per line:
                                                             {"tool": ..., "args": {...}}
    runs/<config>/b<budget>/s<seed>/<task_id>/result.json    {"resolved": true/false}

<config> is an arm (A, B, C, D) or a bridge configuration (e.g. D+bridge).

Outputs (to --out):
    per_task.csv        one row per run (input for loro.py)
    resolution.md       resolution rate per config and budget (mean ± sd over seeds)
    localization.md     file/function recall and calls by category

Localization at 25/50 calls is computed two ways when possible:
    measured    from runs that really had that budget (b25/, b50/)
    truncated   from the largest-budget run, cut at the first 25/50 calls
Resolution is never estimated by truncation (see dspo/localization.py).

Usage:
    python experiments/score.py --runs runs --tasks tasks.jsonl --graphs graphs/ --out results/
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dspo.localization import (GraphFiles, call_category, changed_locations, is_budgeted,  # noqa: E402
                               load_events, patch_alignment, recall, truncate, visited_from_events)
from dspo.probe import load_graph  # noqa: E402

BUDGETS = (25, 50, 100)


def find_runs(root: Path):
    for result in sorted(root.glob("*/b*/s*/*/result.json")):
        task_dir = result.parent
        seed_dir, budget_dir, config_dir = task_dir.parent, task_dir.parent.parent, task_dir.parent.parent.parent
        yield {
            "config": config_dir.name,
            "budget": int(budget_dir.name[1:]),
            "seed": int(seed_dir.name[1:]),
            "task_id": task_dir.name,
            "dir": task_dir,
        }


def graph_for(repo: str, graphs: dict[str, GraphFiles]) -> GraphFiles | None:
    repo = repo.lower()
    matches = [name for name in graphs if name.lower() in repo]
    return graphs[max(matches, key=len)] if matches else None


def fmt(values: list[float]) -> str:
    if not values:
        return "–"
    mean = 100 * statistics.mean(values)
    if len(values) > 1:
        return f"{mean:.1f} ± {100 * statistics.stdev(values):.1f}"
    return f"{mean:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True, help="folder of <repo>.pkl graphs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--id-key", default="instance_id")
    parser.add_argument("--repo-key", default="repo")
    parser.add_argument("--patch-key", default="patch")
    args = parser.parse_args()

    tasks = {}
    with open(args.tasks, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                t = json.loads(line)
                tasks[str(t[args.id_key])] = t
    graphs = {p.stem: GraphFiles(load_graph(p)) for p in sorted(args.graphs.glob("*.pkl"))}

    rows = []
    for run in find_runs(args.runs):
        task = tasks.get(run["task_id"])
        if task is None:
            print(f"warning: unknown task {run['task_id']}", file=sys.stderr)
            continue
        result = json.loads((run["dir"] / "result.json").read_text())
        events_path = run["dir"] / "events.jsonl"
        events = load_events(events_path) if events_path.exists() else []
        row = {k: run[k] for k in ("config", "budget", "seed", "task_id")}
        row["repo"] = task[args.repo_key]
        row["resolved"] = int(bool(result.get("resolved")))
        row["budgeted_calls"] = sum(map(is_budgeted, events))
        cats = Counter(call_category(e.get("tool", ""), e.get("args")) for e in events if is_budgeted(e))
        for cat in ("explore", "graph", "edit", "test", "other"):
            row[f"calls_{cat}"] = cats.get(cat, 0)
        gf = graph_for(row["repo"], graphs)
        if gf is not None:
            gold = changed_locations(task[args.patch_key], gf)
            row["gold_functions"] = len(gold.functions)
            matched, checked = patch_alignment(task[args.patch_key], gf)
            row["alignment"] = round(matched / checked, 3) if checked else ""  # < 1: graph not at base commit
            for k in BUDGETS:
                if k > run["budget"]:
                    continue
                seen = visited_from_events(truncate(events, k), gf)
                row[f"file_recall@{k}"] = recall(gold.files, seen.files)
                row[f"function_recall@{k}"] = recall(gold.functions, seen.functions)
        rows.append(row)

    if not rows:
        sys.exit("no runs found")
    args.out.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    with open(args.out / "per_task.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    configs = sorted({r["config"] for r in rows})
    budgets = sorted({r["budget"] for r in rows})

    # Resolution: mean over tasks per seed, then mean ± sd over seeds.
    def per_seed(config, budget, key):
        by_seed = defaultdict(list)
        for r in rows:
            if r["config"] == config and r["budget"] == budget and r.get(key) is not None:
                by_seed[r["seed"]].append(r[key])
        return [statistics.mean(v) for v in by_seed.values()]

    lines = ["| Config | " + " | ".join(f"{b} calls" for b in budgets) + " | Runs |",
             "|---" * (len(budgets) + 2) + "|"]
    for c in configs:
        n = sum(1 for r in rows if r["config"] == c)
        lines.append(f"| {c} | " + " | ".join(fmt(per_seed(c, b, "resolved")) for b in budgets) + f" | {n} |")
    (args.out / "resolution.md").write_text(
        "Resolution rate (%), measured at each real budget; mean ± sd over seeds.\n\n" + "\n".join(lines) + "\n")

    top = max(budgets)
    lines = ["| Config | Source | " + " | ".join(f"File@{k} | Func@{k}" for k in BUDGETS) +
             " | Explore | Graph | Edit | Test |", "|---" * (2 + 2 * len(BUDGETS) + 4) + "|"]
    for c in configs:
        for b in budgets:
            src = "measured" if b != top else f"b{top} (truncated below {top})"
            cells = []
            for k in BUDGETS:
                ok = k <= b
                cells.append(fmt(per_seed(c, b, f"file_recall@{k}")) if ok else "–")
                cells.append(fmt(per_seed(c, b, f"function_recall@{k}")) if ok else "–")
            calls = []
            for cat in ("explore", "graph", "edit", "test"):
                vals = [r[f"calls_{cat}"] for r in rows if r["config"] == c and r["budget"] == b]
                calls.append(f"{statistics.mean(vals):.1f}" if vals else "–")
            lines.append(f"| {c} | {src} | " + " | ".join(cells) + " | " + " | ".join(calls) + " |")
    (args.out / "localization.md").write_text(
        "Localization recall (%) and mean budgeted calls per task by category.\n"
        "Rows from the largest budget give truncation estimates at smaller budgets.\n\n" + "\n".join(lines) + "\n")
    print(f"{len(rows)} runs scored -> {args.out}")


if __name__ == "__main__":
    main()
