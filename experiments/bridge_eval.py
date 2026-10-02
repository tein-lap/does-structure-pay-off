"""Offline evaluation of the issue-to-symbol bridge (no agent, no GPU).

For each task (issue text + gold patch + base commit), the repository is
checked out at the base commit, a graph is built, and the bridge's top-k
symbols are compared with the functions the gold patch changes.

Baselines on the same candidate pool (functions and classes):
    random   exact chance: P(at least one gold in k random picks)
    tfidf    lexical retrieval: TF-IDF cosine between the issue text and each
             function's source, with identifiers split into sub-words (a cheap
             stand-in for "grep the issue words")

Metrics per task (k = 1, 5):
    func_hit@k   a gold function/class is among the top k symbols
    file_hit@k   a top-k symbol lies in a gold file
    func_recall@5  share of gold functions in the top 5
    empty        the bridge found no symbol at all

Works on any JSONL with instance_id, base_commit, patch and the issue text
(--issue-key), e.g. SWE-bench rows or the competition's tasks.jsonl.

Usage:
    python experiments/bridge_eval.py --tasks tasks.jsonl --repo path/to/clone --out results/bridge
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dspo.bridge import TRACEBACK_FRAME, SymbolIndex, bridge  # noqa: E402
from dspo.graph_builder import GraphBuilder  # noqa: E402
from dspo.localization import GraphFiles, changed_locations, patch_alignment  # noqa: E402

KS = (1, 5)
SUBWORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def tokens(text: str) -> list[str]:
    out = []
    for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        low = ident.lower()
        out.append(low)
        parts = [p.lower() for p in SUBWORD.findall(ident)]
        if len(parts) > 1:
            out.extend(parts)
    return out


def checkout(repo: Path, commit: str, dest: str) -> None:
    subprocess.run(f"git -C {repo} archive {commit} | tar -x -C {dest}", shell=True, check=True,
                   stderr=subprocess.DEVNULL)


def random_hit(n_pool: int, n_gold: int, k: int) -> float:
    """Exact probability that k random picks (without replacement) include a gold one."""
    if n_gold == 0 or n_pool == 0:
        return 0.0
    k = min(k, n_pool)
    return 1 - math.comb(n_pool - n_gold, k) / math.comb(n_pool, k)


def bootstrap_ci(values: list[float], seed: int = 0, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, float)
    means = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_task(task: dict, repo: Path, issue_key: str) -> dict:
    row = {"instance_id": task["instance_id"]}
    issue = task[issue_key] or ""
    row["has_traceback"] = int(bool(TRACEBACK_FRAME.search(issue)))
    row["has_code_formatting"] = int("`" in issue)
    with tempfile.TemporaryDirectory() as d:
        checkout(repo, task["base_commit"], d)
        builder = GraphBuilder(Path(d))
        graph = builder.build()
    row["parse_errors"] = len(builder.parse_errors)
    gf = GraphFiles(graph)
    matched, checked = patch_alignment(task["patch"], gf)
    row["alignment"] = round(matched / checked, 3) if checked else ""
    gold = changed_locations(task["patch"], gf)
    row["gold_functions"] = len(gold.functions)
    row["gold_files"] = len(gold.files)
    if not gold.functions:
        row["status"] = "no_gold_function"
        return row
    if checked and matched / checked < 0.9:
        row["status"] = "misaligned"
        return row
    row["status"] = "ok"

    pool = [n for n, a in graph.nodes(data=True) if a["kind"] in ("function", "class")]
    row["pool"] = len(pool)

    # Bridge
    cands = bridge(issue, SymbolIndex.from_graph(graph), k=max(KS))
    ids = [c.symbol.id for c in cands]
    files = [c.symbol.file for c in cands]
    row["empty"] = int(not cands)
    row["n_candidates"] = len(cands)
    for k in KS:
        row[f"bridge_func_hit@{k}"] = int(bool(set(ids[:k]) & gold.functions))
        row[f"bridge_file_hit@{k}"] = int(bool(set(files[:k]) & gold.files))
    row["bridge_func_recall@5"] = len(set(ids[:5]) & gold.functions) / len(gold.functions)
    first = next((c for c in cands if c.symbol.id in gold.functions), None)
    row["bridge_hit_reason"] = first.reasons[0].split()[0] if first else ""

    # TF-IDF lexical baseline
    docs = [graph.nodes[n]["source"] for n in pool]
    vec = TfidfVectorizer(tokenizer=tokens, lowercase=False, token_pattern=None, sublinear_tf=True)
    mat = vec.fit_transform(docs)
    scores = (mat @ vec.transform([issue]).T).toarray().ravel()
    order = [pool[i] for i in np.argsort(-scores, kind="stable")]
    for k in KS:
        top = order[:k]
        row[f"tfidf_func_hit@{k}"] = int(bool(set(top) & gold.functions))
        row[f"tfidf_file_hit@{k}"] = int(bool({graph.nodes[n]["file"] for n in top} & gold.files))
    row["tfidf_func_recall@5"] = len(set(order[:5]) & gold.functions) / len(gold.functions)

    # Exact random baseline
    for k in KS:
        row[f"random_func_hit@{k}"] = round(random_hit(len(pool), len(gold.functions & set(pool)), k), 4)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--repo", type=Path, required=True, help="git clone with full history")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--issue-key", default="problem_statement")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)

    tasks = [json.loads(line) for line in open(args.tasks, encoding="utf-8") if line.strip()]
    rows = []
    for i, t in enumerate(tasks, 1):
        row = evaluate_task(t, args.repo, args.issue_key)
        rows.append(row)
        print(f"[{i}/{len(tasks)}] {row['instance_id']}: {row['status']}", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out / "per_task.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    status = Counter(r["status"] for r in rows)

    def cell(key, subset):
        vals = [r[key] for r in subset]
        if not vals:
            return "–"
        lo, hi = bootstrap_ci(vals, args.seed)
        return f"{100 * statistics.mean(vals):.1f} [{100 * lo:.0f}–{100 * hi:.0f}]"

    lines = [f"Tasks: {len(rows)} total; " + ", ".join(f"{k}: {v}" for k, v in sorted(status.items())),
             f"Scored: {len(ok)} tasks. Mean candidate pool: {statistics.mean(r['pool'] for r in ok):.0f} "
             f"functions/classes. Mean gold functions per task: {statistics.mean(r['gold_functions'] for r in ok):.2f}.",
             "",
             "Percent of tasks (95% bootstrap CI).",
             "",
             "| Method | Function hit@1 | Function hit@5 | File hit@1 | File hit@5 | Function recall@5 |",
             "|---|---|---|---|---|---|"]
    for m, name in (("bridge", "Issue-to-symbol bridge"), ("tfidf", "TF-IDF lexical baseline")):
        lines.append(f"| {name} | {cell(f'{m}_func_hit@1', ok)} | {cell(f'{m}_func_hit@5', ok)} | "
                     f"{cell(f'{m}_file_hit@1', ok)} | {cell(f'{m}_file_hit@5', ok)} | {cell(f'{m}_func_recall@5', ok)} |")
    rnd = lambda k: f"{100 * statistics.mean(r[f'random_func_hit@{k}'] for r in ok):.1f}"  # noqa: E731
    lines.append(f"| Random (exact) | {rnd(1)} | {rnd(5)} | – | – | – |")

    lines += ["", "Bridge by issue type (function hit@5):", "",
              "| Subset | Tasks | Bridge | TF-IDF | Bridge found nothing |", "|---|---|---|---|---|"]
    subsets = {
        "All scored": ok,
        "With traceback": [r for r in ok if r["has_traceback"]],
        "Without traceback": [r for r in ok if not r["has_traceback"]],
        "With code formatting": [r for r in ok if r["has_code_formatting"]],
        "Without code formatting": [r for r in ok if not r["has_code_formatting"]],
    }
    for name, sub in subsets.items():
        if sub:
            empty = 100 * statistics.mean(r["empty"] for r in sub)
            lines.append(f"| {name} | {len(sub)} | {cell('bridge_func_hit@5', sub)} | "
                         f"{cell('tfidf_func_hit@5', sub)} | {empty:.0f}% |")
    reasons = Counter(r["bridge_hit_reason"] for r in ok if r["bridge_hit_reason"])
    lines += ["", "Where the bridge's first correct symbol came from: " +
              ", ".join(f"{k} {v}" for k, v in reasons.most_common())]
    both = sum(1 for r in ok if r["bridge_func_hit@5"] and r["tfidf_func_hit@5"])
    only_b = sum(1 for r in ok if r["bridge_func_hit@5"] and not r["tfidf_func_hit@5"])
    only_t = sum(1 for r in ok if r["tfidf_func_hit@5"] and not r["bridge_func_hit@5"])
    union = sum(1 for r in ok if r["bridge_func_hit@5"] or r["tfidf_func_hit@5"])
    lines += [f"Function hit@5 overlap: both {both}, bridge only {only_b}, TF-IDF only {only_t}, "
              f"either {union} of {len(ok)}."]
    text = "\n".join(lines) + "\n"
    (args.out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
