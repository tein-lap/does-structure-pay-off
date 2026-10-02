"""Part 4: leave-one-repository-out evaluation.

For each held-out repository, choose the configuration with the best mean
resolution on the other three repositories, then report how that choice does
on the held-out one. The gap estimates how much a choice depends on habits of
the repositories it was tuned on. The best configuration on the held-out
repository itself ("oracle") is shown for reference only.

Input: per_task.csv from score.py.

Usage:
    python experiments/loro.py --per-task results/per_task.csv --budget 100 [--configs A,B,C,D]
"""

import argparse
import csv
import statistics
from collections import defaultdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-task", required=True)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--configs", help="comma-separated configs to choose from (default: all)")
    parser.add_argument("--out")
    args = parser.parse_args()

    allowed = set(args.configs.split(",")) if args.configs else None
    score = defaultdict(list)  # (config, repo) -> resolved values
    with open(args.per_task) as fh:
        for r in csv.DictReader(fh):
            if int(r["budget"]) != args.budget or (allowed and r["config"] not in allowed):
                continue
            score[(r["config"], r["repo"])].append(int(r["resolved"]))
    configs = sorted({c for c, _ in score})
    repos = sorted({rp for _, rp in score})
    if len(repos) < 2:
        raise SystemExit("need runs from at least two repositories")

    def mean(config, repo_set):
        vals = [v for rp in repo_set for v in score.get((config, rp), [])]
        return statistics.mean(vals) if vals else float("nan")

    lines = ["| Held-out repository | Chosen config | In-repository resolution | Held-out resolution | Gap | Oracle on held-out |",
             "|---|---|---|---|---|---|"]
    for held in repos:
        train = [rp for rp in repos if rp != held]
        chosen = max(configs, key=lambda c: mean(c, train))
        inside, outside = mean(chosen, train), mean(chosen, [held])
        oracle = max(configs, key=lambda c: mean(c, [held]))
        lines.append(f"| {held} | {chosen} | {100 * inside:.1f} | {100 * outside:.1f} | "
                     f"{100 * (outside - inside):+.1f} | {oracle} ({100 * mean(oracle, [held]):.1f}) |")
    text = f"Leave-one-repository-out at {args.budget} calls (resolution %).\n\n" + "\n".join(lines) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
