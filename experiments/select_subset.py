"""Pick a fixed random subset of tasks, balanced across repositories (Section 8, Option 2).

The same subset must be used for every arm. The seed and the chosen ids are
written to the output so the paper can state exactly which tasks were run.

Usage:
    python experiments/select_subset.py --tasks tasks.jsonl --n 40 --seed 0 --out subset.json
"""

import argparse
import json
import random
from collections import defaultdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repo-key", default="repo")
    parser.add_argument("--id-key", default="instance_id")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    by_repo = defaultdict(list)
    with open(args.tasks, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                t = json.loads(line)
                by_repo[t[args.repo_key]].append(t[args.id_key])

    rng = random.Random(args.seed)
    repos = sorted(by_repo)
    for r in repos:
        by_repo[r].sort()
        rng.shuffle(by_repo[r])
    # Round-robin across repos so each gets an equal share (or all its tasks if it has fewer).
    chosen, i = [], 0
    while len(chosen) < args.n and any(i < len(by_repo[r]) for r in repos):
        for r in repos:
            if i < len(by_repo[r]) and len(chosen) < args.n:
                chosen.append(by_repo[r][i])
        i += 1

    counts = {r: sum(1 for c in chosen if c in set(by_repo[r])) for r in repos}
    with open(args.out, "w") as fh:
        json.dump({"seed": args.seed, "n": len(chosen), "per_repo": counts, "ids": chosen}, fh, indent=2)
    print(f"{len(chosen)} tasks: {counts}")


if __name__ == "__main__":
    main()
