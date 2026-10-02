# Does Structure Pay Off?

**Helping small coding agents find and fix bugs.** Code for a paper in the Kaggle
*Gemma 4 Developer Agent Paper Track*.

Does the competition's code graph help a small, offline Gemma 4 agent find and fix
bugs when every graph query costs one of its 100 tool calls? This repository holds
everything needed to answer that question and to reproduce the paper's numbers.

## What is here

| Part of the paper | Component | File | Status |
|---|---|---|---|
| Resource | Graph builder: any Python repo → competition-style graph | `dspo/graph_builder.py` | Built, tested |
| Part 1 | Embedding probe: what do the 256-d vectors encode? | `dspo/probe.py` | Built, tested on synthetic embeddings |
| Part 2 | Localization scorer: did the agent look at the gold code? | `dspo/localization.py` | Built, tested on real fix commits |
| Part 2 | Arm definitions A–D and config generator | `agent/arms.yaml`, `agent/make_configs.py` | Built; needs the official `agent.yaml` template |
| Part 2 | Run scoring: resolution, recall, calls by category | `experiments/score.py` | Built, tested on fake runs |
| Part 3 | Issue-to-symbol bridge | `dspo/bridge.py`, `agent/skills/issue_to_symbols.py` | Built, tested |
| Part 3 | Offline bridge evaluation vs TF-IDF and random | `experiments/bridge_eval.py` | Built, run on 44 SWE-bench requests issues |
| Part 4 | Leave-one-repository-out analysis | `experiments/loro.py` | Built, tested on fake runs |
| All | Run manifest (config, seed, commit, hardware, time) | `dspo/runlog.py` | Built |

No agent experiments have been run yet, so there are no results yet.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                      # 29 tests
```

Python 3.10+. Dependencies are BSD or MIT licensed (see `requirements.txt`).

## Usage

### Build a graph from any Python repository

```bash
python -m dspo.graph_builder path/to/repo --out repo.pkl --symbols repo.symbols.json
```

Nodes are modules, classes and functions with `kind, name, qualname, module, file,
lineno, end_lineno, source`. Edges are `DEFINED_IN` (child → parent), `IMPORTS`
(module → module) and `CALLS` (function → function/class). Calls are resolved
exactly where possible (imports, relative imports, re-exports, `self.`/`cls.`
methods including inherited ones). Otherwise a unique-name match is used and
marked `resolution="heuristic"`, except for builtin-like names such as `update` or `get`.
Tests, docs and examples are skipped unless `--include-tests` is given.

Results on public upstream repositories (October 2026):

| Repository (commit) | Modules | Classes | Functions | DEFINED_IN | IMPORTS | CALLS (heuristic) | Calls resolved by exact rules |
|---|---|---|---|---|---|---|---|
| requests (`611c616`) | 19 | 52 | 247 | 299 | 80 | 238 (13) | 225 (95%) |
| httpx (`b5addb6`) | 23 | 87 | 434 | 521 | 87 | 371 (29) | 342 (92%) |

### Part 1: embedding probe (CPU only)

```bash
python -m dspo.probe structure --graph repo.pkl --emb repo_emb.npz
python -m dspo.probe gold --graph repo.pkl --emb repo_emb.npz --tasks tasks.jsonl --repo requests
```

- **structure:** Spearman ρ between cosine similarity and shortest-path distance
  over up to 50,000 node pairs (undirected graph). It also reports:
  - the share of pairs with no path (left out of ρ)
  - a sensitivity ρ with those pairs set to max + 1
  - the distance histogram, because short distances produce many ties

  `--edge-types CALLS,IMPORTS` reruns it without `DEFINED_IN`.
- **gold:** gold-neighbour rate@10 and hit rate, compared with random nodes and
  graph-nearest nodes. Only tasks whose patch changes **at least two** embedded
  functions can score above zero, so only those are used. The number of tasks
  dropped is always reported.

Sanity check on the requests graph: graph-smoothed embeddings give ρ = −0.64,
and random embeddings give ρ = −0.01.

The embedding and task formats are not confirmed yet. The loaders accept a node
attribute, `.npz` (`ids` + `vectors`), `.npy` + an id list, or `.json`/`.pkl`
mappings, and `--patch-key`/`--repo-key`/`--id-key` select the task fields.

### Part 2: tool ablation

1. Copy the official starter `agent.yaml` to `agent/template.yaml`.
2. Replace its tool list with `{{TOOLS}}`. Optionally add `{{BUDGET}}` and
   `{{BRIDGE_NOTE}}` to the instruction.
3. Put the real tool names from HARNESS_README into `agent/arms.yaml`.
4. Generate the configs, then run them on Kaggle:

```bash
python agent/make_configs.py --template agent/template.yaml --out agent/build
```

Arms differ **only** in exploration tools. Editing, testing and submitting tools
are the same in every arm:

| Arm | Exploration tools |
|---|---|
| A | `grep`, `read_file` |
| B | A + `get_code_neighbors`, `get_code_subgraph` |
| C | A + `search_similar_code` |
| D | A + all three graph tools |

Store each run as:

```
runs/<config>/b<budget>/s<seed>/<task_id>/events.jsonl   # {"tool": ..., "args": {...}} per line
runs/<config>/b<budget>/s<seed>/<task_id>/result.json    # {"resolved": true}
```

Then score the runs:

```bash
python experiments/select_subset.py --tasks tasks.jsonl --n 40 --seed 0 --out subset.json   # optional
python experiments/score.py --runs runs --tasks tasks.jsonl --graphs graphs/ --out results/
```

How the measurements are defined:

- **Localization** counts only files the agent *opened* (read-file calls) and
  symbols it *queried* (graph-tool arguments). Paths that merely appear in grep
  output do not count, since one broad grep would otherwise "visit" dozens of files.
- **Gold locations** are the innermost functions/classes containing the patch's
  changed lines, with the graph built at the task's **base** commit. New files and
  test files are excluded. Git's `@@ ... @@` hunk headers are not used by default:
  on real requests commits they often name the enclosing class or the *previous*
  function.
- **Alignment check:** line-based gold locations are right only if the graph was
  built at the task's base commit. Every removed patch line is compared with the
  graph's source at that line. `score.py` reports this as the `alignment` column,
  and `probe gold` drops tasks below `--min-alignment` (default 0.9). On real
  requests commits, the graph at the base commit matched 100% of removed lines
  and the graph at HEAD matched 0%.
- **Truncation:** recall at 25/50 calls can be estimated from a 100-call run by
  cutting it short. Resolution cannot: agents usually submit near the end, so a
  run cut at call 25 rarely contains a patch. Resolution is reported only for
  budgets that were really run.

### Part 3: issue-to-symbol bridge

```bash
python -m dspo.bridge --graph repo.pkl --issue issue.txt -k 5
python agent/skills/issue_to_symbols.py --symbols repo.symbols.json < issue.txt   # skill form
```

It extracts:

- traceback frames, located by file and line (deepest first)
- names in code formatting
- dotted paths
- `.py` paths
- CamelCase/snake_case identifiers

It keeps only names that exist in the graph and returns the top k. It uses only the
standard library, so it can run as a sandboxed skill. On a real `requests` traceback
(a URL without a scheme), the top pick is `requests.models.PreparedRequest.prepare_url`,
the function that raised the error.

Offline evaluation (no agent, no GPU) against the functions each gold patch changes,
compared with a TF-IDF word-matching baseline and exact random chance:

```bash
python experiments/bridge_eval.py --tasks tasks.jsonl --repo path/to/clone --out results/bridge
```

It works on any JSONL with `instance_id`, `base_commit`, `patch` and the issue text, so it
runs on SWE-bench rows now and on the competition's `tasks.jsonl` later.

### Part 4: leave-one-repository-out

```bash
python experiments/loro.py --per-task results/per_task.csv --budget 100
```

For each held-out repository, it chooses the configuration that is best on the other
three, then reports in-repository vs held-out resolution and the gap.

### Reproducibility

```python
from dspo.runlog import record_run
record_run("runs/manifest.jsonl", config={"arm": "D", "budget": 100, "task": "..."}, seed=0)
```

Each call records the UTC time, git commit (and whether the tree had changes), host,
platform, Python version and GPUs.

## Check against HARNESS_README before running agents

- [ ] Exact tool names (update `agent/arms.yaml`, `READ_TOOLS`/`GRAPH_TOOLS` in `dspo/localization.py`)
- [ ] Whether the tool-call budget can be set per config (25/50/100)
- [ ] How skills receive input and return output; whether the graph is readable in the sandbox
- [ ] Whether one skill call counts as one tool call (paper Section 3.3)
- [ ] Graph and embedding file formats; field names in `tasks.jsonl`
- [ ] How to export each run's tool-call log and its resolved/unresolved result

## Competition data

Competition files (tasks, graphs, embeddings) must never be published (rule 2.4b).
`data/`, `runs/`, `*.pkl`, `*.npy`, `*.npz` and `tasks.jsonl` are git-ignored.
Anything this repository releases is built from public upstream repositories.

## License

Apache License 2.0, see [LICENSE](LICENSE).
