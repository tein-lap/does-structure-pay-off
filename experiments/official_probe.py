"""Part 1 (embedding probe) on the competition's official data files.

The official files (Kaggle Data tab) use their own formats:

    tasks.jsonl                 instance_id, repo, base_commit, problem_statement, patch, ...
    graphs/<instance_id>.json   node-link JSON: nodes[*].{id, name, text}, edges[*].{source, target, type, key}
    embeddings/<instance_id>.npz  one float32[256] array per node, stored as "<node_id>.npy"
    snapshots/<instance_id>.tgz   the repository at base_commit

The official graphs have no file paths or line numbers, so the functions a gold
patch changes cannot be read from them directly. For that step only, this
script unpacks the task's snapshot, builds our own graph (which has line
numbers), finds the changed functions, and maps them to official node ids.
Everything else (embeddings, graph distances, candidate nodes) uses the
official files.

Two measurements per repository:

structure  For every task graph of the repository: sample source nodes, run one
           BFS from each, pair it with random targets (up to --pairs pairs per
           graph), and compute the Spearman correlation between embedding
           cosine similarity and undirected shortest-path distance. Reports the
           median and range over graphs, plus the share of disconnected pairs.

gold       Gold-neighbour rate@k and hit rate@k for each task whose patch
           changes at least two embedded functions/classes, compared with k
           random nodes and the k graph-nearest nodes.

Keep competition data private: run this in a PRIVATE Kaggle notebook (data is
already attached there) or locally, and publish only the printed numbers.

Usage (Kaggle notebook):
    python experiments/official_probe.py --data /kaggle/input/gemma-4-developer-agent --repo requests
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dspo.graph_builder import build_graph  # noqa: E402
from dspo.localization import GraphFiles, changed_locations, patch_alignment  # noqa: E402

# ---------------------------------------------------------------- data files


def find_data_dir(given: str | None) -> Path:
    if given:
        p = Path(given)
        if (p / "tasks.jsonl").exists():
            return p
        hits = list(p.rglob("tasks.jsonl"))
        if hits:
            return hits[0].parent
        raise FileNotFoundError(f"no tasks.jsonl under {p}")
    for root in (Path("/kaggle/input"), Path.cwd()):
        hits = list(root.rglob("tasks.jsonl")) if root.exists() else []
        if hits:
            return hits[0].parent
    raise FileNotFoundError("could not find tasks.jsonl; pass --data")


def load_tasks(data: Path, repo: str | None) -> list[dict]:
    tasks = []
    for line in (data / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            if repo is None or repo.lower() in str(t.get("repo", "")).lower() \
                    or str(t.get("instance_id", "")).lower().startswith(repo.lower() + "_"):
                tasks.append(t)
    return tasks


class FileIndex:
    """Find a task's graph / embedding / snapshot file under any of its known names.

    The data description says task-named files (<instance_id>.json) are hard-linked
    to commit-named files (<repo_short>_<base_commit>.json). Hard links may not
    survive uploading, and files may sit in sub-folders, so every file under the
    data folder is indexed once and looked up by instance id or base commit.
    """

    def __init__(self, data: Path):
        self.data = data
        self.by_ext: dict[str, list[Path]] = {}
        for p in data.rglob("*"):
            if p.is_file():
                ext = "".join(p.suffixes[-1:])
                self.by_ext.setdefault(ext, []).append(p)

    def find(self, task: dict, folder: str, ext: str) -> Path | None:
        iid = str(task.get("instance_id", ""))
        commit = str(task.get("base_commit", ""))
        short = str(task.get("repo", "")).split("/")[-1].lower() or iid.split("_")[0]
        files = self.by_ext.get(ext, [])
        in_folder = [p for p in files if folder in p.parts] or files
        names = [f"{iid}{ext}", f"{short}_{commit}{ext}", f"{short}_{commit[:7]}{ext}", f"{commit}{ext}"]
        for name in names:
            for p in in_folder:
                if p.name == name:
                    return p
        if commit:
            hits = [p for p in in_folder if commit in p.name]
            if len(hits) == 1:
                return hits[0]
        return None

    def sample(self, folder: str, n: int = 5) -> list[str]:
        out = [str(p.relative_to(self.data)) for ps in self.by_ext.values() for p in ps if folder in p.parts]
        return sorted(out)[:n]


def kind_from_text(text: str) -> str:
    head = (text or "").lstrip()
    while head.startswith("@"):                       # skip decorators
        head = head.split("\n", 1)[1].lstrip() if "\n" in head else ""
    if head.startswith(("def ", "async def ")):
        return "function"
    if head.startswith("class "):
        return "class"
    return "module"


def load_official_graph(path: Path) -> nx.MultiDiGraph:
    data = json.loads(path.read_text(encoding="utf-8"))
    g = nx.MultiDiGraph(**(data.get("graph") or {}))
    for n in data.get("nodes", []):
        nid = n["id"]
        attrs = {k: v for k, v in n.items() if k != "id"}
        attrs.setdefault("kind", kind_from_text(n.get("text", "")))
        g.add_node(nid, **attrs)
    for e in data.get("edges", data.get("links", [])):
        g.add_edge(e["source"], e["target"], key=e.get("key"), type=str(e.get("type", "")).upper())
    return g


def load_official_embeddings(path: Path, g: nx.MultiDiGraph) -> dict[str, np.ndarray]:
    out = {}
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.keys())
        if "ids" in keys and ("vectors" in keys or "embeddings" in keys):   # alternative layout
            vec = z["vectors"] if "vectors" in keys else z["embeddings"]
            pairs = zip([str(i) for i in z["ids"]], vec)
        else:
            pairs = ((k[:-4] if k.endswith(".npy") else k, z[k]) for k in keys)
        for nid, v in pairs:
            v = np.asarray(v, dtype=np.float64).ravel()
            norm = np.linalg.norm(v)
            if nid in g and norm > 0:
                out[nid] = v / norm
    return out


def undirected(g: nx.MultiDiGraph) -> nx.Graph:
    u = nx.Graph()
    u.add_nodes_from(g.nodes)
    u.add_edges_from((a, b) for a, b in g.edges())
    return u


# ---------------------------------------------------------------- structure


def structure_one(g: nx.MultiDiGraph, emb: dict, n_pairs: int, rng: random.Random) -> dict:
    ug = undirected(g)
    nodes = sorted(n for n in emb if n in ug)
    if len(nodes) < 10:
        return {}
    per_source = 100
    sources = rng.sample(nodes, min(len(nodes), max(1, n_pairs // per_source)))
    sims, dists = [], []
    for s in sources:
        lengths = nx.single_source_shortest_path_length(ug, s)
        targets = rng.sample(nodes, min(len(nodes), per_source + 1))
        for t in targets:
            if t == s:
                continue
            sims.append(float(emb[s] @ emb[t]))
            dists.append(lengths.get(t, -1))
    sims, dists = np.array(sims), np.array(dists)
    conn = dists >= 0
    rho = spearmanr(sims[conn], dists[conn])[0] if conn.sum() > 2 else float("nan")
    return {"nodes": len(nodes), "pairs": int(len(sims)), "rho": float(rho),
            "disconnected_share": float(1 - conn.mean()),
            "hist": Counter(dists[conn].tolist())}


# --------------------------------------------------------------------- gold


def unpack_snapshot(tgz: Path, into: Path) -> Path:
    with tarfile.open(tgz) as tf:
        try:
            tf.extractall(into, filter="data")
        except TypeError:                              # Python < 3.12
            tf.extractall(into)
    entries = [p for p in into.iterdir() if not p.name.startswith(".")]
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else into


def map_to_official(names: set[str], official: set[str]) -> set[str]:
    out = set()
    for n in names:
        if n in official:
            out.add(n)
            continue
        hits = [o for o in official if o.endswith("." + n) or n.endswith("." + o)]
        if len(hits) == 1:
            out.add(hits[0])
    return out


def knn(node, ids, mat, index, k):
    sims = mat @ mat[index[node]]
    sims[index[node]] = -np.inf
    top = np.argpartition(-sims, min(k, len(ids) - 1))[:k]
    return [ids[i] for i in top[np.argsort(-sims[top])]]


def graph_knn(node, ug, cands, k, rng):
    by_d: dict[int, list[str]] = {}
    for other, d in nx.single_source_shortest_path_length(ug, node).items():
        if other != node and other in cands:
            by_d.setdefault(d, []).append(other)
    out = []
    for d in sorted(by_d):
        grp = by_d[d]
        rng.shuffle(grp)
        out.extend(grp[: k - len(out)])
        if len(out) >= k:
            break
    return out


def gold_one(task, g, emb, index: "FileIndex", k: int, rng: random.Random, min_alignment: float):
    tgz = index.find(task, "snapshots", ".tgz")
    if tgz is None:
        return None, "no_snapshot"
    with tempfile.TemporaryDirectory() as tmp:
        root = unpack_snapshot(tgz, Path(tmp))
        ours = build_graph(root)
        gf = GraphFiles(ours)
        matched, checked = patch_alignment(task["patch"], gf)
        if checked and matched / checked < min_alignment:
            return None, "snapshot_not_at_base_commit"
        changed = changed_locations(task["patch"], gf)
        names = set(changed.functions) | set(getattr(changed, "classes", set()))
    cands = {n for n in emb if g.nodes[n].get("kind") in ("function", "class")}
    gold = map_to_official(names, cands)
    if len(gold) < 2:
        return None, "fewer_than_2_gold_functions"
    ids = sorted(cands)
    index = {n: i for i, n in enumerate(ids)}
    mat = np.stack([emb[n] for n in ids])
    ug = undirected(g)
    res = {"embedding": [], "random": [], "graph": []}
    for node in sorted(gold):
        others = gold - {node}
        nb = {"embedding": knn(node, ids, mat, index, k),
              "random": rng.sample([c for c in ids if c != node], min(k, len(ids) - 1)),
              "graph": graph_knn(node, ug, cands, k, rng)}
        for name, lst in nb.items():
            found = len(set(lst) & others)
            res[name].append((found / k, float(found > 0)))
    return res, "used"


# ---------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="folder containing tasks.jsonl, graphs/, embeddings/, snapshots/")
    ap.add_argument("--repo", help="e.g. requests, httpx, rich, fastapi (default: all)")
    ap.add_argument("--pairs", type=int, default=50_000, help="node pairs per graph for the structure probe")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-alignment", type=float, default=0.9)
    ap.add_argument("--max-tasks", type=int, help="limit the number of tasks (quick test)")
    ap.add_argument("--skip-gold", action="store_true")
    ap.add_argument("--out", help="also write the JSON result here")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    data = find_data_dir(args.data)
    tasks = load_tasks(data, args.repo)
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    print(f"data: {data}  tasks: {len(tasks)}", file=sys.stderr)

    structure, gold_rows, status = [], {"embedding": [], "random": [], "graph": []}, Counter()
    seen_graphs = set()
    index = FileIndex(data)
    warned = False
    print(f"indexed files: { {k: len(v) for k, v in index.by_ext.items()} }", file=sys.stderr)
    t0 = time.time()
    for i, t in enumerate(tasks, 1):
        gpath = index.find(t, "graphs", ".json")
        epath = index.find(t, "embeddings", ".npz")
        if gpath is None or epath is None:
            status["missing_graph" if gpath is None else "missing_embeddings"] += 1
            if not warned:
                warned = True
                print(f"  could not find files for {t['instance_id']} (base_commit {str(t.get('base_commit'))[:12]})", file=sys.stderr)
                for folder in ("graphs", "embeddings", "snapshots"):
                    print(f"  sample of {folder}/: {index.sample(folder)}", file=sys.stderr)
            continue
        g = load_official_graph(gpath)
        emb = load_official_embeddings(epath, g)
        key = (t.get("repo"), t.get("base_commit"))
        if key not in seen_graphs:                      # one structure probe per commit
            seen_graphs.add(key)
            s = structure_one(g, emb, args.pairs, rng)
            if s:
                structure.append(s)
        if not args.skip_gold:
            res, why = gold_one(t, g, emb, index, args.k, rng, args.min_alignment)
            status[why] += 1
            if res:
                for name, vals in res.items():
                    gold_rows[name].extend(vals)
        print(f"  [{i}/{len(tasks)}] {t['instance_id']}  ({time.time() - t0:.0f}s)", file=sys.stderr)

    rhos = [s["rho"] for s in structure if s["rho"] == s["rho"]]
    hist = Counter()
    for s in structure:
        hist.update(s["hist"])
    result = {
        "repo": args.repo or "all",
        "tasks": len(tasks),
        "structure": {
            "graphs": len(structure),
            "max_pairs_per_graph": args.pairs,
            "median_pairs_per_graph": int(statistics.median([s["pairs"] for s in structure])) if structure else None,
            "median_spearman_rho": round(statistics.median(rhos), 4) if rhos else None,
            "min_rho": round(min(rhos), 4) if rhos else None,
            "max_rho": round(max(rhos), 4) if rhos else None,
            "median_nodes_with_embeddings": int(statistics.median([s["nodes"] for s in structure])) if structure else None,
            "median_disconnected_share": round(statistics.median([s["disconnected_share"] for s in structure]), 4) if structure else None,
            "distance_histogram_all_graphs": {int(d): int(c) for d, c in sorted(hist.items())},
        },
    }
    if not args.skip_gold:
        def m(rows, j):
            return round(float(np.mean([r[j] for r in rows])), 4) if rows else None
        result["gold"] = {
            "k": args.k,
            "task_status": dict(status),
            "gold_functions_scored": len(gold_rows["embedding"]),
            "gold_neighbour_rate": {n: m(r, 0) for n, r in gold_rows.items()},
            "hit_rate": {n: m(r, 1) for n, r in gold_rows.items()},
        }
    result["seed"] = args.seed
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
