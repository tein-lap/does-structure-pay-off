"""Part 1: what do the provided node embeddings encode? (CPU only, no agent)

structure  Spearman correlation between the cosine similarity of two nodes'
           embeddings and their shortest-path distance in the (undirected)
           graph. Strongly negative = the vectors mostly encode structure.

           Pairs with no path between them have no distance. They are left out
           of the main rho, and the share of such pairs is reported. As a
           sensitivity check, rho is also computed with those pairs set to
           (max distance + 1). DEFINED_IN edges make many distances short, so
           the distance histogram is reported too (many ties lower rho's
           resolution); --edge-types lets you rerun on CALLS/IMPORTS only.

gold       Gold-neighbour rate@k. For each function changed by a task's gold
           patch, how many of its k nearest embedding neighbours are other
           functions changed by the same patch? Compared with k random nodes
           and the k nearest nodes by graph distance (ties broken at random).

           Only tasks whose patch changes at least 2 embedded functions/classes
           can score above zero, so only those are used. Tasks whose removed
           lines do not match the graph's source (graph built at a different
           commit than the task's base) are dropped too. The number of tasks
           kept and dropped, and why, is always reported.

The competition's file formats are not documented in what we have read, so
the loaders accept several layouts; check them against the Data tab.

Keep competition data private: run this locally or in a private notebook and
publish only the numbers.

Usage:
    python -m dspo.probe structure --graph repo.pkl --emb repo_emb.npz [--pairs 50000]
    python -m dspo.probe gold --graph repo.pkl --emb repo_emb.npz --tasks tasks.jsonl --repo requests
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

from dspo.localization import GraphFiles, changed_locations, patch_alignment


# -------------------------------------------------------------------- loading
def load_graph(path: str | Path) -> nx.MultiDiGraph:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_embeddings(graph, path: str | Path | None = None, attr: str = "embedding",
                    ids_path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Node id -> unit-normalised vector.

    Accepted sources:
      - no path: the node attribute `attr` on the graph
      - .npz with arrays "ids" and "vectors" (or "embeddings")
      - .npy matrix plus --ids file (JSON list or one id per line), row i = ids[i]
      - .json / .pkl mapping node id -> vector
    """
    if path is None:
        raw = {n: d[attr] for n, d in graph.nodes(data=True) if d.get(attr) is not None}
    else:
        path = Path(path)
        if path.suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            vec_key = "vectors" if "vectors" in data else "embeddings"
            raw = dict(zip([str(i) for i in data["ids"]], data[vec_key], strict=True))
        elif path.suffix == ".npy":
            if ids_path is None:
                raise ValueError("a .npy matrix needs --ids with the node id of each row")
            text = Path(ids_path).read_text()
            ids = json.loads(text) if text.lstrip().startswith("[") else text.split()
            raw = dict(zip(ids, np.load(path), strict=True))
        elif path.suffix == ".json":
            raw = json.loads(path.read_text())
        else:
            with open(path, "rb") as fh:
                raw = pickle.load(fh)
    out = {}
    for node, vec in raw.items():
        v = np.asarray(vec, dtype=np.float64)
        norm = np.linalg.norm(v)
        if node in graph and norm > 0:
            out[node] = v / norm
    return out


def undirected(graph, edge_types: set[str] | None = None) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(graph.nodes)
    for u, v, d in graph.edges(data=True):
        if edge_types is None or d.get("type") in edge_types:
            g.add_edge(u, v)
    return g


# ------------------------------------------------------------------ structure
def structure_probe(graph, emb: dict[str, np.ndarray], n_pairs: int = 50_000, seed: int = 0,
                    edge_types: set[str] | None = None) -> dict:
    rng = random.Random(seed)
    g = undirected(graph, edge_types)
    nodes = sorted(n for n in emb if n in g)
    n = len(nodes)
    if n < 3:
        raise ValueError(f"only {n} nodes have embeddings")
    total = n * (n - 1) // 2
    if total <= n_pairs:
        pairs = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)]
    else:
        seen: set[tuple[str, str]] = set()
        while len(seen) < n_pairs:
            a, b = rng.sample(nodes, 2)
            seen.add((a, b) if a < b else (b, a))
        pairs = sorted(seen)

    by_source: dict[str, list[str]] = {}
    for a, b in pairs:
        by_source.setdefault(a, []).append(b)
    sims, dists = [], []
    for a, targets in by_source.items():
        lengths = nx.single_source_shortest_path_length(g, a)
        for b in targets:
            sims.append(float(emb[a] @ emb[b]))
            dists.append(lengths.get(b, -1))
    sims, dists = np.array(sims), np.array(dists)
    connected = dists >= 0
    rho, p = spearmanr(sims[connected], dists[connected]) if connected.sum() > 2 else (float("nan"), float("nan"))
    filled = np.where(connected, dists, (dists.max() if connected.any() else 0) + 1)
    rho_all, _ = spearmanr(sims, filled)
    return {
        "nodes_with_embeddings": n,
        "nodes_in_graph": graph.number_of_nodes(),
        "pairs": len(pairs),
        "disconnected_share": round(float(1 - connected.mean()), 4),
        "spearman_rho": round(float(rho), 4),
        "p_value": float(p),
        "spearman_rho_disconnected_as_max_plus_1": round(float(rho_all), 4),
        "distance_histogram": {int(k): int(v) for k, v in sorted(Counter(dists[connected].tolist()).items())},
        "edge_types": sorted(edge_types) if edge_types else "all",
        "seed": seed,
    }


# ----------------------------------------------------------------------- gold
def load_tasks(path: str | Path, repo: str | None = None, repo_key: str = "repo",
               patch_key: str = "patch") -> list[dict]:
    tasks = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            t = json.loads(line)
            if repo and repo.lower() not in str(t.get(repo_key, "")).lower():
                continue
            if patch_key not in t:
                raise KeyError(f"task has no '{patch_key}' field; fields are {sorted(t)} (use --patch-key)")
            tasks.append(t)
    return tasks


def _knn_embedding(node: str, ids: list[str], mat: np.ndarray, index: dict[str, int], k: int) -> list[str]:
    sims = mat @ mat[index[node]]
    sims[index[node]] = -np.inf
    top = np.argpartition(-sims, min(k, len(ids) - 1))[:k]
    return [ids[i] for i in top[np.argsort(-sims[top])]]


def _knn_graph(node: str, g: nx.Graph, candidates: set[str], k: int, rng: random.Random) -> list[str]:
    lengths = nx.single_source_shortest_path_length(g, node)
    by_dist: dict[int, list[str]] = {}
    for other, d in lengths.items():
        if other != node and other in candidates:
            by_dist.setdefault(d, []).append(other)
    out: list[str] = []
    for d in sorted(by_dist):
        group = by_dist[d]
        rng.shuffle(group)
        out.extend(group[: k - len(out)])
        if len(out) >= k:
            break
    return out


def gold_probe(graph, emb: dict[str, np.ndarray], tasks: list[dict], k: int = 10, seed: int = 0,
               patch_key: str = "patch", id_key: str = "instance_id", min_alignment: float = 0.9) -> dict:
    rng = random.Random(seed)
    gf = GraphFiles(graph)
    g = undirected(graph)
    candidates = {n for n in emb if graph.nodes[n].get("kind") in ("function", "class")}
    ids = sorted(candidates)
    index = {n: i for i, n in enumerate(ids)}
    mat = np.stack([emb[n] for n in ids])

    kept, dropped, unverified = [], Counter(), 0
    rates = {"embedding": [], "random": [], "graph": []}
    hits = {"embedding": [], "random": [], "graph": []}
    for t in tasks:
        matched, checked = patch_alignment(t[patch_key], gf)
        if checked and matched / checked < min_alignment:
            dropped["graph_not_at_base_commit"] += 1  # line numbers would point at the wrong code
            continue
        unverified += checked == 0  # additions only: alignment cannot be checked
        gold = changed_locations(t[patch_key], gf).functions & candidates
        if len(gold) < 2:
            dropped["fewer_than_2_gold_functions"] += 1
            continue
        kept.append(t.get(id_key))
        for node in sorted(gold):
            others = gold - {node}
            neighbours = {
                "embedding": _knn_embedding(node, ids, mat, index, k),
                "random": rng.sample([c for c in ids if c != node], min(k, len(ids) - 1)),
                "graph": _knn_graph(node, g, candidates, k, rng),
            }
            for name, nbrs in neighbours.items():
                found = len(set(nbrs) & others)
                rates[name].append(found / k)
                hits[name].append(float(found > 0))

    def mean(xs):
        return round(float(np.mean(xs)), 4) if xs else None

    return {
        "k": k,
        "tasks_total": len(tasks),
        "tasks_used": len(kept),
        "tasks_dropped": dict(dropped),
        "tasks_alignment_unverified": unverified,
        "min_alignment": min_alignment,
        "gold_functions_scored": len(rates["embedding"]),
        "gold_neighbour_rate": {name: mean(v) for name, v in rates.items()},
        "hit_rate": {name: mean(v) for name, v in hits.items()},
        "seed": seed,
    }


# ------------------------------------------------------------------------ cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("structure", "gold"):
        p = sub.add_parser(name)
        p.add_argument("--graph", required=True)
        p.add_argument("--emb", help="embedding file (omit to read the graph's node attribute)")
        p.add_argument("--emb-attr", default="embedding")
        p.add_argument("--ids", help="node ids for a .npy embedding matrix")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", help="write the JSON result here as well")
    sub.choices["structure"].add_argument("--pairs", type=int, default=50_000)
    sub.choices["structure"].add_argument("--edge-types", help="comma-separated, e.g. CALLS,IMPORTS")
    gp = sub.choices["gold"]
    gp.add_argument("--tasks", required=True)
    gp.add_argument("--repo", help="keep only tasks whose repo field contains this")
    gp.add_argument("--repo-key", default="repo")
    gp.add_argument("--patch-key", default="patch")
    gp.add_argument("--id-key", default="instance_id")
    gp.add_argument("-k", type=int, default=10)
    gp.add_argument("--min-alignment", type=float, default=0.9,
                    help="drop tasks whose removed lines match the graph source less often than this")
    args = parser.parse_args(argv)

    graph = load_graph(args.graph)
    emb = load_embeddings(graph, args.emb, args.emb_attr, args.ids)
    if args.cmd == "structure":
        edge_types = set(args.edge_types.split(",")) if args.edge_types else None
        result = structure_probe(graph, emb, args.pairs, args.seed, edge_types)
    else:
        tasks = load_tasks(args.tasks, args.repo, args.repo_key, args.patch_key)
        result = gold_probe(graph, emb, tasks, args.k, args.seed, args.patch_key, args.id_key, args.min_alignment)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
