import json

import networkx as nx
import numpy as np
import pytest

from dspo.probe import gold_probe, load_embeddings, structure_probe


def chain_graph(n=40):
    g = nx.MultiDiGraph()
    for i in range(n):
        g.add_node(f"m.f{i}", kind="function", name=f"f{i}", qualname=f"m.f{i}", module="m",
                   file="m.py", lineno=10 * i + 1, end_lineno=10 * i + 9)
    for i in range(n - 1):
        g.add_edge(f"m.f{i}", f"m.f{i+1}", key="CALLS", type="CALLS")
    return g


def test_structure_probe_detects_structure_and_noise():
    g = chain_graph()
    rng = np.random.default_rng(0)
    # Position on a circle arc: nearby nodes in the chain get similar vectors.
    angles = np.linspace(0, np.pi, 40)
    structured = {f"m.f{i}": np.array([np.cos(a), np.sin(a)]) for i, a in enumerate(angles)}
    noise = {f"m.f{i}": rng.normal(size=16) for i in range(40)}
    assert structure_probe(g, _norm(structured))["spearman_rho"] < -0.9
    assert abs(structure_probe(g, _norm(noise))["spearman_rho"]) < 0.2


def test_disconnected_pairs_reported():
    g = chain_graph(10)
    g.add_node("m.lonely", kind="function", name="lonely", qualname="m.lonely", module="m", file="m.py",
               lineno=500, end_lineno=501)
    rng = np.random.default_rng(1)
    emb = _norm({n: rng.normal(size=8) for n in g.nodes})
    res = structure_probe(g, emb)
    assert res["pairs"] == 55
    assert res["disconnected_share"] == pytest.approx(10 / 55, abs=1e-3)


def test_gold_probe_drops_single_function_tasks_and_finds_gold():
    g = chain_graph()
    # Embedding puts f3 and f30 (the gold pair) next to each other, far from the rest.
    rng = np.random.default_rng(2)
    emb = {n: rng.normal(size=32) for n in g.nodes}
    emb["m.f30"] = emb["m.f3"] + 0.01
    patch_two = ("--- a/m.py\n+++ b/m.py\n@@ -32,1 +32,1 @@\n-x\n+y\n"
                 "@@ -302,1 +302,1 @@\n-x\n+y\n")
    patch_one = "--- a/m.py\n+++ b/m.py\n@@ -52,1 +52,1 @@\n-x\n+y\n"
    tasks = [{"instance_id": "two", "patch": patch_two}, {"instance_id": "one", "patch": patch_one}]
    res = gold_probe(g, _norm(emb), tasks, k=5)
    assert res["tasks_used"] == 1
    assert res["tasks_dropped"] == {"fewer_than_2_gold_functions": 1}
    assert res["hit_rate"]["embedding"] == 1.0
    assert res["gold_neighbour_rate"]["embedding"] == pytest.approx(1 / 5)
    assert res["hit_rate"]["graph"] == 0.0     # 27 hops apart in the chain


def test_embedding_loaders(tmp_path):
    g = chain_graph(3)
    ids = list(g.nodes)
    vecs = np.eye(3)
    np.savez(tmp_path / "e.npz", ids=np.array(ids), vectors=vecs)
    np.save(tmp_path / "e.npy", vecs)
    (tmp_path / "ids.json").write_text(json.dumps(ids))
    a = load_embeddings(g, tmp_path / "e.npz")
    b = load_embeddings(g, tmp_path / "e.npy", ids_path=tmp_path / "ids.json")
    for n in ids:
        g.nodes[n]["embedding"] = vecs[ids.index(n)]
    c = load_embeddings(g)
    assert set(a) == set(b) == set(c) == set(ids)
    assert np.allclose(a[ids[1]], c[ids[1]])


def _norm(d):
    return {k: np.asarray(v, float) / np.linalg.norm(v) for k, v in d.items()}


def test_gold_probe_drops_tasks_from_another_commit():
    g = chain_graph()
    g.add_node("m", kind="module", name="m", qualname="m", module="m", file="m.py", lineno=1, end_lineno=400,
               source="\n".join(f"line {i}" for i in range(1, 401)))
    rng = np.random.default_rng(3)
    emb = _norm({n: rng.normal(size=8) for n in g.nodes})
    good = "--- a/m.py\n+++ b/m.py\n@@ -32,1 +32,1 @@\n-line 32\n+y\n@@ -302,1 +302,1 @@\n-line 302\n+y\n"
    stale = "--- a/m.py\n+++ b/m.py\n@@ -32,1 +32,1 @@\n-old text\n+y\n@@ -302,1 +302,1 @@\n-old text\n+y\n"
    res = gold_probe(g, emb, [{"instance_id": "a", "patch": good}, {"instance_id": "b", "patch": stale}], k=3)
    assert res["tasks_used"] == 1
    assert res["tasks_dropped"] == {"graph_not_at_base_commit": 1}
