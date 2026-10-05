from dspo.graph_builder import symbol_index
from dspo.symbols import index_symbols


def test_stdlib_index_matches_graph_builder(repo, graph):
    by_graph = sorted(symbol_index(graph), key=lambda r: r["id"])
    by_stdlib = sorted(index_symbols(repo), key=lambda r: r["id"])
    assert by_stdlib == by_graph
    assert {"pkg.client.Client.send", "pkg.utils.helper", "pkg"} <= {r["id"] for r in by_stdlib}
