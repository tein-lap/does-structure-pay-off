from dspo.graph_builder import summary, symbol_index


def calls(graph, src):
    return {v: d["resolution"] for _, v, d in graph.out_edges(src, data=True) if d["type"] == "CALLS"}


def test_nodes_and_attributes(graph):
    assert set(graph.nodes) >= {
        "pkg", "pkg.utils", "pkg.client", "pkg.base",
        "pkg.client.Client", "pkg.client.Client.send", "pkg.utils.helper",
    }
    send = graph.nodes["pkg.client.Client.send"]
    assert send["kind"] == "function"
    assert send["file"] == "pkg/client.py"
    assert send["source"].lstrip().startswith("def send")
    assert send["lineno"] < send["end_lineno"]


def test_tests_excluded_by_default(graph):
    assert not any("test_client" in n for n in graph.nodes)


def test_defined_in_and_imports(graph):
    assert graph.has_edge("pkg.client.Client.send", "pkg.client.Client", key="DEFINED_IN")
    assert graph.has_edge("pkg.client.Client", "pkg.client", key="DEFINED_IN")
    assert graph.has_edge("pkg.client", "pkg.utils", key="IMPORTS")
    assert graph.has_edge("pkg.client", "pkg.base", key="IMPORTS")


def test_exact_call_resolution(graph):
    send = calls(graph, "pkg.client.Client.send")
    assert send["pkg.utils.helper"] == "exact"            # from-import
    assert send["pkg.client.Client.prepare"] == "exact"   # self.method
    assert send["pkg.base.Base.finish"] == "exact"        # inherited method
    assert send["pkg.utils.parse_value"] == "exact"       # module.func
    prepare = calls(graph, "pkg.client.Client.prepare")
    assert prepare["pkg.client.make_obj"] == "exact"      # same-module function
    assert prepare["pkg.utils.parse_value"] == "exact"    # import a.b; a.b.func()
    assert calls(graph, "pkg.client.make_obj")["pkg.client.Client"] == "exact"  # constructor


def test_heuristic_and_generic_names(graph):
    prepare = calls(graph, "pkg.client.Client.prepare")
    assert prepare["pkg.utils.very_unique_function_name"] == "heuristic"
    assert not any(t.endswith(".update") for t in prepare)  # builtin-like name: no guess


def test_reexport_resolution(repo):
    from dspo.graph_builder import GraphBuilder
    b = GraphBuilder(repo)
    b.build()
    assert b.resolve("pkg.Client") == "pkg.client.Client"
    assert b.resolve("pkg.public_helper") == "pkg.utils.helper"


def test_summary_and_symbol_index(graph):
    s = summary(graph)
    assert s["nodes"]["module"] == 4
    assert s["edges"]["CALLS"] >= 8
    index = symbol_index(graph)
    assert {"id", "kind", "name", "file", "lineno"} <= set(index[0])
    assert all("source" not in r for r in index)
