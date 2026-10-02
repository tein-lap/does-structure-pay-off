from dspo.bridge import SymbolIndex, bridge, format_for_agent, strip_traceback_frames
from dspo.graph_builder import symbol_index


def ids(cands):
    return [c.symbol.id for c in cands]


def test_traceback_deepest_frame_first(graph):
    line_send = graph.nodes["pkg.client.Client.send"]["lineno"] + 2
    line_parse = graph.nodes["pkg.utils.parse_value"]["lineno"] + 1
    issue = f"""Sending fails:

Traceback (most recent call last):
  File "/home/u/app.py", line 3, in <module>
    client.send(req)
  File "/usr/lib/python3/site-packages/pkg/client.py", line {line_send}, in send
    data = helper(request)
  File "/usr/lib/python3/site-packages/pkg/utils.py", line {line_parse}, in parse_value
    return int(x)
ValueError: invalid literal for int()
"""
    top = ids(bridge(issue, SymbolIndex.from_graph(graph), k=3))
    assert top[:2] == ["pkg.utils.parse_value", "pkg.client.Client.send"]


def test_code_formatting_beats_plain_identifiers(graph):
    issue = "Calling `Client.prepare` breaks, maybe because of parse_value."
    top = ids(bridge(issue, SymbolIndex.from_graph(graph), k=5))
    assert top[0] == "pkg.client.Client.prepare"
    assert "pkg.utils.parse_value" in top


def test_unknown_names_are_dropped(graph):
    assert bridge("Nothing here matches `frobnicate_widget`.", SymbolIndex.from_graph(graph)) == []
    assert "Start with grep" in format_for_agent([])


def test_file_paths(graph):
    top = ids(bridge("The bug is somewhere in pkg/base.py", SymbolIndex.from_graph(graph)))
    assert top == ["pkg.base"]


def test_json_index_gives_same_result(graph):
    issue = "`Client.send` and `helper` misbehave"
    from_graph = ids(bridge(issue, SymbolIndex.from_graph(graph)))
    from_json = ids(bridge(issue, SymbolIndex.from_records(symbol_index(graph))))
    assert from_graph == from_json and from_graph


def test_strip_traceback_frames_removes_echoed_code():
    text = 'x\n  File "a.py", line 1, in f\n    prepare_request(req)\nValueError: bad\n'
    out = strip_traceback_frames(text)
    assert "prepare_request" not in out and "ValueError" in out
