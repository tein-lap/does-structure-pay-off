from dspo.localization import (GraphFiles, changed_lines, changed_locations, localization_scores, recall,
                               truncate, visited_from_events)

PATCH = """diff --git a/pkg/client.py b/pkg/client.py
--- a/pkg/client.py
+++ b/pkg/client.py
@@ -11,3 +11,4 @@ class Client(Base):
     def send(self, request):
-        data = helper(request)
+        data = helper(request).strip()
+        log(data)
         self.prepare(data)
diff --git a/tests/test_client.py b/tests/test_client.py
--- a/tests/test_client.py
+++ b/tests/test_client.py
@@ -1,2 +1,3 @@
 def test_send():
-    pass
+    assert True
+    assert 1
diff --git a/pkg/new.py b/pkg/new.py
new file mode 100644
--- /dev/null
+++ b/pkg/new.py
@@ -0,0 +1,2 @@
+def added():
+    return 1
"""


def test_changed_lines():
    lines = changed_lines(PATCH)
    assert lines["pkg/client.py"] == {12}          # removed line 12; additions attach to it
    assert lines["tests/test_client.py"] == {2}
    assert lines["pkg/new.py"] == set()            # new file: no original lines


def test_removed_line_starting_with_dashes_is_not_a_header():
    patch = "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,1 @@\n--- comment\n keep\n"
    assert changed_lines(patch) == {"x.py": {1}}


def test_changed_locations(graph):
    gold = changed_locations(PATCH, graph)
    assert gold.functions == {"pkg.client.Client.send"}
    assert "tests/test_client.py" not in gold.files   # tests excluded by default
    assert gold.files == {"pkg/client.py"}           # new file pkg/new.py could not be opened


def test_grep_mentions_do_not_count(graph):
    events = [
        {"tool": "grep", "args": {"pattern": "send"}, "result": "pkg/client.py:12: def send"},
        {"tool": "read_file", "args": {"path": "pkg/utils.py"}},
    ]
    seen = visited_from_events(events, graph)
    assert seen.files == {"pkg/utils.py"}
    assert "pkg.client.Client.send" not in seen.functions
    assert "pkg.utils.helper" in seen.functions        # whole file read


def test_ranged_read_and_graph_queries(graph):
    send = graph.nodes["pkg.client.Client.send"]
    events = [
        {"tool": "read_file", "args": {"path": "pkg/client.py", "start_line": send["lineno"], "end_line": send["lineno"] + 1}},
        {"tool": "get_code_neighbors", "args": {"symbol": "pkg.utils.parse_value"}},
        {"tool": "search_similar_code", "args": {"query": "make_obj"}},
    ]
    seen = visited_from_events(events, graph)
    assert "pkg.client.Client.send" in seen.functions
    assert "pkg.client.Client.prepare" not in seen.functions   # outside the read range
    assert {"pkg.utils.parse_value", "pkg.client.make_obj"} <= seen.functions


def test_truncate_skips_free_tools():
    events = [{"tool": "grep"}, {"tool": "get_status"}, {"tool": "read_file"}, {"tool": "submit_patch"}, {"tool": "grep"}]
    assert [e["tool"] for e in truncate(events, 2)] == ["grep", "get_status", "read_file", "submit_patch"]


def test_recall():
    assert recall(set(), {"a"}) is None
    assert recall({"a", "b"}, {"a", "c"}) == 0.5


def test_localization_scores_by_budget(graph):
    events = [{"tool": "grep", "args": {}}] * 30 + [{"tool": "read_file", "args": {"path": "pkg/client.py"}}]
    scores = localization_scores(PATCH, events, graph, budgets=(25, 50))
    assert scores["function_recall@25"] == 0.0
    assert scores["function_recall@50"] == 1.0
    assert scores["file_recall@50"] == 1.0
    assert isinstance(GraphFiles(graph).match_file("/abs/path/pkg/client.py"), str)


def test_patch_alignment_detects_wrong_commit(graph):
    from dspo.localization import patch_alignment
    good = "--- a/pkg/client.py\n+++ b/pkg/client.py\n@@ -12,1 +12,1 @@\n-        data = helper(request)\n+        x\n"
    stale = "--- a/pkg/client.py\n+++ b/pkg/client.py\n@@ -12,1 +12,1 @@\n-        something_else()\n+        x\n"
    assert patch_alignment(good, graph) == (1, 1)
    assert patch_alignment(stale, graph) == (0, 1)
    assert patch_alignment("--- a/pkg/client.py\n+++ b/pkg/client.py\n@@ -12,0 +13,1 @@\n+        x\n", graph) == (0, 0)


def test_run_command_is_split_into_explore_and_test():
    from dspo.localization import call_category
    assert call_category("run_command", {"command": "grep -rn parse_value pkg"}) == "explore"
    assert call_category("run_command", {"command": "git log --oneline -5"}) == "explore"
    assert call_category("run_command", {"command": "python -m pytest tests -x"}) == "test"
    assert call_category("read_file", {"filepath": "pkg/client.py"}) == "explore"
    assert call_category("get_code_neighbors", {"node": "Client"}) == "graph"


def test_shell_reads_count_as_opened(graph):
    from dspo.localization import visited_from_events
    seen = visited_from_events([{"tool": "run_command", "args": {"command": "cat pkg/utils.py"}}], graph)
    assert "pkg/utils.py" in seen.files and "pkg.utils.helper" in seen.functions
    grep = visited_from_events([{"tool": "run_command", "args": {"command": "grep -rn helper pkg/utils.py"}}], graph)
    assert not grep.files
    part = visited_from_events([{"tool": "run_command", "args": {"command": "sed -n '1,2p' pkg/utils.py"}}], graph)
    assert "pkg.utils.helper" in part.functions and "pkg.utils.very_unique_function_name" not in part.functions
