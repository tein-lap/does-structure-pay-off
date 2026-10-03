"""Localization scoring: did the agent look at the code the gold patch changes?

Gold side
    changed_lines(patch)            unified diff -> original line numbers per file
    changed_locations(patch, graph) -> gold files and innermost functions/classes

    Build the graph at the task's BASE commit (before the fix) so line numbers
    match. use_hunk_headers=True also adds the names git writes in "@@ ... @@"
    headers, but treat that only as a last resort: git labels a hunk with the
    last unindented line BEFORE it, which is often the enclosing class or even
    the previous function (checked on real requests fix commits).

Agent side
    visited_from_events(events, graph) -> files and functions the agent OPENED
    or QUERIED. Only read-file calls and graph-tool arguments count. Paths that
    merely appear in grep output or in the agent's reasoning do not: counting
    every mention would let one broad grep "visit" dozens of files and inflate
    recall.

    Tool names follow the competition harness (HARNESS_README section 6): there is
    no separate grep tool, so searches and test runs both go through run_command,
    and call_category() looks at the command to tell them apart. Shell reads such
    as `cat file`, `head`, `tail` or `sed -n 'a,bp' file` count as opening a file.

Budget truncation
    truncate(events, k) keeps the events within the first k budgeted tool calls.
    Use it for localization at smaller budgets only. It cannot estimate
    resolution: an agent with 100 calls usually submits its patch near the end,
    so a run cut at call 25 rarely contains a patch at all.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

READ_TOOLS = {"read_file", "view_file", "open_file", "read", "view", "cat"}
GRAPH_TOOLS = {"get_code_neighbors", "get_code_subgraph", "search_similar_code"}
SEARCH_TOOLS = {"grep", "search", "find_files", "list_files", "ls", "glob"}
EDIT_TOOLS = {"write_file", "edit_file", "apply_patch", "str_replace", "replace_in_file", "create_file"}
TEST_TOOLS = {"run_tests", "bash", "shell", "execute", "python"}   # run_command is split by call_category()
FREE_TOOLS = {"submit_patch", "get_status"}  # do not count against the 100-call budget

PATH_ARGS = ("path", "file", "file_path", "filename", "filepath")
SYMBOL_ARGS = ("symbol", "node", "node_id", "name", "query", "qualname", "symbol_name", "center", "nodes", "symbols")
START_ARGS = ("start_line", "start", "line_start", "offset", "from_line")
END_ARGS = ("end_line", "end", "line_end", "to_line")
LIMIT_ARGS = ("limit", "num_lines", "lines")

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")
HEADER_DEF = re.compile(r"(?:async\s+)?(?:def|class)\s+(\w+)")


@dataclass
class Locations:
    files: set[str] = field(default_factory=set)      # paths as they appear in the graph
    functions: set[str] = field(default_factory=set)  # innermost function/class node ids


# ------------------------------------------------------------------ gold side
def changed_lines(patch: str) -> dict[str, set[int]]:
    """Original-file line numbers touched by a unified diff, per file.

    Removed lines count as themselves. Added lines are attached to the original
    line just before them (or the hunk's first line when they open the hunk).
    New files (--- /dev/null) are recorded with an empty set.
    """
    result: dict[str, set[int]] = {}
    old_file = current = None
    old_pos = hunk_start = 0
    old_left = new_left = 0  # lines remaining in the current hunk
    for line in patch.splitlines():
        if old_left > 0 or new_left > 0:
            if line.startswith("-"):
                if current:
                    result[current].add(old_pos)
                old_pos += 1
                old_left -= 1
            elif line.startswith("+"):
                if current:
                    result[current].add(max(old_pos - 1, hunk_start, 1))
                new_left -= 1
            elif line.startswith(" ") or line == "":
                old_pos += 1
                old_left -= 1
                new_left -= 1
            continue  # "\ No newline at end of file" and similar
        if line.startswith("--- "):
            old_file = _strip_prefix(line[4:])
            continue
        if line.startswith("+++ "):
            new_file = _strip_prefix(line[4:])
            if new_file == "/dev/null":  # deleted file
                current = old_file
            elif old_file == "/dev/null":  # new file: nothing existed before
                current = None
                result.setdefault(new_file, set())
            else:
                current = old_file
            if current:
                result.setdefault(current, set())
            continue
        m = HUNK.match(line)
        if m:
            old_pos = hunk_start = int(m.group(1))
            old_left = int(m.group(2)) if m.group(2) is not None else 1
            new_left = int(m.group(4)) if m.group(4) is not None else 1
    return result


def removed_lines(patch: str) -> dict[str, dict[int, str]]:
    """Text of each removed line, keyed by file and original line number."""
    result: dict[str, dict[int, str]] = defaultdict(dict)
    old_file = current = None
    old_pos = 0
    old_left = new_left = 0
    for line in patch.splitlines():
        if old_left > 0 or new_left > 0:
            if line.startswith("-"):
                if current:
                    result[current][old_pos] = line[1:]
                old_pos += 1
                old_left -= 1
            elif line.startswith("+"):
                new_left -= 1
            elif line.startswith(" ") or line == "":
                old_pos += 1
                old_left -= 1
                new_left -= 1
            continue
        if line.startswith("--- "):
            old_file = _strip_prefix(line[4:])
        elif line.startswith("+++ "):
            current = None if old_file == "/dev/null" else old_file
        elif (m := HUNK.match(line)):
            old_pos = int(m.group(1))
            old_left = int(m.group(2)) if m.group(2) is not None else 1
            new_left = int(m.group(4)) if m.group(4) is not None else 1
    return result


def hunk_header_names(patch: str) -> dict[str, set[str]]:
    """Function/class names from git's hunk headers ("@@ -1,2 +1,2 @@ def foo(...)")."""
    result: dict[str, set[str]] = defaultdict(set)
    current = None
    for line in patch.splitlines():
        if line.startswith("+++ "):
            current = _strip_prefix(line[4:])
        m = HUNK.match(line)
        if m and current:
            d = HEADER_DEF.search(m.group(5) or "")
            if d:
                result[current].add(d.group(1))
    return result


def _strip_prefix(path: str) -> str:
    path = path.split("\t")[0].strip()
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


class GraphFiles:
    """Line-range lookups over a code graph's function/class nodes."""

    def __init__(self, graph):
        self.graph = graph
        self.defs: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        self.module_of_file: dict[str, str] = {}
        self.by_name: dict[str, list[str]] = defaultdict(list)
        self.by_qualname: dict[str, str] = {}
        for node, d in graph.nodes(data=True):
            file = d.get("file")
            qual = d.get("qualname") or node
            self.by_qualname[qual] = node
            self.by_qualname[node] = node
            self.by_name[d.get("name") or qual.rpartition(".")[2]].append(node)
            if not file:
                continue
            if d.get("kind") == "module":
                self.module_of_file[file] = node
            elif d.get("lineno") is not None:
                self.defs[file].append((d["lineno"], d.get("end_lineno") or d["lineno"], node))

    def match_file(self, path: str) -> str | None:
        path = path.replace("\\", "/").lstrip("./")
        if path in self.module_of_file or path in self.defs:
            return path
        candidates = [f for f in set(self.module_of_file) | set(self.defs)
                      if f.endswith("/" + path) or path.endswith("/" + f)]
        return max(candidates, key=len) if candidates else None

    def innermost(self, file: str, line: int) -> str | None:
        best = None
        for start, end, node in self.defs.get(file, []):
            if start <= line <= end and (best is None or end - start < best[1] - best[0]):
                best = (start, end, node)
        return best[2] if best else None

    def overlapping(self, file: str, start: int, end: int) -> set[str]:
        return {node for s, e, node in self.defs.get(file, []) if s <= end and e >= start}

    def resolve_symbol(self, text: str) -> str | None:
        text = text.strip().rstrip("()")
        if text in self.by_qualname:
            return self.by_qualname[text]
        ids = self.by_name.get(text, [])
        if len(ids) == 1:
            return ids[0]
        suffix = [n for n in self.by_qualname.values() if n.endswith("." + text)]
        return suffix[0] if len(set(suffix)) == 1 else None


def patch_alignment(patch: str, graph, exclude_tests: bool = True) -> tuple[int, int]:
    """(matching, checked) removed lines whose text equals the graph's source at that line.

    Gold locations are only right if the graph was built at the task's base
    commit. The module nodes store each file's full source, so compare every
    removed line with the graph's text at the same line number. A share well
    below 1.0 means the graph is from another commit and line-based gold
    locations for that task should not be trusted.
    """
    gf = graph if isinstance(graph, GraphFiles) else GraphFiles(graph)
    matched = checked = 0
    for path, lines in removed_lines(patch).items():
        if not path.endswith(".py") or (exclude_tests and _is_test_path(path)):
            continue
        file = gf.match_file(path)
        module = gf.module_of_file.get(file) if file else None
        if module is None:
            continue
        source = (gf.graph.nodes[module].get("source") or "").splitlines()
        if not source:
            continue
        for lineno, text in lines.items():
            checked += 1
            matched += 0 < lineno <= len(source) and source[lineno - 1].rstrip() == text.rstrip()
    return matched, checked


def changed_locations(patch: str, graph, exclude_tests: bool = True, use_hunk_headers: bool = False) -> Locations:
    gf = graph if isinstance(graph, GraphFiles) else GraphFiles(graph)
    gold = Locations()
    headers = hunk_header_names(patch) if use_hunk_headers else {}
    for path, lines in changed_lines(patch).items():
        if not path.endswith(".py") or (exclude_tests and _is_test_path(path)):
            continue
        file = gf.match_file(path)
        if file is None:  # new file, or not in the graph: nothing the agent could have opened
            continue
        gold.files.add(file)
        for line in lines:
            node = gf.innermost(file, line)
            if node:
                gold.functions.add(node)
        for name in headers.get(path, ()):
            ids = [n for n in gf.by_name.get(name, []) if gf.graph.nodes[n].get("file") == file]
            if len(ids) == 1:
                gold.functions.add(ids[0])
    return gold


def _is_test_path(path: str) -> bool:
    parts = path.split("/")
    return any(p in ("tests", "test", "testing") for p in parts[:-1]) or parts[-1].startswith("test_") \
        or parts[-1] == "conftest.py"


# ----------------------------------------------------------------- agent side
def load_events(path: str | Path) -> list[dict]:
    """Read a JSONL trajectory: one {"tool": name, "args": {...}, ...} per line."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def is_budgeted(event: dict) -> bool:
    return event.get("tool") not in FREE_TOOLS


def truncate(events: list[dict], k: int) -> list[dict]:
    """Events up to and including the k-th budgeted tool call."""
    out, used = [], 0
    for ev in events:
        if is_budgeted(ev):
            if used >= k:
                break
            used += 1
        out.append(ev)
    return out


SHELL_EXPLORE = re.compile(r"^\s*(grep|rg|egrep|fgrep|find|ls|tree|cat|head|tail|sed|awk|wc|git\s+(log|show|grep|ls-files|blame|status))\b")
SHELL_READ = re.compile(r"^\s*(?:cat|head|tail)\s+(?:-n\s*\d+\s+|-\d+\s+)?([\w./-]+\.py)\s*$"
                        r"|^\s*sed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?\s+([\w./-]+\.py)\s*$")


def _command(args: dict) -> str:
    cmd = args.get("command") or args.get("cmd") or ""
    return cmd if isinstance(cmd, str) else ""


def call_category(tool: str, args: dict | None = None) -> str:
    if tool == "run_command":   # one tool for search, reading and tests: look at the command
        return "explore" if SHELL_EXPLORE.match(_command(args or {})) else "test"
    if tool in READ_TOOLS or tool in SEARCH_TOOLS:
        return "explore"
    if tool in GRAPH_TOOLS:
        return "graph"
    if tool in EDIT_TOOLS:
        return "edit"
    if tool in TEST_TOOLS:
        return "test"
    if tool in FREE_TOOLS:
        return "free"
    return "other"


def visited_from_events(events: list[dict], graph) -> Locations:
    gf = graph if isinstance(graph, GraphFiles) else GraphFiles(graph)
    seen = Locations()
    for ev in events:
        tool, args = ev.get("tool"), ev.get("args") or {}
        if tool == "run_command":   # shell reads: cat/head/tail <file>, sed -n 'a,bp' <file>
            m = SHELL_READ.match(_command(args))
            if m:
                path = m.group(1) or m.group(4)
                file = gf.match_file(path)
                if file:
                    seen.files.add(file)
                    if m.group(1):
                        seen.functions |= {n for _, _, n in gf.defs.get(file, [])}
                    else:
                        seen.functions |= gf.overlapping(file, int(m.group(2)), int(m.group(3)))
            continue
        if tool in READ_TOOLS:
            path = _first(args, PATH_ARGS)
            file = gf.match_file(path) if isinstance(path, str) else None
            if not file:
                continue
            seen.files.add(file)
            start = _first(args, START_ARGS)
            end = _first(args, END_ARGS)
            limit = _first(args, LIMIT_ARGS)
            if start is None and end is None:
                seen.functions |= {n for _, _, n in gf.defs.get(file, [])}  # whole file read
            else:
                start = int(start or 1)
                end = int(end) if end is not None else (start + int(limit) - 1 if limit else 10**9)
                seen.functions |= gf.overlapping(file, start, end)
        elif tool in GRAPH_TOOLS:
            for value in _symbol_values(args):
                node = gf.resolve_symbol(value)
                if node:
                    d = gf.graph.nodes[node]
                    if d.get("kind") != "module":
                        seen.functions.add(node)
                    if d.get("file"):
                        seen.files.add(d["file"])
    return seen


def _first(args: dict, keys: tuple[str, ...]):
    for key in keys:
        if args.get(key) is not None:
            return args[key]
    return None


def _symbol_values(args: dict) -> list[str]:
    values = []
    for key in SYMBOL_ARGS:
        v = args.get(key)
        if isinstance(v, str):
            values.append(v)
        elif isinstance(v, list):
            values.extend(x for x in v if isinstance(x, str))
    return values


def recall(gold: set[str], visited: set[str]) -> float | None:
    """Share of gold locations the agent looked at; None when there is no gold."""
    if not gold:
        return None
    return len(gold & visited) / len(gold)


def localization_scores(patch: str, events: list[dict], graph, budgets=(25, 50, 100)) -> dict:
    """File and function recall at each budget (by truncating one trajectory)."""
    gf = GraphFiles(graph)
    gold = changed_locations(patch, gf)
    matched, checked = patch_alignment(patch, gf)
    out = {"gold_files": len(gold.files), "gold_functions": len(gold.functions),
           "alignment": matched / checked if checked else None}
    for k in budgets:
        seen = visited_from_events(truncate(events, k), gf)
        out[f"file_recall@{k}"] = recall(gold.files, seen.files)
        out[f"function_recall@{k}"] = recall(gold.functions, seen.functions)
    return out
