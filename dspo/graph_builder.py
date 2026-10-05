"""Turn any Python repository into a code graph in the competition's format.

Nodes are modules, classes and functions. Each node carries: kind, name,
qualname, module, file (relative to the repo root), lineno, end_lineno and
source. Edges are typed DEFINED_IN (child -> parent), IMPORTS (module ->
module) and CALLS (function -> function or class).

Calls are resolved exactly where the code makes it possible (local names,
imports, relative imports, package re-exports, self./cls. methods including
inherited ones, module.func and Class.method). Anything else falls back to a
unique-name match, marked resolution="heuristic".

Usage:
    python -m dspo.graph_builder path/to/repo --out repo.pkl [--symbols repo.symbols.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from dspo import CALLS, DEFINED_IN, IMPORTS
from dspo.symbols import SKIP_DIRS, TEST_DIRS, definitions, discover_modules, parse_module  # noqa: F401

# Method names shared with builtins (dict.update, list.append, file.read, ...).
# A unique-name match on these is far more likely to be wrong than right.
GENERIC_NAMES = {
    "get", "set", "update", "pop", "copy", "clear", "keys", "values", "items",
    "append", "extend", "insert", "remove", "index", "count", "sort",
    "read", "write", "close", "open", "flush", "seek", "readline", "readlines",
    "join", "split", "strip", "lower", "upper", "replace", "format", "encode",
    "decode", "startswith", "endswith", "add", "discard", "send", "run", "start",
}


@dataclass
class ModuleInfo:
    name: str
    path: Path
    rel: str
    is_package: bool
    tree: ast.Module
    lines: list[str]
    # local alias -> dotted target (a module or a symbol inside a module)
    aliases: dict[str, str] = field(default_factory=dict)


def _resolve_relative(module: ModuleInfo, level: int, target: str | None) -> str:
    base = module.name if module.is_package else module.name.rpartition(".")[0]
    for _ in range(level - 1):
        base = base.rpartition(".")[0]
    if target:
        return f"{base}.{target}" if base else target
    return base


class GraphBuilder:
    def __init__(self, root: Path, include_tests: bool = False):
        self.root = root.resolve()
        self.include_tests = include_tests
        self.graph = nx.MultiDiGraph()
        self.modules: dict[str, ModuleInfo] = {}
        self.by_name: dict[str, list[str]] = defaultdict(list)  # short name -> node ids
        self.class_bases: dict[str, list[ast.expr]] = {}
        self.class_module: dict[str, str] = {}
        self.parse_errors: list[str] = []

    # ---------------------------------------------------------------- parsing
    def build(self) -> nx.MultiDiGraph:
        for name, path in discover_modules(self.root, self.include_tests).items():
            parsed = parse_module(path)
            if parsed is None:
                self.parse_errors.append(f"{path}: not valid Python for this interpreter")
                continue
            tree, lines = parsed
            rel = path.relative_to(self.root).as_posix()
            self.modules[name] = ModuleInfo(
                name=name, path=path, rel=rel, is_package=path.name == "__init__.py",
                tree=tree, lines=lines,
            )
        for mod in self.modules.values():
            self._add_definitions(mod)
        for mod in self.modules.values():
            self._collect_imports(mod)
        for mod in self.modules.values():
            self._add_calls(mod)
        self.graph.graph.update(root=str(self.root), builder="dspo.graph_builder")
        return self.graph

    def _source(self, mod: ModuleInfo, node: ast.AST) -> str:
        start = node.lineno
        if getattr(node, "decorator_list", None):
            start = min([start] + [d.lineno for d in node.decorator_list])
        return "\n".join(mod.lines[start - 1 : node.end_lineno])

    def _add_node(self, node_id: str, **attrs) -> None:
        self.graph.add_node(node_id, **attrs)
        self.by_name[attrs["name"]].append(node_id)

    def _add_definitions(self, mod: ModuleInfo) -> None:
        self._add_node(
            mod.name, kind="module", name=mod.name.rpartition(".")[2] or mod.name,
            qualname=mod.name, module=mod.name, file=mod.rel, lineno=1,
            end_lineno=len(mod.lines), source="\n".join(mod.lines),
        )

        for node_id, kind, name, stmt, parent in definitions(mod.tree, mod.name, self.graph):
            self._add_node(
                node_id, kind=kind, name=name, qualname=node_id,
                module=mod.name, file=mod.rel, lineno=stmt.lineno,
                end_lineno=stmt.end_lineno, source=self._source(mod, stmt),
            )
            self.graph.add_edge(node_id, parent, key=DEFINED_IN, type=DEFINED_IN)
            if kind == "class":
                self.class_bases[node_id] = stmt.bases
                self.class_module[node_id] = mod.name

    def _collect_imports(self, mod: ModuleInfo) -> None:
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        mod.aliases[alias.asname] = alias.name
                    else:
                        top = alias.name.partition(".")[0]
                        mod.aliases.setdefault(top, top)
                    self._import_edge(mod.name, alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = (
                    _resolve_relative(mod, node.level, node.module)
                    if node.level else (node.module or "")
                )
                self._import_edge(mod.name, base)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    target = f"{base}.{alias.name}" if base else alias.name
                    mod.aliases[alias.asname or alias.name] = target
                    if target in self.modules:
                        self._import_edge(mod.name, target)

    def _import_edge(self, src: str, target: str) -> None:
        # Import of a.b.c inside the repo: link to the deepest module that exists.
        while target and target not in self.modules:
            target = target.rpartition(".")[0]
        if target and target != src and not self.graph.has_edge(src, target, key=IMPORTS):
            self.graph.add_edge(src, target, key=IMPORTS, type=IMPORTS)

    # ------------------------------------------------------------- resolution
    def resolve(self, dotted: str, depth: int = 0) -> str | None:
        """Resolve a dotted name to a node id, following package re-exports."""
        if dotted in self.graph:
            return dotted
        if depth > 8:
            return None
        parts = dotted.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix, rest = ".".join(parts[:i]), parts[i:]
            if prefix in self.modules:
                alias_target = self.modules[prefix].aliases.get(rest[0])
                if alias_target:
                    return self.resolve(".".join([alias_target, *rest[1:]]), depth + 1)
                return None
            if prefix in self.graph and self.graph.nodes[prefix]["kind"] == "class":
                return self._resolve_member(prefix, rest[0], depth)
        return None

    def _resolve_member(self, cls: str, attr: str, depth: int = 0) -> str | None:
        """Find attr on cls or its in-repo base classes (depth-first MRO approximation)."""
        seen: set[str] = set()
        stack = [cls]
        while stack:
            current = stack.pop(0)
            if current in seen:
                continue
            seen.add(current)
            candidate = f"{current}.{attr}"
            if candidate in self.graph:
                return candidate
            for base in self.class_bases.get(current, []):
                base_id = self._resolve_expr(base, self.class_module[current], None, depth + 1)
                if base_id and self.graph.nodes[base_id]["kind"] == "class":
                    stack.append(base_id)
        return None

    def _resolve_expr(self, expr: ast.expr, module: str, cls: str | None, depth: int = 0) -> str | None:
        chain = _attr_chain(expr)
        if not chain:
            return None
        head, rest = chain[0], chain[1:]
        if head in ("self", "cls") and cls and rest:
            target = self._resolve_member(cls, rest[0], depth)
            return target if len(rest) == 1 else None
        local = f"{module}.{head}"
        mod = self.modules[module]
        if local in self.graph:
            base = local
        elif head in mod.aliases:
            base = mod.aliases[head]
        else:
            return None
        return self.resolve(".".join([base, *rest]), depth)

    # ------------------------------------------------------------------ calls
    def _add_calls(self, mod: ModuleInfo) -> None:
        def visit(body: list[ast.stmt], parent: str, cls: str | None) -> None:
            for stmt in body:
                if isinstance(stmt, ast.ClassDef):
                    node_id = f"{parent}.{stmt.name}"
                    if node_id in self.graph:
                        visit(stmt.body, node_id, node_id)
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    node_id = f"{parent}.{stmt.name}"
                    if node_id in self.graph:
                        self._calls_in_function(mod, stmt, node_id, cls)
                        visit(stmt.body, node_id, cls)
                elif isinstance(stmt, (ast.If, ast.Try, ast.With, ast.AsyncWith)):
                    for sub in ("body", "orelse", "finalbody", "handlers"):
                        for item in getattr(stmt, sub, []) or []:
                            visit(item.body if isinstance(item, ast.ExceptHandler) else [item], parent, cls)

        visit(mod.tree.body, mod.name, None)

    def _calls_in_function(self, mod: ModuleInfo, func: ast.AST, src: str, cls: str | None) -> None:
        for call in _calls_excluding_nested(func):
            target = self._resolve_expr(call.func, mod.name, cls)
            resolution = "exact"
            if target is None:
                target = self._heuristic(call.func)
                resolution = "heuristic"
            if target is None or target == src:
                continue
            if self.graph.nodes[target]["kind"] not in ("function", "class"):
                continue
            if not self.graph.has_edge(src, target, key=CALLS):
                self.graph.add_edge(src, target, key=CALLS, type=CALLS, resolution=resolution)

    def _heuristic(self, expr: ast.expr) -> str | None:
        chain = _attr_chain(expr)
        name = chain[-1] if chain else (expr.attr if isinstance(expr, ast.Attribute) else None)
        if not name or name in GENERIC_NAMES or (name.startswith("__") and name.endswith("__")):
            return None
        matches = [n for n in self.by_name.get(name, []) if self.graph.nodes[n]["kind"] != "module"]
        return matches[0] if len(matches) == 1 else None


def _attr_chain(expr: ast.expr) -> list[str] | None:
    """`a.b.c` -> ["a", "b", "c"]; None for anything that is not a plain name chain."""
    parts = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        parts.append(expr.id)
        return parts[::-1]
    return None


def _calls_excluding_nested(func: ast.AST):
    """Yield Call nodes in a function body, skipping nested function and class bodies."""
    stack = list(func.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Decorators and defaults are evaluated in the enclosing scope.
            stack.extend(getattr(node, "decorator_list", []))
            if not isinstance(node, ast.ClassDef):
                stack.extend(node.args.defaults + [d for d in node.args.kw_defaults if d])
            continue
        if isinstance(node, ast.Call):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def build_graph(root: str | Path, include_tests: bool = False) -> nx.MultiDiGraph:
    return GraphBuilder(Path(root), include_tests).build()


def symbol_index(graph: nx.MultiDiGraph) -> list[dict]:
    """A small, dependency-free index of the graph's nodes (for the bridge skill)."""
    keys = ("kind", "name", "qualname", "module", "file", "lineno", "end_lineno")
    return [{"id": n, **{k: d.get(k) for k in keys}} for n, d in graph.nodes(data=True)]


def summary(graph: nx.MultiDiGraph) -> dict:
    kinds = defaultdict(int)
    for _, d in graph.nodes(data=True):
        kinds[d["kind"]] += 1
    edges = defaultdict(int)
    heuristic = 0
    for _, _, d in graph.edges(data=True):
        edges[d["type"]] += 1
        heuristic += d.get("resolution") == "heuristic"
    return {"nodes": dict(kinds), "edges": dict(edges), "heuristic_calls": heuristic}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="output pickle (NetworkX MultiDiGraph)")
    parser.add_argument("--symbols", type=Path, help="also write a JSON symbol index for the bridge")
    parser.add_argument("--include-tests", action="store_true")
    args = parser.parse_args(argv)

    builder = GraphBuilder(args.repo, args.include_tests)
    graph = builder.build()
    with open(args.out, "wb") as fh:
        pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
    if args.symbols:
        args.symbols.write_text(json.dumps(symbol_index(graph)))
    stats = summary(graph)
    print(json.dumps(stats))
    for err in builder.parse_errors:
        print(f"warning: could not parse {err}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
