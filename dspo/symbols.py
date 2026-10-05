"""Index the modules, classes and functions of a Python repository (standard library only).

The graph builder uses the same rules, so symbol ids match exactly. This module needs
no third-party package, so it can run inside the competition sandbox, where the bridge
skill indexes /workspace directly.

    from dspo.symbols import index_symbols
    records = index_symbols("/workspace")   # [{"id", "kind", "name", "qualname", "module", "file", "lineno", "end_lineno"}]
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

SKIP_DIRS = {
    ".git", ".hg", ".tox", ".nox", ".venv", "venv", "env", "__pycache__",
    "node_modules", "build", "dist", "site-packages", ".eggs",
    "docs", "doc", "examples", "example", "scripts", "benchmarks",
}
TEST_DIRS = {"tests", "test", "testing"}


def _is_test_file(rel: Path) -> bool:
    name = rel.name
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def discover_modules(root: Path, include_tests: bool = False) -> dict[str, Path]:
    """Map dotted module name -> file path for every .py file worth parsing."""
    root = Path(root).resolve()
    skip = SKIP_DIRS if include_tests else SKIP_DIRS | TEST_DIRS
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(part in skip or part.startswith(".") for part in rel.parts[:-1]):
            continue
        if not include_tests and _is_test_file(rel):
            continue
        if rel.name == "setup.py" and len(rel.parts) == 1:
            continue
        parts = list(rel.with_suffix("").parts)
        if parts and parts[0] == "src":  # src/ layout
            parts = parts[1:]
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts or not all(p.isidentifier() for p in parts):
            continue
        found[".".join(parts)] = path
    return found


def parse_module(path: Path) -> tuple[ast.Module, list[str]] | None:
    """Parse one file; None if it is not valid Python for this interpreter."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        with warnings.catch_warnings():  # old code: invalid escapes in docstrings etc.
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(text, filename=str(path)), text.splitlines()
    except SyntaxError:
        return None


def definitions(tree: ast.Module, module: str, taken):
    """Yield (node_id, kind, name, stmt, parent_id) for every class and function.

    `taken` is any container of ids already used (a set, or the graph itself); a definition
    whose id is taken is skipped together with its body (first definition wins). The CALLER
    adds each yielded id to `taken`. Definitions under if/try/with blocks are included.
    """
    def visit(body, parent):
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                node_id = f"{parent}.{stmt.name}"
                if node_id in taken:
                    continue
                kind = "class" if isinstance(stmt, ast.ClassDef) else "function"
                yield node_id, kind, stmt.name, stmt, parent
                yield from visit(stmt.body, node_id)
            elif isinstance(stmt, (ast.If, ast.Try, ast.With, ast.AsyncWith)):
                for sub in ("body", "orelse", "finalbody", "handlers"):
                    for item in getattr(stmt, sub, []) or []:
                        yield from visit(item.body if isinstance(item, ast.ExceptHandler) else [item], parent)

    yield from visit(tree.body, module)


def index_symbols(root: str | Path, include_tests: bool = False) -> list[dict]:
    root = Path(root).resolve()
    parsed = {}
    for module, path in discover_modules(root, include_tests).items():   # parse first, like the graph builder
        result = parse_module(path)
        if result is not None:
            parsed[module] = (path, *result)
    records, taken = [], set()
    for module, (path, tree, lines) in parsed.items():
        rel = path.relative_to(root).as_posix()
        if module in taken:   # the graph builder overwrites a same-named definition with the module
            records = [r for r in records if r["id"] != module]
        taken.add(module)
        records.append({"id": module, "kind": "module", "name": module.rpartition(".")[2] or module,
                        "qualname": module, "module": module, "file": rel,
                        "lineno": 1, "end_lineno": len(lines)})
        for node_id, kind, name, stmt, _ in definitions(tree, module, taken):
            taken.add(node_id)
            records.append({"id": node_id, "kind": kind, "name": name, "qualname": node_id,
                            "module": module, "file": rel, "lineno": stmt.lineno,
                            "end_lineno": stmt.end_lineno})
    return records
