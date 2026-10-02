"""Issue-to-symbol bridge: turn issue text into graph entry points.

The competition's embedding search accepts only a symbol name, so an agent
cannot go from issue text to code semantically. This step extracts candidate
identifiers from the issue (and any traceback), keeps those that exist in the
repository graph, ranks them and returns the top k as starting points.

Scoring (higher is better):
    traceback frame inside the repo     100 + depth (deepest frame first)
    name in code formatting              60
    dotted path (module.Class.method)    40
    .py file path                        35
    CamelCase / snake_case identifier    20 - 5 per extra match (skipped if > 3 matches)
    kind bonus: function +3, class +2, module +1; +2 per repeat mention (max +6)

Standard library only, so it can be packaged as a sandboxed skill script.

Usage:
    python -m dspo.bridge --graph repo.pkl   --issue issue.txt -k 5
    python -m dspo.bridge --symbols repo.symbols.json --issue - --json < issue.txt
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

TRACEBACK_FRAME = re.compile(r'File "(?P<path>[^"]+)", line (?P<line>\d+), in (?P<func>[\w<>.]+)')
CODE_SPAN = re.compile(r"`([^`\n]+)`")
CODE_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
DOTTED = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")
PY_PATH = re.compile(r"(?<![\w/.])((?:[\w.-]+/)*[\w-]+\.py)\b")
CAMEL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b")
SNAKE = re.compile(r"\b_*[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
IDENT = re.compile(r"[A-Za-z_]\w*")

# Words that look like identifiers but say nothing about where the bug is.
STOPWORDS = {
    "self", "cls", "none", "true", "false", "return", "import", "from", "def",
    "class", "print", "str", "int", "dict", "list", "tuple", "bool", "bytes",
    "float", "object", "type", "len", "isinstance", "raise", "except", "try",
    "lambda", "yield", "async", "await", "python", "pip", "main", "init",
    "test", "tests", "error", "exception", "traceback", "value", "args", "kwargs",
}

KIND_BONUS = {"function": 3, "class": 2, "module": 1}


@dataclass
class Symbol:
    id: str
    kind: str
    name: str
    qualname: str
    module: str
    file: str | None
    lineno: int | None
    end_lineno: int | None


@dataclass
class Candidate:
    symbol: Symbol
    score: float
    reasons: list[str] = field(default_factory=list)


class SymbolIndex:
    """Lookups over the graph's nodes by name, dotted suffix and file path."""

    def __init__(self, symbols: list[Symbol]):
        self.symbols = {s.id: s for s in symbols}
        self.by_name: dict[str, list[str]] = defaultdict(list)
        self.by_suffix: dict[str, list[str]] = defaultdict(list)
        self.by_file: dict[str, str] = {}  # file path -> module node id
        self.defs_in_file: dict[str, list[Symbol]] = defaultdict(list)
        for s in symbols:
            self.by_name[s.name].append(s.id)
            parts = s.qualname.split(".")
            for i in range(len(parts) - 1):  # every dotted suffix of length >= 2
                self.by_suffix[".".join(parts[i:])].append(s.id)
            if s.file:
                if s.kind == "module":
                    self.by_file[s.file] = s.id
                elif s.lineno is not None:
                    self.defs_in_file[s.file].append(s)

    @classmethod
    def from_graph(cls, graph) -> "SymbolIndex":
        return cls([_symbol(n, d) for n, d in graph.nodes(data=True)])

    @classmethod
    def from_records(cls, records: list[dict]) -> "SymbolIndex":
        return cls([_symbol(r["id"], r) for r in records])

    def match_file(self, path: str) -> str | None:
        """Longest-suffix match of a (possibly absolute) path against repo files."""
        path = path.replace("\\", "/")
        best, best_len = None, 0
        for file in self.by_file:
            if path == file or path.endswith("/" + file):
                if len(file) > best_len:
                    best, best_len = file, len(file)
        if best is None:  # repo root prefixes like src/ may be absent from the path
            for file in self.by_file:
                tail = file.split("/", 1)[-1]
                if "/" in file and (path == tail or path.endswith("/" + tail)) and len(tail) > best_len:
                    best, best_len = file, len(tail)
        return best

    def innermost(self, file: str, line: int) -> Symbol | None:
        best = None
        for s in self.defs_in_file.get(file, []):
            if s.lineno <= line <= (s.end_lineno or s.lineno):
                if best is None or (s.end_lineno - s.lineno) < (best.end_lineno - best.lineno):
                    best = s
        return best

    def lookup(self, text: str) -> list[str]:
        text = text.strip().rstrip("()").strip(".")
        if text in self.symbols:
            return [text]
        if "." in text:
            return list(self.by_suffix.get(text, []))
        return list(self.by_name.get(text, []))


def _symbol(node_id: str, d: dict) -> Symbol:
    qualname = d.get("qualname") or node_id
    return Symbol(
        id=node_id, kind=d.get("kind", "function"), name=d.get("name") or qualname.rpartition(".")[2],
        qualname=qualname, module=d.get("module", ""), file=d.get("file"),
        lineno=d.get("lineno"), end_lineno=d.get("end_lineno"),
    )


def strip_traceback_frames(issue: str) -> str:
    """Drop each 'File "...", line N, in f' line and the indented code lines under it."""
    kept, in_frame = [], False
    for line in issue.splitlines():
        if TRACEBACK_FRAME.search(line):
            in_frame = True
            continue
        if in_frame and line[:1].isspace() and line.strip():
            continue
        in_frame = False
        kept.append(line)
    return "\n".join(kept)


def extract_candidates(issue: str, index: SymbolIndex) -> list[Candidate]:
    found: dict[str, Candidate] = {}
    mentions: dict[str, int] = defaultdict(int)

    def add(node_id: str, score: float, reason: str) -> None:
        mentions[node_id] += 1
        cand = found.get(node_id)
        if cand is None:
            found[node_id] = Candidate(index.symbols[node_id], score, [reason])
        else:
            if reason not in cand.reasons:
                cand.reasons.append(reason)
            cand.score = max(cand.score, score)

    # 1. Traceback frames, located by file + line (exact), else by function name.
    frames = list(TRACEBACK_FRAME.finditer(issue))
    for depth, m in enumerate(frames):
        file = index.match_file(m["path"])
        if not file:
            continue  # frame from the standard library or another package
        sym = index.innermost(file, int(m["line"]))
        if sym is None or sym.name != m["func"].rpartition(".")[2]:
            ids = [i for i in index.by_name.get(m["func"], []) if index.symbols[i].file == file]
            sym = index.symbols[ids[0]] if len(ids) == 1 else None
        add(sym.id if sym else index.by_file[file], 100 + depth, f"traceback frame {depth}")

    # Remove traceback frames and their echoed source lines so they are not counted again.
    text = strip_traceback_frames(issue)

    # 2. Names in code formatting (inline spans and fenced blocks).
    code_bits = CODE_SPAN.findall(text) + CODE_FENCE.findall(text)
    for bit in code_bits:
        for token in set(DOTTED.findall(bit)) | set(IDENT.findall(bit)):
            if token.lower() in STOPWORDS:
                continue
            ids = index.lookup(token)
            if 0 < len(ids) <= 3:
                for i in ids:
                    add(i, 60 - 5 * (len(ids) - 1), f"code `{token}`")

    # 3. Dotted paths anywhere in the text.
    for token in set(DOTTED.findall(text)):
        if token.endswith(".py"):
            continue
        ids = index.lookup(token)
        if 0 < len(ids) <= 3:
            for i in ids:
                add(i, 40 - 5 * (len(ids) - 1), f"dotted {token}")

    # 4. File paths.
    for path in set(PY_PATH.findall(text)):
        file = index.match_file(path)
        if file:
            add(index.by_file[file], 35, f"file {path}")

    # 5. CamelCase and snake_case identifiers in plain text.
    for token in set(CAMEL.findall(text)) | set(SNAKE.findall(text)):
        if token.lower() in STOPWORDS:
            continue
        ids = index.lookup(token)
        if 0 < len(ids) <= 3:
            for i in ids:
                add(i, 20 - 5 * (len(ids) - 1), f"identifier {token}")

    for node_id, cand in found.items():
        cand.score += KIND_BONUS.get(cand.symbol.kind, 0) + min(2 * (mentions[node_id] - 1), 6)
    return sorted(found.values(), key=lambda c: (-c.score, c.symbol.id))


def bridge(issue: str, index: SymbolIndex, k: int = 5) -> list[Candidate]:
    return extract_candidates(issue, index)[:k]


def format_for_agent(cands: list[Candidate]) -> str:
    if not cands:
        return "No symbols from the issue were found in the code graph. Start with grep."
    lines = ["Likely starting points from the issue (use these with the graph tools):"]
    for i, c in enumerate(cands, 1):
        loc = f"{c.symbol.file}:{c.symbol.lineno}" if c.symbol.file else ""
        lines.append(f"{i}. {c.symbol.id} ({c.symbol.kind}) {loc} - {c.reasons[0]}")
    return "\n".join(lines)


def load_index(graph_path: str | None = None, symbols_path: str | None = None) -> SymbolIndex:
    if symbols_path:
        with open(symbols_path) as fh:
            return SymbolIndex.from_records(json.load(fh))
    with open(graph_path, "rb") as fh:
        return SymbolIndex.from_graph(pickle.load(fh))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--graph", help="pickled NetworkX graph")
    source.add_argument("--symbols", help="JSON symbol index from graph_builder --symbols")
    parser.add_argument("--issue", required=True, help="issue text file, or - for stdin")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="print JSON instead of the agent message")
    args = parser.parse_args(argv)

    issue = sys.stdin.read() if args.issue == "-" else open(args.issue, encoding="utf-8").read()
    cands = bridge(issue, load_index(args.graph, args.symbols), args.k)
    if args.json:
        print(json.dumps([{"id": c.symbol.id, "kind": c.symbol.kind, "file": c.symbol.file,
                           "lineno": c.symbol.lineno, "score": c.score, "reasons": c.reasons} for c in cands], indent=2))
    else:
        print(format_for_agent(cands))
    return 0


if __name__ == "__main__":
    sys.exit(main())
