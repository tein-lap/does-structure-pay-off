"""Skill script: suggest the repository symbols an issue most likely refers to.

Runs inside the competition sandbox with the standard library only. It indexes the
repository in /workspace (dspo/symbols.py) and ranks the symbols named in the issue
(dspo/bridge.py). make_configs.py copies bridge.py and symbols.py next to this file
when it builds a submission; in this repository they are imported from dspo/.

Usage:
    python issue_to_symbols.py "Traceback ... File \"/workspace/pkg/mod.py\", line 12, in f ..."
    python issue_to_symbols.py --issue-file issue.txt
    echo "issue text" | python issue_to_symbols.py
"""

import argparse
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))                          # bridge.py / symbols.py shipped next to this file
sys.path.insert(0, str(here.parents[3]))               # or the dspo package in this repository

try:
    from bridge import SymbolIndex, bridge, format_for_agent
    from symbols import index_symbols
except ImportError:
    from dspo.bridge import SymbolIndex, bridge, format_for_agent
    from dspo.symbols import index_symbols


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Suggest starting symbols for an issue.")
    parser.add_argument("text", nargs="*", help="issue text (or pass --issue-file, or use stdin)")
    parser.add_argument("--issue-file", help="file containing the issue text")
    parser.add_argument("--root", default="/workspace", help="repository root (default /workspace)")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args(argv)

    if args.issue_file:
        issue = Path(args.issue_file).read_text(encoding="utf-8", errors="replace")
    elif args.text:
        issue = " ".join(args.text)
    elif not sys.stdin.isatty():
        issue = sys.stdin.read()
    else:
        issue = ""
    if not issue.strip():
        print("Pass the issue text: issue_to_symbols.py \"<issue text>\"")
        return 1
    root = Path(args.root)
    if not root.is_dir():
        print(f"Repository folder {root} not found; pass --root")
        return 1
    index = SymbolIndex.from_records(index_symbols(root))
    print(format_for_agent(bridge(issue, index, args.k)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
