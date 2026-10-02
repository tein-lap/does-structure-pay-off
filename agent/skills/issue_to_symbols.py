"""Skill script: the issue-to-symbol bridge, packaged for the competition sandbox.

Standard library only. Ship it together with dspo/bridge.py (copied next to
this file as bridge.py) and a JSON symbol index made by
`python -m dspo.graph_builder <repo> --out g.pkl --symbols symbols.json`
or converted from the competition graph.

Before using it in the competition, check in HARNESS_README:
  - how a skill receives its input (argument, stdin or file) and returns output;
  - whether the repository graph is readable from inside the sandbox;
  - whether calling a skill counts as one of the 100 tool calls.
Adjust main() below to match.

Usage (local test):
    python agent/skills/issue_to_symbols.py --symbols symbols.json < issue.txt
"""

import argparse
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))                      # bridge.py copied next to this file
sys.path.insert(0, str(here.parent.parent))        # or the dspo package in this repo

try:
    from bridge import bridge, format_for_agent, load_index
except ImportError:
    from dspo.bridge import bridge, format_for_agent, load_index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=str(here / "symbols.json"))
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    issue = sys.stdin.read()
    print(format_for_agent(bridge(issue, load_index(symbols_path=args.symbols), args.k)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
