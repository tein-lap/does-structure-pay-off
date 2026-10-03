import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

TEMPLATE = """name: swe_agent
model: gemma-4-31b-it-qat-w4a16-ct
instruction: |
  Fix the issue. You have {{BUDGET}} tool calls. {{BRIDGE_NOTE}}
tools:
  {{TOOLS}}
"""


def test_configs_have_tools_budget_and_skill(tmp_path):
    (tmp_path / "template.yaml").write_text(TEMPLATE)
    out = tmp_path / "build"
    subprocess.run([sys.executable, str(ROOT / "agent/make_configs.py"), "--template", str(tmp_path / "template.yaml"),
                    "--out", str(out)], check=True, capture_output=True, text=True)
    a = yaml.safe_load((out / "A" / "b25" / "agent.yaml").read_text())
    assert a["tools"] == ["run_command", "read_file", "edit_file", "write_file", "submit_patch", "get_status"]
    assert "skills" not in a
    d = yaml.safe_load((out / "D" / "b100" / "agent.yaml").read_text())
    assert {"get_code_neighbors", "get_code_subgraph", "search_similar_code"} <= set(d["tools"])
    c = yaml.safe_load((out / "C" / "b50" / "agent.yaml").read_text())
    assert "search_similar_code" in c["tools"] and "get_code_neighbors" not in c["tools"]
    br = yaml.safe_load((out / "D+bridge" / "b100" / "agent.yaml").read_text())
    assert br["skills"] == ["skills/issue_to_symbols"] and "issue_to_symbols" not in br["tools"]
    ev = yaml.safe_load((out / "A" / "b25" / "eval_config.yaml").read_text())
    assert ev["evaluation"]["max_tool_calls"] == 25
