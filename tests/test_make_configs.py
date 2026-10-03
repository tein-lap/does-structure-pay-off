import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


class _L(yaml.SafeLoader):
    pass


_L.add_constructor("!include", lambda loader, node: loader.construct_scalar(node))


def test_builds_fair_submissions(tmp_path):
    out = tmp_path / "build"
    r = subprocess.run([sys.executable, str(ROOT / "agent/make_configs.py"), "--out", str(out),
                        "--skills-dir", str(tmp_path / "no_skill")], check=True, capture_output=True, text=True)
    assert "12 submissions" in r.stdout
    a = yaml.load((out / "A" / "b25" / "agent.yaml").read_text(), Loader=_L)
    assert a["tools"] == ["run_command", "read_file", "edit_file", "write_file", "submit_patch", "get_status"]
    assert "adapter" not in a and "skills" not in a and a["instruction"] == "prompts/system.md"
    d = yaml.load((out / "D" / "b100" / "agent.yaml").read_text(), Loader=_L)
    assert {"get_code_neighbors", "get_code_subgraph", "search_similar_code"} <= set(d["tools"])
    c = yaml.load((out / "C" / "b50" / "agent.yaml").read_text(), Loader=_L)
    assert "search_similar_code" in c["tools"] and "get_code_neighbors" not in c["tools"]
    prompt_a = (out / "A" / "b25" / "prompts" / "system.md").read_text()
    assert "You have 25 tool calls" in prompt_a and "get_code_neighbors" not in prompt_a
    assert "{{" not in prompt_a
    ev = yaml.safe_load((out / "A" / "b25" / "eval_config.yaml").read_text())
    assert ev["evaluation"]["max_tool_calls"] == 25
    with zipfile.ZipFile(out / "zips" / "D_b100.zip") as zf:
        names = set(zf.namelist())
    assert {"agent.yaml", "eval_config.yaml", "prompts/system.md", "configs/sampling.yaml"} <= names
    assert not (out / "D+bridge").exists()          # no SKILL.md yet -> bridge arms skipped
    # all arms share the same sampling settings
    assert len({(out / arm / "b50" / "configs" / "sampling.yaml").read_text() for arm in "ABCD"}) == 1
