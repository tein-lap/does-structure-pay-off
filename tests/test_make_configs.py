import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = str(ROOT / "agent/make_configs.py")


def build(tmp_path, *extra):
    out = tmp_path / "build"
    r = subprocess.run([sys.executable, SCRIPT, "--out", str(out), "--skills-dir", str(tmp_path / "no_skill"), *extra],
                       capture_output=True, text=True)
    return out, r


def test_builds_fair_submissions(tmp_path):
    out, r = build(tmp_path)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "13 submissions" in r.stdout                     # 12 arm/budget zips + 1 smoke zip
    # Plain YAML: no custom tags, so any standard loader can read it.
    a = yaml.safe_load((out / "A" / "b25" / "agent.yaml").read_text())
    assert a["tools"] == ["run_command", "read_file", "edit_file", "write_file", "submit_patch", "get_status"]
    assert "adapter" not in a and "skills" not in a
    # The real prompt is inside agent.yaml, not a path to another file.
    assert "You have 25 tool calls" in a["instruction"] and "submit_patch" in a["instruction"]
    assert "get_code_neighbors" not in a["instruction"]
    assert a["generate_content_config"]["max_output_tokens"] <= 4096
    assert a["generate_content_config"]["thinking_config"]["include_thoughts"] is False
    d = yaml.safe_load((out / "D" / "b100" / "agent.yaml").read_text())
    assert {"get_code_neighbors", "get_code_subgraph", "search_similar_code"} <= set(d["tools"])
    c = yaml.safe_load((out / "C" / "b50" / "agent.yaml").read_text())
    assert "search_similar_code" in c["tools"] and "get_code_neighbors" not in c["tools"]
    assert not (out / "D+bridge").exists()                 # no SKILL.md yet -> bridge arms skipped
    # All arms share the same generation settings.
    assert len({str(yaml.safe_load((out / arm / "b50" / "agent.yaml").read_text())["generate_content_config"])
                for arm in "ABCD"}) == 1


def test_runtime_fits_the_total_limit(tmp_path):
    out, r = build(tmp_path)
    assert r.returncode == 0, r.stderr
    ev = yaml.safe_load((out / "D" / "b100" / "eval_config.yaml").read_text())["evaluation"]
    assert ev["max_tool_calls"] == 100
    assert ev["max_time_minutes"] == 4                      # floor(12 h * 60 * 0.7 / 120 tasks)
    assert 120 * ev["max_time_minutes"] <= 12 * 60 * 0.7    # worst case fits with the margin
    assert ev["timeout_seconds"] == 60 and ev["max_turns"] == 150   # 60 s: the cap (a third of 4 min is 80 s)
    assert "timeout 60 python -m pytest" in yaml.safe_load((out / "D" / "b100" / "agent.yaml").read_text())["instruction"]
    prompt = yaml.safe_load((out / "D" / "b100" / "agent.yaml").read_text())["instruction"]
    assert "about 4 minutes" in prompt


def test_smoke_submission(tmp_path):
    out, r = build(tmp_path)
    with zipfile.ZipFile(out / "zips" / "SMOKE_A_b3.zip") as zf:
        names = set(zf.namelist())
        agent = yaml.safe_load(zf.read("agent.yaml"))
        ev = yaml.safe_load(zf.read("eval_config.yaml"))["evaluation"]
    assert names == {"agent.yaml", "eval_config.yaml"}       # nothing else to read, all at the zip root
    assert ev["max_tool_calls"] == 3 and ev["max_time_minutes"] == 1
    assert 120 * ev["max_time_minutes"] <= 2 * 60                # finishes even under a 2-hour limit
    assert agent["generate_content_config"]["thinking_config"]["thinking_budget"] == 0
    assert "about 1 minute for this task" in agent["instruction"]
    assert ev["timeout_seconds"] == 20 and "timeout 20 python -m pytest" in agent["instruction"]   # a third of 1 minute


def test_longer_limit_gives_more_time_per_task(tmp_path):
    out, r = build(tmp_path, "--total-hours", "12", "--parallel-tasks", "2")
    assert r.returncode == 0, r.stderr
    ev = yaml.safe_load((out / "A" / "b25" / "eval_config.yaml").read_text())["evaluation"]
    assert ev["max_time_minutes"] == 8                      # floor(12 * 60 * 0.7 * 2 / 120)


def test_unsafe_template_is_refused(tmp_path):
    template = tmp_path / "tpl"
    subprocess.run(["cp", "-r", str(ROOT / "agent/submission_template"), str(template)], check=True)
    (template / "configs" / "sampling.yaml").write_text("temperature: 0.2\nmax_output_tokens: 16384\n")
    out, r = build(tmp_path, "--template", str(template))
    assert r.returncode != 0 and "max_output_tokens" in (r.stderr + r.stdout)
    assert not list((out / "zips").glob("A_*.zip"))           # no zip written for a failing config
