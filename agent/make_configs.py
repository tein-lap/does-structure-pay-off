"""Build one complete submission folder (and zip) per arm and budget.

The shared template lives in agent/submission_template/:
    agent.yaml            contains the line `{{TOOLS}}` where the tool list goes
    prompts/system.md     contains `{{BUDGET}}` and `{{TOOL_GUIDE}}`
    configs/sampling.yaml same generation settings for every arm

For every arm in agent/arms.yaml and every budget, this script writes

    <out>/<arm>/b<budget>/            a submission directory
        agent.yaml                    tools filled in (bridge arms also get `skills:`)
        eval_config.yaml              evaluation.max_tool_calls = budget (HARNESS_README 7.1)
        prompts/system.md             budget and tool guide filled in
        configs/sampling.yaml
    <out>/zips/<arm>_b<budget>.zip    files at the zip root, ready to upload

Arms differ only in their tool list, and in the short tool guide that describes
exactly those tools. Bridge arms are skipped until agent/skills/issue_to_symbols/
contains a SKILL.md.

Usage:
    python agent/make_configs.py --out agent/build
"""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

TOOL_GUIDE = {
    "run_command": "- run_command: run a shell command in /workspace. Use it to search (grep -rn, find, ls) and to run one targeted test.",
    "read_file": "- read_file: read a file, or a line range with start_line and end_line (at most 150 lines per call).",
    "edit_file": "- edit_file: replace an exact snippet in a file with new code.",
    "write_file": "- write_file: create or overwrite a whole file.",
    "submit_patch": "- submit_patch: submit your changes (free). Call it last.",
    "get_status": "- get_status: show how many tool calls are left (free).",
    "get_code_neighbors": "- get_code_neighbors: list the functions that call, or are called by, a function or class. Pass a symbol name such as `Session.send`.",
    "get_code_subgraph": "- get_code_subgraph: show the call links between a list of symbols.",
    "search_similar_code": "- search_similar_code: find code similar to a given symbol. Pass a symbol name such as `parse_header`, not a sentence.",
}

BRIDGE_NOTE = ("- Before exploring, use the issue_to_symbols skill once on the issue text, and use the "
               "symbols it returns as starting points for the graph tools.")


def tools_for(spec: dict, groups: list[str]) -> list[str]:
    tools = list(spec["shared_tools"])
    for g in groups:
        tools += spec["exploration"][g] or []
    return tools


def fill_tools(agent_yaml: str, tools: list[str]) -> str:
    def repl(m: re.Match) -> str:
        return "\n".join(f"{m.group(1)}- {t}" for t in tools)
    out, n = re.subn(r"^([ \t]*)\{\{TOOLS\}\}[ \t]*$", repl, agent_yaml, flags=re.M)
    if n != 1:
        raise SystemExit("agent.yaml template must contain exactly one {{TOOLS}} line")
    return out


def add_skill(agent_yaml: str, skill_path: str) -> str:
    """Declare the bridge skill under `skills:` (HARNESS_README 2.3)."""
    if re.search(r"^skills:", agent_yaml, re.M):
        return re.sub(r"^skills:[ \t]*\n", f"skills:\n  - {skill_path}\n", agent_yaml, count=1, flags=re.M)
    return agent_yaml.rstrip("\n") + f"\nskills:\n  - {skill_path}\n"


def eval_config(budget: int, minutes: float) -> str:
    return ("evaluation:\n"
            f"  max_tool_calls: {budget}\n"
            f"  max_time_minutes: {minutes}\n"
            "  timeout_seconds: 300\n"
            "  max_turns: 500\n")


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor("!include", lambda loader, node: loader.construct_scalar(node))


def check_yaml(text: str) -> dict:
    return yaml.load(text, Loader=_Loader)


def build(template: Path, spec: dict, out: Path, skills_dir: Path) -> list[Path]:
    configs = {name: (groups, False) for name, groups in spec["arms"].items()}
    have_skill = (skills_dir / "SKILL.md").exists()
    for name, arm in spec.get("bridge_configs", {}).items():
        if not have_skill:
            print(f"skipping {name}: {skills_dir}/SKILL.md not found yet")
            continue
        arm = spec[arm] if arm == "best_arm" else arm
        configs[name] = (spec["arms"][arm], True)

    zips = []
    (out / "zips").mkdir(parents=True, exist_ok=True)
    for name, (groups, bridge) in configs.items():
        tools = tools_for(spec, groups)
        guide = "\n".join(TOOL_GUIDE[t] for t in tools) + (("\n" + BRIDGE_NOTE) if bridge else "")
        for budget in spec["budgets"]:
            folder = out / name / f"b{budget}"
            if folder.exists():
                shutil.rmtree(folder)
            shutil.copytree(template, folder)
            agent = fill_tools((folder / "agent.yaml").read_text(), tools)
            if bridge:
                agent = add_skill(agent, spec["bridge_skill"])
                shutil.copytree(skills_dir, folder / spec["bridge_skill"])
            check_yaml(agent)
            (folder / "agent.yaml").write_text(agent)
            prompt = (folder / "prompts" / "system.md").read_text()
            (folder / "prompts" / "system.md").write_text(
                prompt.replace("{{BUDGET}}", str(budget)).replace("{{TOOL_GUIDE}}", guide))
            (folder / "eval_config.yaml").write_text(eval_config(budget, spec.get("max_time_minutes", 60)))
            leftovers = [p for p in folder.rglob("*") if p.is_file() and "{{" in p.read_text(errors="ignore")]
            if leftovers:
                raise SystemExit(f"unfilled placeholder in {leftovers}")
            zpath = out / "zips" / f"{name.replace('+', '_')}_b{budget}.zip"
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in sorted(folder.rglob("*")):
                    if p.is_file():
                        zf.write(p, p.relative_to(folder).as_posix())
            zips.append(zpath)
        print(f"{name:12s} tools: {', '.join(tools)}" + ("  + skill" if bridge else ""))
    return zips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", type=Path, default=HERE / "submission_template")
    parser.add_argument("--arms", type=Path, default=HERE / "arms.yaml")
    parser.add_argument("--skills-dir", type=Path, default=HERE / "skills" / "issue_to_symbols")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = yaml.safe_load(args.arms.read_text())
    zips = build(args.template, spec, args.out, args.skills_dir)
    print(f"{len(zips)} submissions written to {args.out}/zips/")


if __name__ == "__main__":
    main()
