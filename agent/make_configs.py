"""Build one complete submission folder (and zip) per arm and budget.

The shared template lives in agent/submission_template/:
    agent.yaml            placeholders {{INSTRUCTION}}, {{TOOLS}}, {{SAMPLING}}
    prompts/system.md     the instruction; placeholders {{BUDGET}}, {{MINUTES}} ("4 minutes"), {{TOOL_GUIDE}}
    configs/sampling.yaml generation settings, the same for every arm

The prompt and the sampling settings are INLINED into agent.yaml. The submission
never reads another file (no !include), so agent.yaml loads with any standard
YAML loader and cannot fail because a referenced file is missing.

For every arm in agent/arms.yaml and every budget, this script writes

    <out>/<arm>/b<budget>/agent.yaml        tools, instruction and sampling filled in
    <out>/<arm>/b<budget>/eval_config.yaml  max_tool_calls, max_time_minutes, timeout_seconds, max_turns
    <out>/zips/<arm>_b<budget>.zip          files at the zip root, ready to upload
                                            (timeout_seconds for one command = a third of the task time, max 60 s)
    <out>/zips/SMOKE_<arm>_b<budget>.zip    the minimal "verify fundamentals" submission

max_time_minutes is calculated from the runtime section of arms.yaml so that the
worst case (every hidden task using its full time) fits inside the total limit
with a safety margin. Every zip passes these checks before it is written:
    - agent.yaml loads with yaml.safe_load (no custom tags)
    - no {{placeholder}} is left anywhere
    - agent.yaml and eval_config.yaml sit at the zip root
    - worst-case runtime <= total limit * safety

Bridge arms are skipped until agent/skills/issue_to_symbols/ contains a SKILL.md.

Usage:
    python agent/make_configs.py --out agent/build [--total-hours 12] [--parallel-tasks 1]
"""

from __future__ import annotations

import argparse
import math
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
    "submit_patch": "- submit_patch: submit your changes (free). Call it as soon as you have a plausible fix.",
    "get_status": "- get_status: show how many tool calls are left (free). Use it at most once every 10 actions.",
    "get_code_neighbors": "- get_code_neighbors: list the functions that call, or are called by, a function or class. Pass a symbol name such as `Session.send`.",
    "get_code_subgraph": "- get_code_subgraph: show the call links between a list of symbols.",
    # HARNESS_README 6.3: each result carries its full source and there is no output cap, so a
    # small k keeps one call from flooding the 32,768-token context.
    "search_similar_code": "- search_similar_code: find code similar to a given symbol. Pass a symbol name such as `parse_header`, not a sentence, and set k=5 or less.",
}

GRAPH_TOOLS = ("get_code_neighbors", "get_code_subgraph", "search_similar_code")

# HARNESS_README 5.2: the harness appends a "Code Intelligence Tools" section to the task
# message whenever graph data exists, for EVERY agent. Arms without some of these tools must
# be told not to call them, or the comparison between arms is not fair.
MISSING_TOOLS_NOTE = ("- Only the tools listed above are available to you. The task message may also mention {missing}; "
                      "you do not have {them}, so do not call {them}.")

BRIDGE_NOTE = ("- Before exploring, run the issue_to_symbols skill once (its script scripts/issue_to_symbols.py) "
               "with the issue text, and use the symbols it prints as starting points for the graph tools.")

# HARNESS_README 2.4: the only file types a submission may contain.
ALLOWED_EXTENSIONS = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}

# Shipped next to the skill script so it runs with the standard library only in the sandbox.
SKILL_MODULES = ("bridge.py", "symbols.py")

MAX_OUTPUT_TOKENS = 4096   # keep each reply far below the ~14k-token compaction point of the 32k context


def tools_for(spec: dict, groups: list[str]) -> list[str]:
    tools = list(spec["shared_tools"])
    for g in groups:
        tools += spec["exploration"][g] or []
    return tools


def minutes_per_task(runtime: dict) -> int:
    """Largest whole number of minutes per task whose worst case fits the total limit."""
    total = runtime["total_hours"] * 60 * runtime["safety"]
    return max(1, math.floor(total * runtime.get("parallel_tasks", 1) / runtime["hidden_tasks"]))


def command_timeout(runtime: dict, minutes: int) -> int:
    """Seconds one run_command may take: at most a third of the task's time, at least 15 s."""
    return min(runtime["timeout_seconds"], max(15, minutes * 60 // 3))


def worst_case_minutes(runtime: dict, minutes: int) -> float:
    return runtime["hidden_tasks"] * minutes / runtime.get("parallel_tasks", 1)


def fill_block(text: str, key: str, block: str) -> str:
    """Replace the line `<indent>{{KEY}}` with `block`, every line at that indent."""
    def repl(m: re.Match) -> str:
        indent = m.group(1)
        return "\n".join((indent + line) if line.strip() else "" for line in block.splitlines())
    out, n = re.subn(r"^([ \t]*)\{\{" + key + r"\}\}[ \t]*$", repl, text, flags=re.M)
    if n != 1:
        raise SystemExit(f"agent.yaml template must contain exactly one {{{{{key}}}}} line")
    return out


def add_skill(agent_yaml: str, skill_path: str) -> str:
    """Declare the bridge skill under `skills:` (HARNESS_README 2.3)."""
    if re.search(r"^skills:", agent_yaml, re.M):
        return re.sub(r"^skills:[ \t]*\n", f"skills:\n  - {skill_path}\n", agent_yaml, count=1, flags=re.M)
    return agent_yaml.rstrip("\n") + f"\nskills:\n  - {skill_path}\n"


def eval_config(budget: int, minutes: int, runtime: dict) -> str:
    return ("evaluation:\n"
            f"  max_tool_calls: {budget}\n"
            f"  max_time_minutes: {minutes}\n"
            f"  timeout_seconds: {command_timeout(runtime, minutes)}\n"
            f"  max_turns: {budget + runtime['extra_turns']}\n")


def render_agent(template: Path, tools: list[str], budget: int, minutes: int, bridge: bool,
                 sampling_overrides: dict | None = None, test_timeout: int = 60) -> str:
    guide = "\n".join(TOOL_GUIDE[t] for t in tools)
    missing = [t for t in GRAPH_TOOLS if t not in tools]
    if missing:
        names = ", ".join(missing)
        guide += "\n" + MISSING_TOOLS_NOTE.format(missing=names, them="it" if len(missing) == 1 else "them")
    if bridge:
        guide += "\n" + BRIDGE_NOTE
    prompt = (template / "prompts" / "system.md").read_text()
    prompt = prompt.replace("{{BUDGET}}", str(budget))
    prompt = prompt.replace("{{MINUTES}}", f"{minutes} minute" + ("" if minutes == 1 else "s"))
    prompt = prompt.replace("{{TOOL_GUIDE}}", guide).replace("{{TEST_TIMEOUT}}", str(test_timeout))

    sampling = yaml.safe_load((template / "configs" / "sampling.yaml").read_text())
    for key, value in (sampling_overrides or {}).items():
        if key == "thinking_budget":
            thinking = sampling.setdefault("thinking_config", {})
            thinking["thinking_budget"] = value
            thinking["include_thoughts"] = value > 0     # HARNESS_README 2.4: both are needed to turn thinking on
        else:
            sampling[key] = value
    sampling_text = yaml.safe_dump(sampling, sort_keys=False).rstrip("\n")

    agent = (template / "agent.yaml").read_text()
    agent = fill_block(agent, "TOOLS", "\n".join(f"- {t}" for t in tools))
    agent = fill_block(agent, "INSTRUCTION", prompt.rstrip("\n"))
    agent = fill_block(agent, "SAMPLING", sampling_text)
    return agent


def check_submission(folder: Path, budget: int, minutes: int, runtime: dict, tools: list[str]) -> None:
    """Sanity checks; any failure stops the build before a bad zip is written."""
    text = (folder / "agent.yaml").read_text()
    try:
        agent = yaml.safe_load(text)          # standard loader: fails on custom tags such as !include
    except yaml.YAMLError as exc:
        raise SystemExit(f"{folder}/agent.yaml is not plain YAML: {exc}") from exc
    problems = []
    if agent.get("tools") != tools:
        problems.append(f"tools are {agent.get('tools')}, expected {tools}")
    instruction = agent.get("instruction") or ""
    if f"You have {budget} tool calls" not in instruction or "submit_patch" not in instruction:
        problems.append("instruction is missing the budget or the submit rule")
    # HARNESS_README 2.3/5.1: ADK fills {name} placeholders in the instruction from session state;
    # an unknown one can break the agent at runtime, so the prompt must contain no braces at all.
    if re.search(r"[{}]", instruction):
        problems.append(f"instruction contains {{...}} placeholders ADK would try to fill: {re.findall(r'[{][^{}]*[}]', instruction)}")
    missing = [t for t in GRAPH_TOOLS if t not in tools]
    if missing and not all(t in instruction.split("Only the tools listed above")[-1] for t in missing):
        problems.append(f"instruction does not warn that {missing} are unavailable")
    sampling = agent.get("generate_content_config") or {}
    if not isinstance(sampling, dict) or sampling.get("max_output_tokens", 0) > MAX_OUTPUT_TOKENS:
        problems.append(f"generate_content_config missing or max_output_tokens > {MAX_OUTPUT_TOKENS}")
    thinking = sampling.get("thinking_config") or {} if isinstance(sampling, dict) else {}
    if thinking and (thinking.get("thinking_budget", 0) > 0) != bool(thinking.get("include_thoughts")):
        problems.append("thinking_budget and include_thoughts disagree (HARNESS_README 2.4 needs both to enable thinking)")
    ev = yaml.safe_load((folder / "eval_config.yaml").read_text())["evaluation"]
    if ev["max_tool_calls"] != budget or ev["max_time_minutes"] != minutes:
        problems.append("eval_config.yaml does not match the budget or time")
    if ev["timeout_seconds"] * 3 > max(45, minutes * 60) or f"timeout {ev['timeout_seconds']} " not in instruction:
        problems.append("command timeout is not a third of the task time, or the prompt uses a different one")
    total = runtime["total_hours"] * 60
    if worst_case_minutes(runtime, minutes) > total * runtime["safety"]:
        problems.append(f"worst case {worst_case_minutes(runtime, minutes):.0f} min > "
                        f"{runtime['safety']:.0%} of the {total:.0f}-minute limit")
    bad_types = [p.name for p in folder.rglob("*") if p.is_file() and p.suffix not in ALLOWED_EXTENSIONS]
    if bad_types:
        problems.append(f"file types the harness rejects: {bad_types}")
    for skill in agent.get("skills") or []:
        manifest = folder / skill / "SKILL.md"
        if not manifest.is_file() or not manifest.read_text().startswith("---\nname:"):
            problems.append(f"{skill}/SKILL.md missing or without 'name:' front matter")
    leftovers = [p.name for p in folder.rglob("*") if p.is_file() and p.suffix in {".yaml", ".yml"}
                 and "{{" in p.read_text(errors="ignore")]
    if leftovers:
        problems.append(f"unfilled placeholder in {leftovers}")
    if problems:
        raise SystemExit(f"{folder}: " + "; ".join(problems))


def write_zip(folder: Path, zpath: Path) -> Path:
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(folder).as_posix())
    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
    if not {"agent.yaml", "eval_config.yaml"} <= names:
        raise SystemExit(f"{zpath}: agent.yaml and eval_config.yaml must be at the zip root")
    return zpath


def build_one(template: Path, folder: Path, tools: list[str], budget: int, minutes: int, runtime: dict,
              bridge: bool = False, bridge_skill: str = "", skills_dir: Path | None = None,
              sampling_overrides: dict | None = None) -> None:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    agent = render_agent(template, tools, budget, minutes, bridge, sampling_overrides, command_timeout(runtime, minutes))
    if bridge:
        agent = add_skill(agent, bridge_skill)
        target = folder / bridge_skill
        shutil.copytree(skills_dir, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for module in SKILL_MODULES:
            shutil.copy(HERE.parent / "dspo" / module, target / "scripts" / module)
    (folder / "agent.yaml").write_text(agent)
    (folder / "eval_config.yaml").write_text(eval_config(budget, minutes, runtime))
    check_submission(folder, budget, minutes, runtime, tools)


def build(template: Path, spec: dict, out: Path, skills_dir: Path) -> list[Path]:
    runtime = spec["runtime"]
    minutes = minutes_per_task(runtime)
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

    smoke = spec.get("smoke")
    if smoke:
        tools = tools_for(spec, spec["arms"][smoke["arm"]])
        folder = out / "SMOKE" / f"{smoke['arm']}_b{smoke['budget']}"
        smoke_minutes = min(smoke["max_time_minutes"], minutes)
        build_one(template, folder, tools, smoke["budget"], smoke_minutes, runtime,
                  sampling_overrides={"thinking_budget": smoke["thinking_budget"]})
        zips.append(write_zip(folder, out / "zips" / f"SMOKE_{smoke['arm']}_b{smoke['budget']}.zip"))
        print(f"SMOKE        arm {smoke['arm']}, {smoke['budget']} calls, {smoke_minutes} min/task")

    for name, (groups, bridge) in configs.items():
        tools = tools_for(spec, groups)
        for budget in spec["budgets"]:
            folder = out / name / f"b{budget}"
            build_one(template, folder, tools, budget, minutes, runtime, bridge, spec["bridge_skill"], skills_dir)
            zips.append(write_zip(folder, out / "zips" / f"{name.replace('+', '_')}_b{budget}.zip"))
        print(f"{name:12s} tools: {', '.join(tools)}" + ("  + skill" if bridge else ""))
    print(f"time per task: {minutes} min; worst case {worst_case_minutes(runtime, minutes):.0f} of "
          f"{runtime['total_hours'] * 60:.0f} min ({runtime['hidden_tasks']} tasks, "
          f"{runtime.get('parallel_tasks', 1)} in parallel, safety {runtime['safety']})")
    return zips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", type=Path, default=HERE / "submission_template")
    parser.add_argument("--arms", type=Path, default=HERE / "arms.yaml")
    parser.add_argument("--skills-dir", type=Path, default=HERE / "skills" / "issue_to_symbols")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total-hours", type=float, help="override runtime.total_hours (default 12, from the competition Overview > Evaluation)")
    parser.add_argument("--parallel-tasks", type=int, help="override runtime.parallel_tasks")
    args = parser.parse_args()
    spec = yaml.safe_load(args.arms.read_text())
    if args.total_hours:
        spec["runtime"]["total_hours"] = args.total_hours
    if args.parallel_tasks:
        spec["runtime"]["parallel_tasks"] = args.parallel_tasks
    zips = build(args.template, spec, args.out, args.skills_dir)
    print(f"{len(zips)} submissions written to {args.out}/zips/")


if __name__ == "__main__":
    main()
