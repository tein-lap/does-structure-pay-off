"""Generate one agent config per arm from the official starter agent.yaml.

1. Copy the starter agent.yaml (from the competition's starter / HARNESS_README)
   to agent/template.yaml.
2. Replace its tool list with the line `{{TOOLS}}` (keep the indentation the
   list items need, e.g. "    {{TOOLS}}").
3. Optionally put `{{BUDGET}}` where the instruction mentions the call budget,
   and `{{BRIDGE_NOTE}}` where the agent should be told to call the bridge first.
4. Run:  python agent/make_configs.py --template agent/template.yaml --out agent/build

Each output folder is a complete submission directory:
    agent.yaml         tools filled in; bridge configs also get `skills: [skills/issue_to_symbols]`
    eval_config.yaml   evaluation.max_tool_calls = the budget (HARNESS_README 7.1)
Each config differs ONLY in its tools (and budget), as the ablation requires.
"""

import argparse
import re
from pathlib import Path

import yaml

BRIDGE_NOTE = ("Before exploring, call the issue_to_symbols skill once with the issue text "
               "and use the symbols it returns as starting points for the graph tools.")


def tools_for(spec: dict, groups: list[str]) -> list[str]:
    tools = []
    for g in groups:
        tools += spec["exploration"][g] or []
    return spec["shared_tools"] + tools


def render(template: str, tools: list[str], budget: int, bridge: bool) -> str:
    def fill_tools(m: re.Match) -> str:
        indent = m.group(1)
        return "\n".join(f"{indent}- {t}" for t in tools)

    out = re.sub(r"^([ \t]*)\{\{TOOLS\}\}[ \t]*$", fill_tools, template, flags=re.M)
    out = out.replace("{{BUDGET}}", str(budget))
    out = out.replace("{{BRIDGE_NOTE}}", BRIDGE_NOTE if bridge else "")
    yaml.safe_load(out)  # fail early if the result is not valid YAML
    return out


def add_skill(agent_yaml: str, skill_path: str) -> str:
    """Declare the bridge skill under `skills:` (HARNESS_README 2.3), keeping the rest unchanged."""
    if re.search(r"^skills:", agent_yaml, re.M):
        return re.sub(r"^skills:[ \t]*\n", f"skills:\n  - {skill_path}\n", agent_yaml, count=1, flags=re.M)
    return agent_yaml.rstrip("\n") + f"\nskills:\n  - {skill_path}\n"


def eval_config(budget: int, minutes: float) -> str:
    return f"evaluation:\n  max_tool_calls: {budget}\n  max_time_minutes: {minutes}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--arms", type=Path, default=Path(__file__).with_name("arms.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    template = args.template.read_text()
    if "{{TOOLS}}" not in template:
        raise SystemExit("template has no {{TOOLS}} line; see the instructions at the top of this file")
    spec = yaml.safe_load(args.arms.read_text())

    configs = {name: (groups, False) for name, groups in spec["arms"].items()}
    for name, arm in spec.get("bridge_configs", {}).items():
        arm = spec[arm] if arm == "best_arm" else arm
        configs[name] = (spec["arms"][arm], True)

    for name, (groups, bridge) in configs.items():
        tools = tools_for(spec, groups)
        for budget in spec["budgets"]:
            folder = args.out / name / f"b{budget}"
            folder.mkdir(parents=True, exist_ok=True)
            text = render(template, tools, budget, bridge)
            if bridge:
                text = add_skill(text, spec["bridge_skill"])
            yaml.safe_load(text)
            (folder / "agent.yaml").write_text(text)
            (folder / "eval_config.yaml").write_text(eval_config(budget, spec.get("max_time_minutes", 60)))
        print(f"{name:12s} tools: {', '.join(tools)}" + (f"  + skill {spec['bridge_skill']}" if bridge else ""))
    print(f"configs written to {args.out}/<config>/b<budget>/agent.yaml")


if __name__ == "__main__":
    main()
