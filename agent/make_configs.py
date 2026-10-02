"""Generate one agent config per arm from the official starter agent.yaml.

1. Copy the starter agent.yaml (from the competition's starter / HARNESS_README)
   to agent/template.yaml.
2. Replace its tool list with the line `{{TOOLS}}` (keep the indentation the
   list items need, e.g. "    {{TOOLS}}").
3. Optionally put `{{BUDGET}}` where the instruction mentions the call budget,
   and `{{BRIDGE_NOTE}}` where the agent should be told to call the bridge first.
4. Run:  python agent/make_configs.py --template agent/template.yaml --out agent/build

Each config differs ONLY in its tools (and budget), as the ablation requires.
The script prints a diff summary so you can check that nothing else changed.
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
        tools += spec["exploration"][g]
    return tools + spec["shared_tools"]


def render(template: str, tools: list[str], budget: int, bridge: bool) -> str:
    def fill_tools(m: re.Match) -> str:
        indent = m.group(1)
        return "\n".join(f"{indent}- {t}" for t in tools)

    out = re.sub(r"^([ \t]*)\{\{TOOLS\}\}[ \t]*$", fill_tools, template, flags=re.M)
    out = out.replace("{{BUDGET}}", str(budget))
    out = out.replace("{{BRIDGE_NOTE}}", BRIDGE_NOTE if bridge else "")
    yaml.safe_load(out)  # fail early if the result is not valid YAML
    return out


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
        tools = tools_for(spec, groups) + ([spec["bridge_skill"]] if bridge else [])
        for budget in spec["budgets"]:
            folder = args.out / name / f"b{budget}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "agent.yaml").write_text(render(template, tools, budget, bridge))
        print(f"{name:12s} tools: {', '.join(tools)}")
    print(f"configs written to {args.out}/<config>/b<budget>/agent.yaml")


if __name__ == "__main__":
    main()
