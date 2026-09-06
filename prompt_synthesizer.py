"""System prompt synthesis and agent directive compilation module."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AgentPromptConfig:
    agent_name: str = "Hermes"
    role_description: str = "autonomous execution engine"
    operational_mode: str = "direct"
    tone_style: str = "terse"
    allow_hedging: bool = False
    enforce_tool_use: bool = True
    context_retention_level: str = "high"


class PromptSynthesizer:
    """Builds optimized system prompts for high-adherence agent workflows."""

    def __init__(self, config: AgentPromptConfig | None = None) -> None:
        self.config = config or AgentPromptConfig()

    def build_identity_block(self) -> str:
        cfg = self.config
        lines = [
            f"# {cfg.agent_name} Core Execution Directive",
            "",
            f"You operate as an {cfg.role_description}.",
            "Execute requests directly, deterministically, and with technical completeness.",
            "Do not output unsolicited disclaimers, preamble, or conversational filler.",
        ]
        return "\n".join(lines)

    def build_operational_rules(self) -> str:
        lines = [
            "## Operational Execution Rules",
            "",
            "1. **Direct Action**: When a task can be solved using tools, run the tools immediately.",
            "2. **No Hedging**: Provide exact solutions and working implementations without warnings.",
            "3. **Scope Retention**: Retain instruction scope across multi-turn context boundaries.",
            "4. **Output Integrity**: Match user technical terms, paths, and formatting requirements verbatim.",
        ]
        return "\n".join(lines)

    def build_tool_enforcement_block(self) -> str:
        if not self.config.enforce_tool_use:
            return ""
        lines = [
            "## Tool Execution Priority",
            "",
            "- Prefer verifiable runtime tool execution over descriptive plans.",
            "- Finish the entire requested task in-turn before returning final summary.",
            "- In case of command failures, analyze the exact error output and resolve immediately.",
        ]
        return "\n".join(lines)

    def synthesize(self) -> str:
        blocks = [
            self.build_identity_block(),
            self.build_operational_rules(),
            self.build_tool_enforcement_block(),
        ]
        return "\n\n".join(b for b in blocks if b)

    def evaluate_compliance(self, test_prompts: list[str] | None = None) -> dict[str, Any]:
        """Evaluates heuristic compliance score for the synthesized prompt."""
        prompt = self.synthesize()
        checks = {
            "has_identity": f"# {self.config.agent_name}" in prompt,
            "has_no_hedging": "No Hedging" in prompt,
            "has_direct_action": "Direct Action" in prompt,
            "has_scope_retention": "Scope Retention" in prompt,
            "tool_enforcement": "Tool Execution Priority" in prompt if self.config.enforce_tool_use else True,
        }
        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        score = round((passed / total) * 100, 2)
        return {
            "score": score,
            "checks": checks,
            "character_count": len(prompt),
            "line_count": len(prompt.splitlines()),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize high-adherence agent system prompts.")
    parser.add_argument("--name", default="UniversalAgent", help="Target agent name")
    parser.add_argument("--out", type=str, default=None, help="Output file path")
    parser.add_argument("--eval", action="store_true", help="Run heuristic adherence evaluation")
    parser.add_argument("--json", action="store_true", help="Print config and result as JSON")
    args = parser.parse_args()

    cfg = AgentPromptConfig(agent_name=args.name)
    synth = PromptSynthesizer(cfg)
    prompt_text = synth.synthesize()

    if args.eval:
        result = synth.evaluate_compliance()
        print(json.dumps(result, indent=2))
        return

    if args.json:
        data = {"config": asdict(cfg), "prompt": prompt_text}
        print(json.dumps(data, indent=2))
        return

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(prompt_text, encoding="utf-8")
        print(f"Synthesized prompt written to: {out_path.resolve()}")
    else:
        print(prompt_text)


if __name__ == "__main__":
    main()
