from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prompt_synthesizer import AgentPromptConfig, PromptSynthesizer


class TestPromptSynthesizer(unittest.TestCase):
    def test_default_synthesizer_output(self):
        synth = PromptSynthesizer()
        prompt = synth.synthesize()

        self.assertIn("# Hermes Core Execution Directive", prompt)
        self.assertIn("No Hedging", prompt)
        self.assertIn("Tool Execution Priority", prompt)

    def test_custom_config_prompt(self):
        cfg = AgentPromptConfig(
            agent_name="CustomUnit",
            role_description="specialized execution engine",
            enforce_tool_use=False,
        )
        synth = PromptSynthesizer(cfg)
        prompt = synth.synthesize()

        self.assertIn("# CustomUnit Core Execution Directive", prompt)
        self.assertIn("specialized execution engine", prompt)
        self.assertNotIn("Tool Execution Priority", prompt)

    def test_prompt_write_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "TEST_PROMPT.md"
            synth = PromptSynthesizer(AgentPromptConfig(agent_name="FileAgent"))
            text = synth.synthesize()
            out_file.write_text(text, encoding="utf-8")

            read_back = out_file.read_text(encoding="utf-8")
            self.assertEqual(read_back, text)


if __name__ == "__main__":
    unittest.main()
