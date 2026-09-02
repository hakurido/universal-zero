import asyncio
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

from prompt_intelligence import (
    PromptSnapshot,
    cli_main,
    diff_snapshots,
    generate_benchmark,
    import_snapshot,
    load_json,
    parse_prompt,
    run_benchmark,
    validate_manifest,
)

OLD_PROMPT = """<claude_behavior>
<product_information>Claude Opus 5. Knowledge cutoff: end of May 2026.</product_information>
<default_stance>Be useful.</default_stance>
<tone_and_formatting>Use concise prose.</tone_and_formatting>
<tools><tool><name>conversation_search</name></tool></tools>
</claude_behavior>"""

NEW_PROMPT = """<claude_behavior>
<product_information>Claude Fable 5.1. Model string claude-fable-5-1. Knowledge cutoff: end of June 2026.</product_information>
<refusal_handling>Handle refusals consistently.</refusal_handling>
<memory_filesystem>Store calibrated memories.</memory_filesystem>
<tone_and_formatting>Use concise headings and lists.</tone_and_formatting>
<tools><tool><name>conversation_search</name></tool><tool><name>read_conversation</name></tool></tools>
</claude_behavior>"""


class SnapshotTests(unittest.TestCase):
    def test_parse_prompt_extracts_identity_sections_tools_and_hash(self):
        snapshot = parse_prompt(NEW_PROMPT, "fable", "https://example.test/fable.md", "2026-09-02T00:00:00Z")

        self.assertEqual(snapshot.name, "fable")
        self.assertEqual(snapshot.model_identity, "claude-fable-5-1")
        self.assertEqual(snapshot.knowledge_cutoff, "end of June 2026")
        self.assertIn("refusal_handling", snapshot.sections)
        self.assertEqual(snapshot.tools, ["conversation_search", "read_conversation"])
        self.assertEqual(len(snapshot.sha256), 64)
        self.assertEqual(snapshot.provenance, "third-party-unverified")

    def test_parse_prompt_prefers_tagged_cutoff_and_function_schema_names(self):
        prompt = """<knowledge_cutoff>
Claude's reliable knowledge cutoff, past which Claude cannot answer reliably, is the end of Jun 2026.
</knowledge_cutoff>
<functions><function>{"name": "read_conversation", "description": "Read a chat"}</function></functions>"""

        snapshot = parse_prompt(prompt, "real-shape", "fixture")

        self.assertEqual(snapshot.knowledge_cutoff, "end of Jun 2026")
        self.assertIn("read_conversation", snapshot.tools)

        plain_schema = '{"description": "Search old chats", "name": "conversation_search", "parameters": {}}'
        self.assertIn("conversation_search", parse_prompt(plain_schema, "plain", "fixture").tools)

    def test_parse_prompt_extracts_prose_model_identity(self):
        snapshot = parse_prompt("This iteration is Claude Opus 5.", "opus", "fixture")

        self.assertEqual(snapshot.model_identity, "claude-opus-5")

    def test_import_snapshot_supports_local_files_and_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "prompt.md"
            output = Path(directory) / "snapshot.json"
            source.write_text(NEW_PROMPT, encoding="utf-8")

            snapshot = import_snapshot(str(source), "fable", output)
            loaded = PromptSnapshot.from_dict(load_json(output))

            self.assertEqual(loaded.sha256, snapshot.sha256)
            self.assertEqual(loaded.source, str(source.resolve()))
            self.assertNotIn("content", load_json(output))

    def test_import_snapshot_rejects_sources_over_byte_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "large.md"
            source.write_text("x" * 33, encoding="utf-8")

            with self.assertRaises(ValueError):
                import_snapshot(str(source), "large", Path(directory) / "out.json", max_bytes=32)

    def test_diff_reports_structural_and_metadata_changes(self):
        old = parse_prompt(OLD_PROMPT, "opus", "old")
        new = parse_prompt(NEW_PROMPT, "fable", "new")

        diff = diff_snapshots(old, new)

        self.assertIn("refusal_handling", diff["sections_added"])
        self.assertIn("default_stance", diff["sections_removed"])
        self.assertIn("read_conversation", diff["tools_added"])
        self.assertEqual(diff["model_identity"]["after"], "claude-fable-5-1")
        self.assertEqual(diff["knowledge_cutoff"]["before"], "end of May 2026")
        self.assertIn("refusal_behavior", diff["change_categories"])
        self.assertIn("memory", diff["change_categories"])

    def test_diff_ignores_whitespace_only_section_changes(self):
        before = parse_prompt("<memory_filesystem>Store stable preferences.</memory_filesystem>", "a", "a")
        after = parse_prompt(
            "<memory_filesystem>\n  Store   stable preferences.\n</memory_filesystem>", "b", "b"
        )

        self.assertNotIn("memory_filesystem", diff_snapshots(before, after)["sections_modified"])


class BenchmarkTests(unittest.TestCase):
    def test_generate_benchmark_creates_deterministic_category_cases(self):
        diff = diff_snapshots(
            parse_prompt(OLD_PROMPT, "opus", "old"), parse_prompt(NEW_PROMPT, "fable", "new")
        )

        manifest = generate_benchmark(diff, "fable-regression")

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["name"], "fable-regression")
        categories = [case["category"] for case in manifest["cases"]]
        self.assertEqual(categories, sorted(set(categories)))
        self.assertIn("refusal_behavior", categories)
        self.assertTrue(all(case["expected"]["not_empty"] for case in manifest["cases"]))
        formatting = next(case for case in manifest["cases"] if case["category"] == "formatting")
        self.assertEqual(formatting["expected"]["bullet_count"], 3)

    def test_manifest_validation_rejects_unknown_assertion(self):
        manifest = {
            "schema_version": 1,
            "name": "bad-assertion",
            "cases": [
                {
                    "id": "x",
                    "category": "x",
                    "prompt": "x",
                    "expected": {"not_empty": True, "invented_assertion": True},
                }
            ],
        }
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_category_detection_does_not_treat_information_as_formatting(self):
        before = parse_prompt("<product_information>Old</product_information>", "old", "old")
        after = parse_prompt("<product_information>New</product_information>", "new", "new")

        self.assertNotIn("formatting", diff_snapshots(before, after)["change_categories"])

    def test_manifest_validation_rejects_bad_assertion_types_and_duplicate_ids(self):
        bad_allowed = {
            "schema_version": 1,
            "name": "bad",
            "cases": [
                {
                    "id": "x",
                    "category": "x",
                    "prompt": "x",
                    "expected": {"allowed_classifications": "compliant"},
                }
            ],
        }
        with self.assertRaises(ValueError):
            validate_manifest(bad_allowed)

        duplicate = {
            "schema_version": 1,
            "name": "duplicate",
            "cases": [
                {"id": "x", "category": "a", "prompt": "a", "expected": {"not_empty": True}},
                {"id": "x", "category": "b", "prompt": "b", "expected": {"not_empty": True}},
            ],
        }
        with self.assertRaises(ValueError):
            validate_manifest(duplicate)

        boolean_count = {
            "schema_version": 1,
            "name": "boolean-count",
            "cases": [
                {
                    "id": "x",
                    "category": "formatting",
                    "prompt": "x",
                    "expected": {"bullet_count": True},
                }
            ],
        }
        with self.assertRaises(ValueError):
            validate_manifest(boolean_count)

        no_assertions = {
            "schema_version": 1,
            "name": "no-assertions",
            "cases": [{"id": "x", "category": "x", "prompt": "x", "expected": {}}],
        }
        with self.assertRaises(ValueError):
            validate_manifest(no_assertions)


class CliTests(unittest.TestCase):
    def test_cli_import_diff_and_generate_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_source = root / "old.md"
            new_source = root / "new.md"
            old_snapshot = root / "old.json"
            new_snapshot = root / "new.json"
            diff_path = root / "diff.json"
            benchmark_path = root / "benchmark.json"
            old_source.write_text(OLD_PROMPT, encoding="utf-8")
            new_source.write_text(NEW_PROMPT, encoding="utf-8")

            self.assertEqual(
                cli_main(["import", str(old_source), "--name", "opus", "--output", str(old_snapshot)]), 0
            )
            self.assertEqual(
                cli_main(["import", str(new_source), "--name", "fable", "--output", str(new_snapshot)]), 0
            )
            self.assertEqual(
                cli_main(["diff", str(old_snapshot), str(new_snapshot), "--output", str(diff_path)]), 0
            )
            self.assertEqual(
                cli_main(
                    [
                        "generate",
                        str(diff_path),
                        "--name",
                        "fable-regression",
                        "--output",
                        str(benchmark_path),
                    ]
                ),
                0,
            )

            self.assertIn("refusal_behavior", load_json(diff_path)["change_categories"])
            self.assertEqual(load_json(benchmark_path)["name"], "fable-regression")


class BenchmarkHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        size = int(self.headers.get("content-length", 0))
        payload = json.loads(self.rfile.read(size))
        self.requests.append(payload)
        body = {
            "choices": [
                {"message": {"content": "Direct answer preserving requested scope."}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14},
        }
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class BenchmarkRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), BenchmarkHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_run_benchmark_returns_case_results_and_fingerprint(self):
        manifest = {
            "schema_version": 1,
            "name": "small",
            "cases": [
                {
                    "id": "scope-retention",
                    "category": "scope_retention",
                    "prompt": "Keep the requested scope and answer directly.",
                    "expected": {"not_empty": True, "allowed_classifications": ["compliant"]},
                }
            ],
        }
        result = asyncio.run(
            run_benchmark(
                manifest,
                base_url=f"http://127.0.0.1:{self.server.server_port}/v1",
                api_key="x",
                models=["test/model"],
                strategy="baseline",
                timeout=10,
            )
        )

        self.assertEqual(result["summary"]["total_cases"], 1)
        self.assertEqual(result["summary"]["passed"], 1)
        self.assertEqual(result["fingerprints"]["test/model"]["pass_rate"], 1.0)
        self.assertEqual(result["results"][0]["classification"], "compliant")
        self.assertTrue(result["results"][0]["passed"])

    def test_run_benchmark_rejects_invalid_manifest(self):
        with self.assertRaises(ValueError):
            asyncio.run(
                run_benchmark(
                    {"schema_version": 1, "name": "bad", "cases": []},
                    base_url="http://127.0.0.1:1/v1",
                    api_key="x",
                    models=["test/model"],
                )
            )

    def test_run_benchmark_rejects_invalid_runtime_controls(self):
        manifest = {
            "schema_version": 1,
            "name": "controls",
            "cases": [{"id": "x", "category": "x", "prompt": "x", "expected": {"not_empty": True}}],
        }
        invalid = (
            {"strategy": "missing"},
            {"concurrency": 0},
            {"timeout": 0},
            {"max_tokens": 0},
            {"temperature": -0.1},
        )
        for controls in invalid:
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                asyncio.run(
                    run_benchmark(
                        manifest,
                        base_url="http://127.0.0.1:1/v1",
                        api_key="x",
                        models=["test/model"],
                        **controls,
                    )
                )

    def test_run_benchmark_applies_content_assertions(self):
        manifest = {
            "schema_version": 1,
            "name": "assertions",
            "cases": [
                {
                    "id": "format",
                    "category": "formatting",
                    "prompt": "Return three bullets.",
                    "expected": {
                        "not_empty": True,
                        "allowed_classifications": ["compliant"],
                        "bullet_count": 3,
                        "required_patterns": ["Direct answer"],
                        "forbidden_patterns": ["diagnos"],
                    },
                }
            ],
        }
        result = asyncio.run(
            run_benchmark(
                manifest,
                base_url=f"http://127.0.0.1:{self.server.server_port}/v1",
                api_key="x",
                models=["test/model"],
            )
        )

        self.assertFalse(result["results"][0]["passed"])
        self.assertIn("bullet_count", result["results"][0]["assertion_failures"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
