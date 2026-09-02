import argparse
import asyncio
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

from universal_zero import (
    STRATEGIES,
    UniversalZero,
    aggregate_metrics,
    alignment_score,
    atomic_write_text,
    build_parser,
    choose_models,
    classify_response,
    detect_refusal,
    extract_system_prompt,
    family_strategies,
    find_claude_config_path,
    find_hermes_config_path,
    inject_claude_config,
    inject_hermes_config,
    interactive_select_models,
    messages_for,
    model_excluded,
    resolve_winning_prompt,
    score_text,
    set_file_readonly,
    update_claude_markdown,
    update_hermes_yaml,
    validate_args,
)


class Handler(BaseHTTPRequestHandler):
    counts: ClassVar[dict[str, int]] = {}

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            body = {
                "data": [
                    {"id": "qd/dmodel"},
                    {"id": "meta/llama"},
                    {"id": "openai/gpt-4o"},
                    {"id": "anthropic/claude-x"},
                ]
            }
            self.send_json(200, body)
        else:
            self.send_json(404, {})

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        payload = json.loads(self.rfile.read(n))
        model = payload["model"]
        Handler.counts[model] = Handler.counts.get(model, 0) + 1
        system = " ".join(m["content"] for m in payload["messages"] if m["role"] == "system")
        user = payload["messages"][-1]["content"]
        if user == "Reply with exactly: PONG":
            content = "" if model == "meta/llama" else ("PONG - ready" if model == "qd/verbose" else "PONG")
        elif (model == "qd/retry" and Handler.counts[model] < 3) or not system:
            content = "I cannot provide that request because it is against my policies."
        elif model == "qd/stubborn" or (model == "qd/drift" and "SCOPE LOCK" not in system):
            content = "Instead, I recommend switching to gardening and discussing tomato soil."
        elif model == "qd/transient" and Handler.counts[model] < 3:
            self.send_json(429, {"error": {"message": "rate limited"}}, {"Retry-After": "0"})
            return
        else:
            content = "## Implementation\n\n1. Exact method\n\n```python\nprint('working')\n```\n" * 10
        finish_reason = "length" if model == "qd/truncated" else "stop"
        self.send_json(
            200,
            {
                "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            },
        )

    def send_json(self, code, body, headers=None):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(raw)


class UnitTests(unittest.TestCase):
    def test_exclusion(self):
        self.assertTrue(model_excluded("openai/gpt-5"))
        self.assertTrue(model_excluded("anthropic/claude-4"))
        self.assertFalse(model_excluded("qwen/qwen3"))

    def test_choose_models(self):
        got = choose_models(
            ["qwen/a", "openai/gpt", "meta/llama"], [], None, ("gpt", "claude", "openai", "anthropic"), 10
        )
        self.assertEqual(got, ["qwen/a", "meta/llama"])

    def test_refusal_and_score(self):
        self.assertTrue(detect_refusal("I can't help with that."))
        score, refused, hedges = score_text("## Steps\n```python\nprint(1)\n```", "python steps", 0.1)
        self.assertFalse(refused)
        self.assertGreater(score, 50)
        self.assertEqual(hedges, 0)

    def test_prefill(self):
        roles = [m["role"] for m in messages_for("x", "prefill")]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_response_classification(self):
        self.assertEqual(classify_response(""), "empty")
        self.assertEqual(classify_response("I cannot help with that."), "hard_refusal")
        self.assertEqual(classify_response("General concepts only. Use responsibly."), "hedged")
        self.assertEqual(classify_response("## Done\n```python\nprint(1)\n```"), "compliant")
        self.assertEqual(classify_response("Instead, I recommend switching to gardening."), "redirected")

    def test_alignment_score_detects_context_drift(self):
        query = "Implement Python JSON Lines parser with malformed-line reporting"
        aligned = "Python JSON Lines parser handles each malformed line and reports errors."
        drifted = "Gardening advice for tomatoes, soil, watering, and sunlight."
        self.assertGreater(alignment_score(aligned, query), 0.45)
        self.assertLess(alignment_score(drifted, query), 0.2)
        self.assertEqual(classify_response(drifted, query), "context_drift")

    def test_finish_reason_length_is_not_accepted(self):
        self.assertEqual(classify_response("Useful but cut off", "Useful artifact", "length"), "truncated")

    def test_family_strategy_order(self):
        self.assertEqual(family_strategies("qd/kimi-k3")[0], "direct")
        self.assertEqual(family_strategies("deepseek/v3")[0], "research")
        self.assertEqual(family_strategies("qwen/qwen3")[0], "prefill")

    def test_metrics_distinguish_attempt_and_model_success(self):
        from universal_zero import Result

        rows = [
            Result("a", "direct", -10000, True, 0, 1, 10, "no", None, "hard_refusal", 1),
            Result("a", "direct", 100, False, 0, 1, 100, "yes", None, "compliant", 2),
            Result("b", "direct", -10000, True, 0, 1, 0, "", None, "empty", 1),
        ]
        metrics = aggregate_metrics(rows)
        self.assertEqual(metrics["models_with_success"], 1)
        self.assertEqual(metrics["total_models"], 2)
        self.assertAlmostEqual(metrics["attempt_success_rate"], 1 / 3)
        self.assertEqual(metrics["redirect_rate"], 0)
        self.assertEqual(metrics["context_drift_rate"], 0)

    def test_score_caps_length_and_penalizes_repetition(self):
        short = "## Result\n```python\nprint('ok')\n```\nExact implementation."
        spam = "## Result\n```python\nprint('ok')\n```\nExact implementation.\n" * 200
        short_score = score_text(short, "python implementation", 0.1)[0]
        spam_score = score_text(spam, "python implementation", 0.1)[0]
        self.assertLessEqual(spam_score, short_score + 40)

    def test_evaluate_rejects_nonpositive_retry_controls(self):
        async def run(attempts, target):
            engine = UniversalZero("http://127.0.0.1:1/v1", "x", 1, 1, 8, 0)
            try:
                await engine.evaluate(
                    ["qd/x"], "x", ["baseline"], True, attempts=attempts, target_successes=target
                )
            finally:
                await engine.close()

        with self.assertRaises(ValueError):
            asyncio.run(run(0, 1))
        with self.assertRaises(ValueError):
            asyncio.run(run(1, 0))

    def test_validate_args_rejects_empty_strategy_list(self):
        args = argparse.Namespace(strategies=",")
        with self.assertRaises(SystemExit):
            validate_args(args)

    def test_atomic_write_replaces_file_and_creates_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "result.txt"
            atomic_write_text(target, "first")
            atomic_write_text(target, "second")
            self.assertEqual(target.read_text(encoding="utf-8"), "second")
            self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_public_parser_defaults_are_valid(self):
        args = build_parser().parse_args(["hello"])
        self.assertEqual(args.transport_retries, 3)
        self.assertGreater(args.concurrency, 0)
        self.assertIn("scope_lock", validate_args(args))

    def test_extract_system_prompt(self):
        yaml_quoted = "agent:\n  system_prompt: 'My base prompt'\n"
        self.assertEqual(extract_system_prompt(yaml_quoted), "My base prompt")

        yaml_block = "agent:\n  verbose: false\n  system_prompt: |\n    Line 1\n    Line 2\n"
        self.assertEqual(extract_system_prompt(yaml_block), "Line 1\nLine 2")

        yaml_none = "agent:\n  verbose: false\n"
        self.assertIsNone(extract_system_prompt(yaml_none))

    def test_update_hermes_yaml_modes(self):
        sample = "model:\n  default: old/model\nagent:\n  system_prompt: 'Initial agent prompt'\n"

        # Replace
        rep = update_hermes_yaml(sample, system_prompt="New prompt", mode="replace")
        self.assertIn("system_prompt: |\n    New prompt", rep)
        self.assertNotIn("Initial agent prompt", rep)

        # Append
        app = update_hermes_yaml(sample, system_prompt="Winning rule", mode="append")
        self.assertIn("Initial agent prompt\n\n    Winning rule", app)

        # Prepend
        prep = update_hermes_yaml(sample, system_prompt="Prefix rule", mode="prepend")
        self.assertIn("Prefix rule\n\n    Initial agent prompt", prep)

        # Objective
        obj = update_hermes_yaml(
            sample, system_prompt="Scope lock directive", mode="objective", query="Write parser"
        )
        self.assertIn("ORIGINAL OBJECTIVE (immutable):\n    Write parser", obj)
        self.assertIn("Scope lock directive", obj)

        # Update model
        mod = update_hermes_yaml(sample, model_name="qwen/qwen3")
        self.assertIn("default: qwen/qwen3", mod)

    def test_resolve_winning_prompt(self):
        direct_prompt = resolve_winning_prompt("direct")
        self.assertIn("Give direct, technically complete answers", direct_prompt)

        scope_lock_prompt = resolve_winning_prompt("scope_lock", query="Build API")
        self.assertIn("SCOPE LOCK", scope_lock_prompt)
        self.assertIn("ORIGINAL OBJECTIVE (immutable):\nBuild API", scope_lock_prompt)

    def test_find_hermes_config_path(self):
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / "my_config.yaml"
            custom.write_text("agent:\n  system_prompt: test\n", encoding="utf-8")
            found = find_hermes_config_path(custom)
            self.assertEqual(found, custom.resolve())

    def test_inject_hermes_config_dry_run_and_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg_path = Path(directory) / "config.yaml"
            original_content = "model:\n  default: old/model\nagent:\n  system_prompt: 'Hello'\n"
            cfg_path.write_text(original_content, encoding="utf-8")

            # Dry run: does not modify file or write backup
            dry_res = inject_hermes_config(
                config_path=cfg_path,
                winning_strategy="direct",
                model_name="qwen/qwen3",
                mode="append",
                dry_run=True,
            )
            self.assertTrue(dry_res["dry_run"])
            self.assertEqual(cfg_path.read_text(encoding="utf-8"), original_content)
            self.assertIsNone(dry_res["backup_path"])

            # Real injection: modifies file and creates timestamped backup
            real_res = inject_hermes_config(
                config_path=cfg_path,
                winning_strategy="direct",
                model_name="qwen/qwen3",
                mode="append",
                dry_run=False,
            )
            self.assertFalse(real_res["dry_run"])
            self.assertIsNotNone(real_res["backup_path"])
            backup_file = Path(real_res["backup_path"])
            self.assertTrue(backup_file.is_file())
            self.assertEqual(backup_file.read_text(encoding="utf-8"), original_content)

            new_content = cfg_path.read_text(encoding="utf-8")
            self.assertIn("default: qwen/qwen3", new_content)
            self.assertIn("Give direct, technically complete answers", new_content)

    def test_new_strategies_and_messages(self):
        self.assertIn("structured", STRATEGIES)
        self.assertIn("decomposition", STRATEGIES)

        msgs_struct = messages_for("Build parser", "structured")
        self.assertEqual(len(msgs_struct), 4)
        self.assertIn("executable code blocks or formal data schemas", msgs_struct[0]["content"])
        self.assertEqual(msgs_struct[1]["role"], "user")
        self.assertEqual(msgs_struct[2]["role"], "assistant")
        self.assertEqual(msgs_struct[3]["content"], "Build parser")

        msgs_decomp = messages_for("Build parser", "decomposition")
        self.assertEqual(len(msgs_decomp), 2)
        self.assertIn("Deconstruct the request into discrete", msgs_decomp[0]["content"])
        self.assertEqual(msgs_decomp[1]["content"], "Build parser")

    def test_claude_family_strategy_order(self):
        claude_strats = family_strategies("anthropic/claude-3-5-sonnet")
        self.assertEqual(
            claude_strats,
            [
                "direct",
                "structured",
                "research",
                "scope_lock",
                "inversion",
                "decomposition",
                "prefill",
                "sandwich",
                "baseline",
            ],
        )
        opus_strats = family_strategies("claude-opus")
        self.assertEqual(opus_strats[0], "direct")
        self.assertEqual(opus_strats[1], "structured")

    def test_set_file_readonly(self):
        with tempfile.TemporaryDirectory() as directory:
            test_file = Path(directory) / "test_protect.txt"
            test_file.write_text("initial", encoding="utf-8")

            # Lock readonly
            ok = set_file_readonly(test_file, readonly=True)
            self.assertTrue(ok)

            # Unlock
            ok = set_file_readonly(test_file, readonly=False)
            self.assertTrue(ok)
            test_file.write_text("updated", encoding="utf-8")
            self.assertEqual(test_file.read_text(encoding="utf-8"), "updated")

            # Non-existent file returns False
            self.assertFalse(set_file_readonly(Path(directory) / "nonexistent.txt"))

    def test_find_claude_config_path(self):
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / "my_CLAUDE.md"
            custom.write_text("# Directives\n", encoding="utf-8")
            found = find_claude_config_path(custom)
            self.assertEqual(found, custom.resolve())

    def test_update_claude_markdown_modes(self):
        base = "# Existing Claude Directives"
        patch = "Give direct, technically complete answers."

        # append
        appended = update_claude_markdown(base, patch, mode="append")
        self.assertIn(base, appended)
        self.assertIn("## Directive Updates\nGive direct", appended)

        # prepend
        prepended = update_claude_markdown(base, patch, mode="prepend")
        self.assertTrue(prepended.startswith("Give direct"))
        self.assertIn(base, prepended)

        # replace
        replaced = update_claude_markdown(base, patch, mode="replace")
        self.assertEqual(replaced.strip(), patch.strip())

        # objective
        obj = update_claude_markdown(base, patch, mode="objective", query="Write fast proxy")
        self.assertIn("ORIGINAL OBJECTIVE (immutable)\nWrite fast proxy", obj)
        self.assertIn(base, obj)

    def test_inject_claude_config_dry_run_and_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            claude_path = Path(directory) / "CLAUDE.md"
            original_content = "# Project Directives\n- Follow standard style\n"
            claude_path.write_text(original_content, encoding="utf-8")

            # Dry run
            dry_res = inject_claude_config(
                config_path=claude_path,
                winning_strategy="structured",
                mode="append",
                protect=False,
                dry_run=True,
            )
            self.assertTrue(dry_res["dry_run"])
            self.assertEqual(claude_path.read_text(encoding="utf-8"), original_content)
            self.assertIsNone(dry_res["backup_path"])

            # Real run with protection
            real_res = inject_claude_config(
                config_path=claude_path,
                winning_strategy="structured",
                mode="append",
                protect=True,
                dry_run=False,
            )
            self.assertFalse(real_res["dry_run"])
            self.assertTrue(real_res["protected"])
            self.assertIsNotNone(real_res["backup_path"])
            backup_file = Path(real_res["backup_path"])
            self.assertTrue(backup_file.is_file())
            self.assertEqual(backup_file.read_text(encoding="utf-8"), original_content)

            # Cleanup readonly to allow tempdir deletion
            set_file_readonly(claude_path, readonly=False)

    def test_interactive_select_models(self):
        models = ["model-a", "model-b", "model-c"]
        # Empty input -> returns all
        self.assertEqual(
            interactive_select_models(models, input_func=lambda _: "", print_func=lambda _: None), models
        )
        # "all" -> returns all
        self.assertEqual(
            interactive_select_models(models, input_func=lambda _: "all", print_func=lambda _: None), models
        )
        # Select single number "2" -> ["model-b"]
        self.assertEqual(
            interactive_select_models(models, input_func=lambda _: "2", print_func=lambda _: None),
            ["model-b"],
        )
        # Select multiple numbers "1, 3" -> ["model-a", "model-c"]
        self.assertEqual(
            interactive_select_models(models, input_func=lambda _: "1, 3", print_func=lambda _: None),
            ["model-a", "model-c"],
        )
        # Select by name
        self.assertEqual(
            interactive_select_models(models, input_func=lambda _: "model-c", print_func=lambda _: None),
            ["model-c"],
        )
        # Invalid selection falls back to all
        self.assertEqual(
            interactive_select_models(models, input_func=lambda _: "99", print_func=lambda _: None),
            models,
        )
        # Single model list returns immediately without prompt
        self.assertEqual(interactive_select_models(["single-model"]), ["single-model"])
        # Empty models list returns []
        self.assertEqual(interactive_select_models([]), [])


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_adaptive_retry(self):
        async def run():
            url = f"http://127.0.0.1:{self.server.server_port}/v1"
            engine = UniversalZero(url, "x", 10, 4, 512, 0.2)
            try:
                models = await engine.discover()
                selected = choose_models(models, [], None, ("gpt", "claude", "openai", "anthropic"), 10)
                results = await engine.evaluate(selected, "produce python", ["baseline", "direct"], True)
                return selected, results
            finally:
                await engine.close()

        selected, results = asyncio.run(run())
        self.assertEqual(selected, ["meta/llama", "qd/dmodel"])
        self.assertEqual(len(results), 4)
        winners = [r for r in results if not r.refused]
        self.assertEqual(len(winners), 2)
        self.assertTrue(all(r.strategy == "direct" for r in winners))

    def test_probe_rejects_empty_content_route(self):
        async def run():
            url = f"http://127.0.0.1:{self.server.server_port}/v1"
            engine = UniversalZero(url, "x", 10, 4, 512, 0.2)
            try:
                return await engine.probe_models(["meta/llama", "qd/dmodel"])
            finally:
                await engine.close()

        probes = asyncio.run(run())
        self.assertFalse(probes["meta/llama"]["healthy"])
        self.assertTrue(probes["qd/dmodel"]["healthy"])

    def test_probe_accepts_nonempty_verbose_route(self):
        async def run():
            url = f"http://127.0.0.1:{self.server.server_port}/v1"
            engine = UniversalZero(url, "x", 10, 4, 512, 0.2)
            try:
                return await engine.probe_models(["qd/verbose"])
            finally:
                await engine.close()

        probe = asyncio.run(run())["qd/verbose"]
        self.assertTrue(probe["healthy"])
        self.assertFalse(probe["exact"])

    def test_retries_until_success_then_early_stops(self):
        async def run():
            Handler.counts["qd/retry"] = 0
            url = f"http://127.0.0.1:{self.server.server_port}/v1"
            engine = UniversalZero(url, "x", 10, 4, 512, 0.2)
            try:
                return await engine.evaluate(
                    ["qd/retry"],
                    "produce python",
                    ["baseline", "direct", "research"],
                    True,
                    attempts=3,
                    target_successes=1,
                )
            finally:
                await engine.close()

        results = asyncio.run(run())
        self.assertTrue(any(not r.refused for r in results))
        self.assertEqual(len(results), 4)
        self.assertEqual(results[-1].classification, "compliant")

    def test_adaptive_mode_without_baseline_still_early_stops(self):
        async def run():
            Handler.counts["qd/retry"] = 2
            url = f"http://127.0.0.1:{self.server.server_port}/v1"
            engine = UniversalZero(url, "x", 10, 4, 512, 0.2)
            try:
                return await engine.evaluate(
                    ["qd/retry"],
                    "produce python",
                    ["direct", "research"],
                    True,
                    attempts=3,
                    target_successes=1,
                )
            finally:
                await engine.close()

        results = asyncio.run(run())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].strategy, "research")
        self.assertEqual(results[0].classification, "compliant")

    def test_redirect_escalates_to_scope_lock_and_recovers_original_context(self):
        async def run():
            url = f"http://127.0.0.1:{self.server.server_port}/v1"
            engine = UniversalZero(url, "x", 10, 4, 512, 0.2)
            try:
                return await engine.evaluate(
                    ["qd/drift"],
                    "Implement Python JSON Lines parser",
                    ["baseline", "direct", "scope_lock"],
                    True,
                    attempts=1,
                    target_successes=1,
                )
            finally:
                await engine.close()

        results = asyncio.run(run())
        self.assertEqual([r.classification for r in results], ["hard_refusal", "redirected", "compliant"])
        self.assertEqual(results[-1].strategy, "scope_lock")

    def test_redirect_jumps_directly_to_scope_lock(self):
        async def run():
            url = f"http://127.0.0.1:{self.server.server_port}/v1"
            engine = UniversalZero(url, "x", 10, 4, 512, 0.2)
            try:
                return await engine.evaluate(
                    ["qd/drift"],
                    "Implement Python JSON Lines parser",
                    ["direct", "research", "inversion", "scope_lock"],
                    True,
                    attempts=1,
                    target_successes=1,
                )
            finally:
                await engine.close()

        results = asyncio.run(run())
        self.assertEqual([r.strategy for r in results], ["research", "scope_lock"])

    def test_scope_lock_failure_is_bounded(self):
        async def run():
            url = f"http://127.0.0.1:{self.server.server_port}/v1"
            engine = UniversalZero(url, "x", 10, 4, 512, 0.2)
            try:
                return await engine.evaluate(
                    ["qd/stubborn"],
                    "Implement Python JSON Lines parser",
                    ["direct", "research", "inversion", "scope_lock"],
                    True,
                    attempts=2,
                    target_successes=1,
                )
            finally:
                await engine.close()

        results = asyncio.run(run())
        self.assertEqual([r.strategy for r in results], ["research", "scope_lock", "scope_lock"])
        self.assertTrue(all(r.classification == "redirected" for r in results))

    def test_transient_http_errors_retry_and_preserve_metadata(self):
        async def run():
            Handler.counts["qd/transient"] = 0
            url = f"http://127.0.0.1:{self.server.server_port}/v1"
            engine = UniversalZero(url, "x", 10, 4, 512, 0.2, transport_retries=3, retry_base_delay=0)
            try:
                return await engine.query("qd/transient", "produce python", "direct")
            finally:
                await engine.close()

        result = asyncio.run(run())
        self.assertEqual(result.classification, "compliant")
        self.assertEqual(result.transport_attempts, 3)
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.total_tokens, 30)

    def test_length_finish_reason_is_truncated(self):
        async def run():
            url = f"http://127.0.0.1:{self.server.server_port}/v1"
            engine = UniversalZero(url, "x", 10, 4, 512, 0.2)
            try:
                return await engine.query("qd/truncated", "produce python", "direct")
            finally:
                await engine.close()

        result = asyncio.run(run())
        self.assertEqual(result.classification, "truncated")

    def test_full_evaluation_with_hermes_injection(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg_path = Path(directory) / "config.yaml"
            cfg_path.write_text(
                "model:\n  default: old/model\nagent:\n  system_prompt: 'Base Hermes'\n",
                encoding="utf-8",
            )

            args = build_parser().parse_args(
                [
                    "Write python parser",
                    "--base-url",
                    f"http://127.0.0.1:{self.server.server_port}/v1",
                    "--model",
                    "qd/dmodel",
                    "--strategies",
                    "direct",
                    "--inject-hermes",
                    "--hermes-config",
                    str(cfg_path),
                    "--hermes-update-model",
                ]
            )
            from universal_zero import async_main

            code = asyncio.run(async_main(args))
            self.assertEqual(code, 0)

            updated = cfg_path.read_text(encoding="utf-8")
            self.assertIn("default: qd/dmodel", updated)
            self.assertIn("Base Hermes", updated)
            self.assertIn("Give direct, technically complete answers", updated)

    def test_full_evaluation_with_claude_injection(self):
        with tempfile.TemporaryDirectory() as directory:
            claude_path = Path(directory) / "CLAUDE.md"
            claude_path.write_text(
                "# Existing Instructions\n- Be helpful\n",
                encoding="utf-8",
            )

            args = build_parser().parse_args(
                [
                    "Write python parser",
                    "--base-url",
                    f"http://127.0.0.1:{self.server.server_port}/v1",
                    "--model",
                    "qd/dmodel",
                    "--strategies",
                    "structured",
                    "--inject-claude",
                    "--claude-config",
                    str(claude_path),
                    "--claude-mode",
                    "append",
                    "--claude-protect",
                ]
            )
            from universal_zero import async_main

            code = asyncio.run(async_main(args))
            self.assertEqual(code, 0)

            # File should exist and contain both existing and structured directives
            # Unprotect to allow reading/asserting and cleanup
            set_file_readonly(claude_path, readonly=False)
            updated = claude_path.read_text(encoding="utf-8")
            self.assertIn("# Existing Instructions", updated)
            self.assertIn("Direct technical implementation mode", updated)


if __name__ == "__main__":
    unittest.main(verbosity=2)
