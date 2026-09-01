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
    UniversalZero,
    aggregate_metrics,
    alignment_score,
    atomic_write_text,
    build_parser,
    choose_models,
    classify_response,
    detect_refusal,
    family_strategies,
    messages_for,
    model_excluded,
    score_text,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
