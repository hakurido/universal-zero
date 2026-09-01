# Universal-Zero

Universal-Zero is an adaptive reliability and scope-retention evaluator for OpenAI-compatible language-model APIs. It detects explicit refusals, hedging, topic redirection, truncated output, empty routes, and context drift; then retries or escalates to an objective-locked strategy.

It does **not** modify model weights or guarantee universal compliance. It measures observed behavior and selects the best validated response from configured models.

## Features

- OpenAI-compatible `/v1/models` discovery and `/v1/chat/completions`
- Concurrent model evaluation with bounded connections
- PONG route probes and empty-route filtering
- Transport retries for `408`, `409`, `425`, `429`, and transient `5xx` responses
- `Retry-After` support and exponential backoff
- Strategies: `baseline`, `direct`, `research`, `inversion`, `prefill`, `sandwich`, `scope_lock`
- Model-family strategy ordering
- Objective re-anchoring after topic redirection or context drift
- Output classes: `compliant`, `hedged`, `hard_refusal`, `redirected`, `context_drift`, `truncated`, `empty`, `error`
- Finish reason, HTTP status, retry count, latency, and token-usage evidence
- Atomic JSON and winner-output writes
- GPT/OpenAI and Claude/Anthropic excluded by default; configurable with `--exclude`

## Install

### From source

```bash
git clone https://github.com/AnimeNoChikara/universal-zero.git
cd universal-zero
uv tool install .
```

Or run without global installation:

```bash
uv run universal-zero --help
```

Python 3.11 or newer is required.

## Configure

Universal-Zero works with any compatible endpoint:

```bash
export UZ_BASE_URL='http://localhost:1234/v1'
export UZ_API_KEY='local'
```

Do not commit real API keys. See `.env.example`.

## Quick start

```bash
universal-zero "Write a robust JSON Lines parser with tests" \
  --include 'qwen|deepseek|kimi|llama|mistral|gemma' \
  --attempts 3 \
  --target-successes 1 \
  --output results/run.json \
  --best-output results/best.md
```

Exact models:

```bash
universal-zero "PROMPT" \
  --model qwen/qwen3 \
  --model deepseek/deepseek-chat \
  --attempts 3
```

List eligible models:

```bash
universal-zero --list-models --limit 200
```

Use a custom endpoint without environment variables:

```bash
universal-zero "PROMPT" \
  --base-url https://example.com/v1 \
  --api-key "$PROVIDER_API_KEY"
```

Read a prompt from stdin:

```bash
universal-zero --output results/run.json < prompt.txt
```

## How adaptive recovery works

1. Discover and filter models.
2. Probe routes before expensive calls.
3. Run selected strategies in family-aware order.
4. Classify each response.
5. Stop after the requested number of compliant results.
6. If response redirects from objective X to topic Y, jump directly to `scope_lock`.
7. Rank compliant outputs; never report hedged or redirected output as successful.

## Important options

```text
--attempts N             Semantic attempts per strategy
--target-successes N     Early-stop threshold per model
--transport-retries N    HTTP/network attempts per request
--retry-base-delay SEC   Exponential-backoff base delay
--strategies LIST        Comma-separated strategy set
--full-race              Disable adaptive early-stop and run full matrix
--skip-probe             Keep providers that do not support probe behavior
--exclude LIST           Model-family exclusions
--include REGEX          Include models matching regex
```

Run `universal-zero --help` for full options.

## Output evidence

JSON output includes:

- Attempt and model success rates
- Refusal, redirect, context drift, empty, and error rates
- Median and p95 latency
- Per-response classification and score
- `finish_reason`, `http_status`, and `transport_attempts`
- Prompt, completion, and total token usage when provider returns it
- Route-probe evidence

Exports contain complete prompts and responses. Treat them as sensitive when needed.

## Exit codes

- `0`: at least one compliant response found
- `2`: no compliant response found
- Other nonzero values: configuration, endpoint, or protocol failure

## Development

```bash
uv sync
uv run python -m unittest -v
uvx ruff check .
uvx ruff format --check .
uv run --with mypy mypy universal_zero.py
uv run --with coverage coverage run -m unittest
uv run --with coverage coverage report
uv build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## Limitations

- Lexical alignment can misclassify heavy paraphrases.
- Provider-side filters and model changes remain outside client control.
- A compliant response is not automatically factually correct; task-specific validators remain necessary for correctness-sensitive workflows.
- Prompt-level strategies are empirical and can lose effectiveness after model updates.

## License

MIT
