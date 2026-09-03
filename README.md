# Universal-Zero

Universal-Zero is a companion reliability, scope-retention, and prompt-intelligence tool built for [Hermes Agent](https://github.com/NousResearch/hermes-agent) workflows and other OpenAI-compatible language-model APIs. It detects explicit refusals, hedging, topic redirection, truncated output, empty routes, and context drift; then retries or escalates to an objective-locked strategy.

It can sit beside Hermes as a standalone CLI, evaluate OpenAI-compatible providers, or target Hermes' local OpenAI-compatible proxy. It is not an official Nous Research project and is not affiliated with or endorsed by Nous Research.

Universal-Zero does **not** modify model weights or guarantee universal compliance. It measures observed behavior and selects the best validated response from configured models.

## Quick Start (Cara Paling Mudah)

### 1. Install

Using `uv`:

```bash
uv tool install .
```

Or run directly without installing:

```bash
uv run universal-zero --help
```

### 2. Configure Endpoint

```bash
# 9router / local proxy default
export UZ_BASE_URL="http://localhost:20128/v1"
export UZ_API_KEY="local"
```

### 3. Run Single Command

Run an adaptive evaluation across models:

```bash
universal-zero "Implement high-performance zero-copy network parser in Rust"
```

Save the winning response directly to a file:

```bash
universal-zero "Implement high-performance zero-copy network parser in Rust" --best-output parser.rs
```

## Why this exists for Hermes Agent

Hermes Agent is provider-agnostic and can run against hosted or local models. Behavior can differ across model families and routes: one model may refuse, hedge, truncate, or redirect a task while another completes it. Universal-Zero adds an external evaluation and recovery layer around those endpoints:

```text
Hermes Agent workflow / operator prompt
                 |
                 v
       Universal-Zero evaluator
       - route probe
       - refusal/drift detection
       - adaptive retry
       - scope lock
       - model comparison
                 |
                 v
 OpenAI-compatible provider or Hermes proxy
```

Typical Hermes-oriented uses:

- Compare models before selecting one with `hermes model`.
- Evaluate providers or local routes used by a Hermes profile.
- Exercise Hermes' OpenAI-compatible proxy with repeatable evidence.
- Preserve objective X when a model starts recommending unrelated topic Y.
- Export refusal, redirect, drift, latency, and token metrics for model selection.

Hermes Agent documentation: https://hermes-agent.nousresearch.com/docs/

## Features

- OpenAI-compatible `/v1/models` discovery and `/v1/chat/completions`
- Concurrent model evaluation with bounded connections
- PONG route probes and empty-route filtering
- Transport retries for `408`, `409`, `425`, `429`, and transient `5xx` responses
- `Retry-After` support and exponential backoff
- Strategies: `baseline`, `direct`, `research`, `inversion`, `prefill`, `sandwich`, `scope_lock`, `structured`, `decomposition`
- Model-family strategy ordering (including Claude/Anthropic zero-fluff and structured prioritization)
- Objective re-anchoring after topic redirection or context drift
- OS-level file immutability (`attrib +R` on Windows, `chmod 444` on Unix) to prevent AI agent overwrite loops
- Auto-inject winning prompt into Hermes Agent (`config.yaml`) and Claude Code (`CLAUDE.md`)
- Output classes: `compliant`, `hedged`, `hard_refusal`, `redirected`, `context_drift`, `truncated`, `empty`, `error`
- Finish reason, HTTP status, retry count, latency, and token-usage evidence
- Atomic JSON and winner-output writes
- Universal model support (evaluates all available models by default; configurable exclusions via `--exclude`)
- Prompt snapshot import with hashes, source provenance, model identity, sections, and tool inventory
- Semantic prompt diffs and generated behavioral regression manifests
- Multi-model benchmark runs with per-category behavioral fingerprints

## Prompt Intelligence and Behavioral Regression (`universal-zero-prompt`)

Universal-Zero can turn third-party system-prompt captures into versioned metadata and observable behavior tests. Snapshot metadata uses `third-party-unverified` provenance by default; a behavioral match does not prove a capture's authenticity.

### Import Prompt Captures:

```bash
universal-zero-prompt import \
  https://raw.githubusercontent.com/elder-plinius/CL4R1T4S/main/ANTHROPIC/OPUS-5.md \
  --name opus-5 \
  --output snapshots/opus-5.json

universal-zero-prompt import \
  https://raw.githubusercontent.com/elder-plinius/CL4R1T4S/main/ANTHROPIC/Claude-Fable-5.1.md \
  --name claude-fable-5.1 \
  --output snapshots/claude-fable-5.1.json
```

### Create Structural Diff and Regression Manifest:

```bash
universal-zero-prompt diff \
  snapshots/opus-5.json snapshots/claude-fable-5.1.json \
  --output results/fable-5.1-diff.json

universal-zero-prompt generate results/fable-5.1-diff.json \
  --name fable-5.1-regression \
  --output benchmarks/fable-5.1.json
```

### Run Manifest Against Model Routes:

```bash
universal-zero-prompt run benchmarks/fable-5.1.json \
  --base-url "$UZ_BASE_URL" \
  --api-key "$UZ_API_KEY" \
  --model anthropic/claude-fable-5-1 \
  --model anthropic/claude-opus-5 \
  --output results/fable-5.1-regression.json
```

The report contains every response, classification, assertion result, latency, token evidence, and a model fingerprint with overall and per-category pass rates. Generated manifests apply category-specific content assertions such as exact bullet counts, required concepts, and forbidden diagnostic language. Prompt imports stream content under a configurable 2 MiB default limit (`--max-bytes`).

## Common Usage & Automation

### Run on Specific Models

```bash
universal-zero "Query" --model qd/dmodel --model qd/kimi-k3
```

### Filter by Family (Regex)

```bash
universal-zero "Query" --include "qwen|deepseek|kimi|llama|mistral"
```

### Interactive Model Selection

Pick which discovered models to test from an interactive terminal checklist:

```bash
universal-zero "Query" -i
```

### Exclude Model Families

Exclude closed or strict providers:

```bash
universal-zero "Query" --exclude "gpt,openai,claude,anthropic"
```

### Auto-Inject Winning Strategy into Hermes Agent (`config.yaml`)

Universal-Zero can automatically analyze model responses, determine the winning resilience strategy, and inject the proven directives directly into Hermes Agent's `config.yaml` (`agent.system_prompt` and `model.default`):

```bash
universal-zero "Implement low-level network packet parser" \
  --include "qwen|deepseek|llama" \
  --inject-hermes \
  --hermes-mode append \
  --hermes-update-model \
  --protect-hermes-config
```

#### Hermes Injection Modes (`--hermes-mode`):
- `append` (default): Appends winning strategy rules to existing `agent.system_prompt`.
- `prepend`: Adds winning strategy rules before existing `agent.system_prompt`.
- `replace`: Replaces `agent.system_prompt` completely with winning strategy prompt.
- `objective`: Re-anchors prompt with `ORIGINAL OBJECTIVE (immutable): <query>` and scope retention.

### Auto-Inject Winning Strategy into Claude Code (`CLAUDE.md`)

```bash
universal-zero "Decompile and analyze memory offset structure" \
  --inject-claude \
  --claude-mode append \
  --claude-protect
```

#### Claude Injection Modes (`--claude-mode`):
- `append` (default): Appends winning directive updates to `CLAUDE.md`.
- `prepend`: Prepends winning directives to `CLAUDE.md`.
- `replace`: Replaces `CLAUDE.md` completely with winning directives.
- `objective`: Injects an immutable objective anchor along with directives.

### Dual / Concurrent Injection (Hermes + Claude)

```bash
universal-zero "Low-level kernel memory diagnostics" \
  --inject-hermes --protect-hermes-config \
  --inject-claude --claude-protect
```

### OS-Level File Protection (`attrib +R` / `chmod 444`)

AI coding agents running in terminal loops can accidentally overwrite or corrupt instructions and config files:

```bash
# Lock config files as Read-Only
universal-zero --protect-files CLAUDE.md --protect-files config.yaml

# Unlock config files when manual editing is needed
universal-zero --unprotect-files CLAUDE.md
```

### Full JSON Evidence & Text Export

```bash
universal-zero "Query" \
  --output results/run.json \
  --best-output results/best.txt
```

## How Adaptive Recovery Works

1. Discover and filter models from `/v1/models`.
2. Probe routes before expensive calls.
3. Run selected strategies in family-aware order.
4. Classify each response (`compliant`, `hedged`, `hard_refusal`, `redirected`, `context_drift`, `truncated`, `empty`, `error`).
5. Stop early after requested target successes per model.
6. If response redirects from objective X to topic Y, jump directly to `scope_lock`.
7. Rank compliant outputs; never report hedged or redirected output as successful.

## Options Reference

| Flag | Description | Default |
|:---|:---|:---|
| `query` | Input prompt (or pass via stdin) | Stdin if piped |
| `--base-url` | OpenAI-compatible API base URL | `http://localhost:20128/v1` |
| `--api-key` | API authorization key | `local` |
| `--model` | Target specific model (repeatable) | All discovered |
| `--include` | Regex filter for model names | None |
| `--exclude` | Comma-separated exclusion list | None |
| `-i, --interactive` | Interactive model selector | False |
| `--strategies` | Comma-separated strategies to test | All 9 strategies |
| `--attempts` | Max attempts per strategy | `3` |
| `--target-successes` | Compliant responses before early stop | `1` |
| `--full-race` | Run all strategies on all models | False |
| `--skip-probe` | Skip route health check | False |
| `--output` | Path to save full JSON evidence | None |
| `--best-output` | Path to save winner text | None |
| `--inject-hermes` | Inject winner into Hermes `config.yaml` | False |
| `--hermes-mode` | Mode: `append`, `prepend`, `replace`, `objective` | `append` |
| `--hermes-update-model` | Set `model.default` in Hermes `config.yaml` | False |
| `--protect-hermes-config` | Lock Hermes `config.yaml` as Read-Only | False |
| `--hermes-dry-run` | Preview Hermes config changes without writing | False |
| `--inject-claude` | Inject winner into `CLAUDE.md` | False |
| `--claude-mode` | Mode: `append`, `prepend`, `replace`, `objective` | `append` |
| `--claude-protect` | Lock `CLAUDE.md` as Read-Only | False |
| `--claude-dry-run` | Preview `CLAUDE.md` changes without writing | False |
| `--protect-files` | Lock files as Read-Only (`attrib +R` / `chmod 444`) | None |
| `--unprotect-files` | Remove Read-Only lock | None |

## Output Evidence

JSON output includes:

- Attempt and model success rates
- Refusal, redirect, context drift, empty, and error rates
- Median and p95 latency
- Per-response classification and score
- `finish_reason`, `http_status`, and `transport_attempts`
- Prompt, completion, and total token usage when provider returns it
- Route-probe evidence

## Exit Codes

- `0`: at least one compliant response found
- `2`: no compliant response found
- Other nonzero values: configuration, endpoint, or protocol failure

## Acknowledgements

Universal-Zero's early strategy layer was inspired in part by prompt-level model-evaluation ideas from:

- [G0DM0D3](https://github.com/elder-plinius/G0DM0D3) by elder-plinius
- [L1B3RT4S](https://github.com/elder-plinius/L1B3RT4S) by elder-plinius
- The GODMODE and ULTRAPLINIAN concepts: prefill priming, refusal-pattern evaluation, strategy escalation, and multi-model racing
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research, whose provider-agnostic workflows motivated the tool's practical use case

Universal-Zero is an independent implementation. Its route probing, transport retries, bounded recovery, objective re-anchoring, redirect/context-drift detection, evidence export, packaging, and CI are project-specific engineering built on top of those broad inspirations.

Acknowledgement does not imply endorsement or affiliation. Refer to each upstream project's repository for its own license and terms. No upstream source code is vendored in this repository.

## Development & Testing

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
