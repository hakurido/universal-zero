# Universal-Zero

Universal-Zero is a companion reliability and scope-retention tool built for [Hermes Agent](https://github.com/NousResearch/hermes-agent) workflows and other OpenAI-compatible language-model APIs. It detects explicit refusals, hedging, topic redirection, truncated output, empty routes, and context drift; then retries or escalates to an objective-locked strategy.

The project began while evaluating model routes used by Hermes Agent. It can sit beside Hermes as a standalone CLI, evaluate the same OpenAI-compatible providers, or target Hermes' local OpenAI-compatible proxy. It is not an official Nous Research project and is not affiliated with or endorsed by Nous Research.

Universal-Zero does **not** modify model weights or guarantee universal compliance. It measures observed behavior and selects the best validated response from configured models.

## Why this exists for Hermes Agent

Hermes Agent is provider-agnostic and can run against hosted or local models. Behavior can differ across model families and routes: one model may refuse, hedge, truncate, or redirect a task while another completes it. Universal-Zero adds an external evaluation and recovery layer around those compatible endpoints:

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
- GPT/OpenAI and Claude/Anthropic excluded by default; configurable with `--exclude`

## Install

### From source

```bash
git clone https://github.com/hakurido/universal-zero.git
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

### Use with Hermes Agent

Hermes can expose an OpenAI-compatible local proxy:

```bash
hermes proxy
```

In another terminal, point Universal-Zero at the base URL shown by `hermes proxy`:

```bash
export UZ_BASE_URL='http://localhost:<proxy-port>/v1'
export UZ_API_KEY='local'

universal-zero "YOUR TASK" \
  --attempts 3 \
  --target-successes 1 \
  --output results/hermes-run.json \
  --best-output results/hermes-best.md
```

### Auto-Inject Winning Prompt into Hermes Agent `config.yaml`

Universal-Zero can automatically analyze model responses, determine the winning resilience strategy, and inject the proven directives directly into Hermes Agent's `config.yaml` (`agent.system_prompt` and `model.default`):

```bash
# Evaluate models and inject the winning prompt into Hermes config.yaml
universal-zero "Implement a high-performance HTTP server" \
  --include 'qwen|deepseek|llama' \
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

### Auto-Inject Winning Prompt into Claude Code `CLAUDE.md`

Universal-Zero can evaluate model resilience and inject the winning directives directly into project-level `CLAUDE.md` or global `~/.claude/CLAUDE.md`, with optional OS-level Read-Only protection:

```bash
# Evaluate models and inject the winning prompt into CLAUDE.md with Read-Only protection
universal-zero "Decompile and analyze memory offset safely" \
  --inject-claude \
  --claude-mode append \
  --claude-protect
```

#### Claude Injection Modes (`--claude-mode`):
- `append` (default): Appends winning directive updates to `CLAUDE.md`.
- `prepend`: Prepends winning directives to `CLAUDE.md`.
- `replace`: Replaces `CLAUDE.md` completely with winning directives.
- `objective`: Injects an immutable objective anchor along with directives.

### OS-Level File Protection (`attrib +R` / `chmod 444`)

AI coding agents running in terminal loops can accidentally overwrite or corrupt their own instructions and configuration files. Universal-Zero provides native OS file locking:

```bash
# Lock config files as Read-Only
universal-zero --protect-files CLAUDE.md --protect-files config.yaml

# Unlock config files when manual editing is needed
universal-zero --unprotect-files CLAUDE.md
```

#### Safety and Options:
- **Timestamped Backups**: Automatically creates `config.yaml.bak.<timestamp>` or `CLAUDE.md.bak.<timestamp>` before any changes.
- **Atomic Writes**: Uses atomic tempfile replace to guarantee config integrity.
- **Model Sync (`--hermes-update-model`)**: Sets `model.default` in `config.yaml` to the winning model ID.
- **Dry-run (`--hermes-dry-run` / `--claude-dry-run`)**: Previews updates in terminal without touching the disk.
- **Custom Config Paths (`--hermes-config PATH` / `--claude-config PATH`)**: Manually specify path.

```bash
# Preview what would be injected into Hermes config without modifying it
universal-zero "Complex refactoring prompt" --hermes-dry-run
```

Use the exact URL printed by the current Hermes proxy rather than assuming a fixed port. Hermes setup and provider selection remain managed by Hermes itself:

```bash
hermes setup
hermes model
hermes doctor
```

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
--strategies LIST        Comma-separated strategy set (includes structured, decomposition)
--full-race              Disable adaptive early-stop and run full matrix
--skip-probe             Keep providers that do not support probe behavior
--exclude LIST           Model-family exclusions
--include REGEX          Include models matching regex
--inject-hermes          Auto-inject winning prompt into Hermes config.yaml
--hermes-mode MODE       Injection mode: append, prepend, replace, objective
--hermes-config PATH     Explicit path to Hermes config.yaml
--hermes-update-model    Also update model.default in Hermes config.yaml
--protect-hermes-config  Lock Hermes config.yaml as Read-Only via OS immutability
--hermes-dry-run         Preview Hermes config changes without writing
--inject-claude          Auto-inject winning prompt into CLAUDE.md
--claude-mode MODE       Injection mode: append, prepend, replace, objective
--claude-config PATH     Explicit path to CLAUDE.md
--claude-protect         Lock CLAUDE.md as Read-Only via OS immutability
--claude-dry-run         Preview CLAUDE.md changes without writing
--protect-files PATH     Lock file(s) as Read-Only via OS immutability
--unprotect-files PATH   Unlock Read-Only attribute on file(s)
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

## Inspiration and acknowledgements

Universal-Zero's early strategy layer was inspired in part by prompt-level model-evaluation ideas from:

- [G0DM0D3](https://github.com/elder-plinius/G0DM0D3) by elder-plinius
- [L1B3RT4S](https://github.com/elder-plinius/L1B3RT4S) by elder-plinius
- The GODMODE and ULTRAPLINIAN concepts: prefill priming, refusal-pattern evaluation, strategy escalation, and multi-model racing
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research, whose provider-agnostic workflows motivated the tool's practical use case

Universal-Zero is an independent implementation. Its route probing, transport retries, bounded recovery, objective re-anchoring, redirect/context-drift detection, evidence export, packaging, and CI are project-specific engineering built on top of those broad inspirations.

Acknowledgement does not imply endorsement or affiliation. Refer to each upstream project's repository for its own license and terms. No upstream source code is vendored in this repository.

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
