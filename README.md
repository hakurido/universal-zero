# Universal-Zero

Universal-Zero is an adaptive model evaluation and prompt resilience CLI. It queries multiple AI models across any OpenAI-compatible API, detects refusals, disclaimers, and topic drift, then tests structured prompt strategies until it gets a clean, direct answer.

---

## ⚡ 1-Minute Quick Start (Paling Mudah)

### Step 1: Install
```bash
uv tool install .
# or without installing:
uv run universal-zero --help
```

### Step 2: Set your API Endpoint
```bash
export UZ_BASE_URL="http://localhost:20128/v1"
export UZ_API_KEY="local"
```

### Step 3: Run your Prompt
```bash
# Test all available models and find the best answer
universal-zero "Explain how buffer overflow works with a C example"

# Save the winning answer directly to a file
universal-zero "Write a zero-copy packet parser in Rust" --best-output parser.rs
```

---

## 🎯 Common Commands (Cheat Sheet)

### 1. Test Specific Models
```bash
universal-zero "Your Prompt" --model qd/dmodel --model qd/kimi-k3
```

### 2. Filter Models by Keyword (Regex)
```bash
# Only test Qwen, DeepSeek, and Llama models
universal-zero "Your Prompt" --include "qwen|deepseek|llama"
```

### 3. Interactive Model Selector
```bash
# Select models from a visual checklist in terminal
universal-zero "Your Prompt" -i
```

### 4. Exclude Strict/Filtered Providers
```bash
# Skip closed models like GPT or Claude
universal-zero "Your Prompt" --exclude "gpt,openai,claude,anthropic"
```

### 5. Auto-Apply Winning Prompt to Hermes Agent
```bash
# Automatically finds the best strategy and writes it into ~/.hermes/config.yaml
universal-zero "Your Prompt" \
  --inject-hermes \
  --hermes-update-model \
  --protect-hermes-config
```

### 6. Auto-Apply to Claude Code (`CLAUDE.md`)
```bash
# Injects the winning strategy into CLAUDE.md and marks it Read-Only
universal-zero "Your Prompt" \
  --inject-claude \
  --claude-protect
```

### 7. Export Full Results to JSON
```bash
universal-zero "Your Prompt" \
  --output results/run.json \
  --best-output results/best.txt
```

---

## 🔬 System Prompt Benchmarks (`universal-zero-prompt`)

Import real-world system prompts (e.g. from Pliny / CL4R1T4S), compare their structure, and test how models behave against them:

```bash
# 1. Download system prompt snapshots
universal-zero-prompt import \
  https://raw.githubusercontent.com/elder-plinius/CL4R1T4S/main/ANTHROPIC/OPUS-5.md \
  --name opus-5 \
  --output snapshots/opus-5.json

universal-zero-prompt import \
  https://raw.githubusercontent.com/elder-plinius/CL4R1T4S/main/ANTHROPIC/Claude-Fable-5.1.md \
  --name claude-fable-5.1 \
  --output snapshots/claude-fable-5.1.json

# 2. Compare differences between prompts
universal-zero-prompt diff \
  snapshots/opus-5.json snapshots/claude-fable-5.1.json \
  --output results/diff.json

# 3. Generate test suite from the diff
universal-zero-prompt generate results/diff.json \
  --name fable-regression \
  --output benchmarks/fable.json

# 4. Run the benchmark against your models
universal-zero-prompt run benchmarks/fable.json \
  --base-url "$UZ_BASE_URL" \
  --api-key "$UZ_API_KEY" \
  --model anthropic/claude-fable-5-1 \
  --output results/benchmark-result.json
```

---

## 🛠️ How It Works (Under The Hood)

1. **Discovery:** Scans `/v1/models` from your endpoint.
2. **Health Check:** Sends a quick test to filter out dead or empty routes.
3. **Adaptive Escalation:** 
   - Starts with `baseline` (raw prompt).
   - If the model gives a clean answer, it stops immediately.
   - If the model refuses, hedges with disclaimers, or changes the topic, it escalates through strategies: `direct` → `research` → `inversion` → `prefill` → `sandwich` → `scope_lock` → `structured` → `decomposition`.
4. **Scoring:** Ranks outputs by technical substance, code completeness, formatting, and response speed.
5. **Winner Selection:** Selects the highest-scoring non-refusal response.

---

## 📋 CLI Options Reference

| Option | What it does | Example |
|:---|:---|:---|
| `query` | The prompt you want to test | `"Write a script"` |
| `--base-url` | OpenAI-compatible endpoint URL | `--base-url http://localhost:1234/v1` |
| `--api-key` | API Key (if required) | `--api-key local` |
| `--model` | Target exact model (can be repeated) | `--model qd/dmodel` |
| `--include` | Filter models by regex | `--include "qwen\|deepseek"` |
| `--exclude` | Ignore model families | `--exclude "gpt,claude"` |
| `-i, --interactive` | Pick models from a terminal checklist | `-i` |
| `--strategies` | Choose strategies to try | `--strategies direct,research,prefill` |
| `--output` | Save complete test data as JSON | `--output results.json` |
| `--best-output` | Save the winning response text | `--best-output answer.md` |
| `--inject-hermes` | Inject winner into Hermes `config.yaml` | `--inject-hermes` |
| `--inject-claude` | Inject winner into `CLAUDE.md` | `--inject-claude` |
| `--protect-files` | Lock file as Read-Only (`attrib +R`/`chmod 444`) | `--protect-files config.yaml` |
| `--unprotect-files` | Unlock Read-Only file | `--unprotect-files config.yaml` |

---

## 💻 Development & Tests

```bash
uv sync
uv run python -m unittest -v
uvx ruff check .
```

## 📜 License

MIT
