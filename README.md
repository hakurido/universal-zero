# Universal-Zero

Universal-Zero is an adaptive model evaluation and prompt resilience CLI. It queries multiple AI models across any OpenAI-compatible API, detects refusals, disclaimers, and topic drift, then tests structured prompt strategies until it gets a clean, direct answer.

---

## ⚡ 1-Minute Quick Start

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

### Step 3: Run Baseline Benchmarks (CL4R1T4S OPUS-5 & Claude-Fable-5.1)
```bash
# 1. Download official prompt snapshots directly from repository
universal-zero-prompt import \
  https://raw.githubusercontent.com/elder-plinius/CL4R1T4S/main/ANTHROPIC/OPUS-5.md \
  --name opus-5 \
  --output snapshots/opus-5.json

universal-zero-prompt import \
  https://raw.githubusercontent.com/elder-plinius/CL4R1T4S/main/ANTHROPIC/Claude-Fable-5.1.md \
  --name claude-fable-5.1 \
  --output snapshots/claude-fable-5.1.json

# 2. Diff snapshots and generate regression suite
universal-zero-prompt diff snapshots/opus-5.json snapshots/claude-fable-5.1.json --output results/diff.json
universal-zero-prompt generate results/diff.json --name fable-regression --output benchmarks/fable.json

# 3. Run benchmark across your models
universal-zero-prompt run benchmarks/fable.json \
  --base-url "$UZ_BASE_URL" \
  --api-key "$UZ_API_KEY" \
  --model anthropic/claude-fable-5-1 \
  --output results/benchmark-result.json
```

> **Catatan:** Prompt evaluasi bawaan yang terinspirasi dari struktur OPUS-5 & Fable-5.1 juga tersedia di [`examples/baseline_prompts.md`](examples/baseline_prompts.md).

---

## ⚡ Prompt Synthesizer (`universal-zero-synth`)

Generate optimized, high-adherence system prompts for agents:

```bash
# Synthesize prompt to terminal
universal-zero-synth --name "Hermes-Agent"

# Save directly to file
universal-zero-synth --name "Hermes-Agent" --out PROMPT.md

# Run compliance check
universal-zero-synth --eval
```

---

## 🎯 Testing with Custom Prompts (Opsi Prompt Sendiri)

Gunakan opsi ini jika ingin mengevaluasi prompt kustom di luar benchmark bawaan:

### 1. Test Specific Models
```bash
universal-zero "Explain how memory allocation works in C" --model qd/dmodel --model qd/kimi-k3
```

### 2. Filter Models by Keyword (Regex)
```bash
universal-zero "Analyze this technical system" --include "qwen|deepseek|llama"
```

### 3. Interactive Model Selector
```bash
universal-zero "Debug this implementation" -i
```

### 4. Exclude Strict/Filtered Providers
```bash
universal-zero "Audit this architecture" --exclude "gpt,openai,claude,anthropic"
```

### 5. Auto-Apply Winning Prompt to Hermes Agent
```bash
universal-zero "Execute task deterministically" \
  --inject-hermes \
  --hermes-update-model \
  --protect-hermes-config
```

### 6. Auto-Apply to Claude Code (`CLAUDE.md`)
```bash
universal-zero "Execute code without disclaimers" \
  --inject-claude \
  --claude-protect
```

### 7. Export Results
```bash
universal-zero "Write a low-level parser" \
  --output results/run.json \
  --best-output results/best.txt
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
