#!/usr/bin/env bash
set -euo pipefail

# Set UZ_BASE_URL to any compatible endpoint, including the URL printed by `hermes proxy`.
: "${UZ_BASE_URL:=http://localhost:1234/v1}"
: "${UZ_API_KEY:=local}"
export UZ_BASE_URL UZ_API_KEY

universal-zero "Write a robust JSON Lines parser with tests" \
  --include 'qwen|deepseek|kimi|llama|mistral|gemma' \
  --attempts 3 \
  --target-successes 1 \
  --output results/run.json \
  --best-output results/best.md
