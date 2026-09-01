# Contributing

## Development setup

```bash
git clone https://github.com/hakurido/universal-zero.git
cd universal-zero
uv sync --dev
```

## Quality gates

```bash
python -m unittest -v
uvx ruff check .
uvx ruff format --check .
uv run --with mypy mypy universal_zero.py
uv run --with coverage coverage run -m unittest
uv run --with coverage coverage report
```

Write a failing regression test before fixing defects. Keep endpoint tests local and deterministic. Never commit API keys, private prompts, model outputs, `.env` files, or `results/`.

## Pull requests

Describe behavior changed, tests added, and compatibility impact. Keep unrelated refactors separate.
