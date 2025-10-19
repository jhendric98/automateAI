# Contributing

Use Python 3.13 and uv for dependency management.

Install dev deps:

```bash
uv sync --all-extras
```

Run checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/automateai
uv run pytest -q
```

Submit PRs with a clear description. Small, focused changes are preferred.


