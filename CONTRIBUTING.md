# Contributing to Faraday

Thank you for your interest in contributing to Faraday.

## Development Setup

```bash
git clone https://github.com/teerthsharma/faraday.git
cd faraday
python -m pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=faraday --cov-report=term-missing

# One file
pytest tests/test_core.py -v
```

## Code Quality

We enforce the following standards on every PR:

```bash
ruff check faraday/        # Lint
ruff format --check faraday/  # Format check
mypy faraday/             # Type check
```

Install pre-commit hooks once with `pre-commit install` — they run locally before every commit.

## Code Style

- Follow [PEP 8](https://pep8.org/)
- Add `from __future__ import annotations` to all modules
- Use `structlog` for all logging (no `print()`)
- All public APIs must have numpydoc docstrings
- Type annotations required on all function signatures
- Test every public method; aim for ≥ 80% coverage

## Adding a New Module

1. Add the module under `faraday/`
2. Export public symbols in `faraday/__init__.py`
3. Add types to `faraday/_types.py` if needed
4. Add exceptions to `faraday/exceptions.py` if needed
5. Add tests in `tests/`
6. Document in `docs/source/api.rst`

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes and add tests
4. Ensure `pytest tests/ -v` passes
5. Commit with a clear message (see commit convention below)
6. Push and open a Pull Request

### Commit Message Convention

```
type(scope): short description

Optional longer description here.

Types: feat | fix | docs | test | refactor | perf | ci
```

## Reporting Issues

Bug reports and feature requests are welcome. Please include:

- Python version, platform
- Minimal reproducible example
- Expected vs actual behavior
- `pip list` output for dependencies
