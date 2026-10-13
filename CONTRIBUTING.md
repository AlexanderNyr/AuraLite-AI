# Contributing / Как контрибьютить

## Quick start

```bash
git clone https://github.com/AlexanderNyr/AuraLite-AI.git && cd AuraLite-AI
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cpu
pre-commit install                                # ruff + ruff-format + prettier
```

## Before opening a PR

```bash
ruff check .                                                  # must pass
ruff check --select E9,F,B --ignore E501,F401,B008,B017 .      # extended gate (blocking in CI)
pytest tests/ -q --ignore=tests/bench_generation.py            # must stay green (616+ tests)
python benchmarks/benchmark_generation.py --quick              # sanity benchmark
```

- Add a regression test for every bug fix (`tests/test_improvements_<version>.py`
  collects the roadmap tests; keep new tests next to the feature they cover).
- Prefer the shared fixtures from `tests/conftest.py` (`tiny_model`,
  `trained_engine`, `trained_ckpt_path`) over new fakes — the export bug of
  v2.6.4 slipped through precisely because tests used `DummyModel`.
- Checkpoints must stay backward compatible: load paths go through
  `model_engine/safe_io.py`, and old `.pt` files (numpy RNG state) are still
  supported through a scoped allowlist.
- Keep the public API of `model_engine/__init__.py` and `gui_app.py` intact;
  internal refactors must keep `from gui_app import AIApp` and the module-level
  names (`HAS_MATPLOTLIB`, `messagebox`, …) working.

## Commit style

`fix:`, `feat:`, `refactor:`, `docs:`, `chore:` — short imperative subject, then
a body that explains *why* and lists verification steps (numbers help).
