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
  `auralite/engine/safe_io.py`, and old `.pt` files (numpy RNG state) are still
  supported through a scoped allowlist.
- New code goes into the `auralite` package — never a new top-level module
  (`tests/test_packaging.py` fails: it would not be in the wheel). Import with
  the package name (`from auralite.engine import …`), not the deprecated 2.6
  aliases.
- Keep the public API of `auralite/__init__.py` and `auralite/engine/__init__.py`
  intact. Until 2.8 the repository-root aliases (`model_engine.py`, `gui_app.py`,
  …) must keep resolving to the same module objects (`tests/test_compat_aliases.py`).

## Commit style

`fix:`, `feat:`, `refactor:`, `docs:`, `chore:` — short imperative subject, then
a body that explains *why* and lists verification steps (numbers help).
