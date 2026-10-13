# Releasing AuraLite

A release is a git tag. Everything after the tag is done by
[`.github/workflows/release.yml`](../.github/workflows/release.yml); nothing is
built or uploaded from a laptop.

## Checklist

1. **One version everywhere.** Bump `pyproject.toml` (`[project] version`),
   `model_engine/__init__.py` (`__version__`) and add the `CHANGELOG.md` heading.
   The server imports the package version, so there is no third place to edit.

   ```bash
   python scripts/check_version_consistency.py          # all strings agree?
   python scripts/check_version_consistency.py v2.6.6   # ...and match the tag you will push?
   ```

2. **Green CI on the commit you will tag**, including the `package` job: it
   builds the sdist + wheel, installs the wheel into a clean virtualenv and runs
   [`scripts/smoke_installed_wheel.py`](../scripts/smoke_installed_wheel.py).
   This job exists because every other job uses an *editable* install, which
   sees the whole checkout: a module missing from `pyproject.toml` stays
   invisible there (that is how wheels built from the v2.6.5 development tree
   lacked `auralite_cli`, so the installed `auralite` command died with
   `ModuleNotFoundError`).

   The same check by hand:

   ```bash
   python -m build
   python -m venv /tmp/wheel-venv
   /tmp/wheel-venv/bin/python -m pip install "$(ls dist/*.whl)[serve]" \
       --extra-index-url https://download.pytorch.org/whl/cpu
   cd /tmp && /tmp/wheel-venv/bin/python ~/AuraLite-AI/scripts/smoke_installed_wheel.py
   ```

3. **Tag and push.**

   ```bash
   git tag v2.6.6
   git push origin v2.6.6
   ```

## What the workflow does

| Job | What happens | Fails when |
|---|---|---|
| `verify` | version strings vs. the tag, ruff gate, fast test suite | the tag and the code disagree, lint or tests fail |
| `build` | sdist + wheel, `twine check`, clean-venv install + smoke script | metadata is invalid, a module is not shipped, `auralite --version` fails |
| `release` | GitHub Release with the artifacts and generated notes | — |
| `image` | `ghcr.io/<owner>/auralite-ai:<version>` and `:latest` (CPU runtime) | the Docker build fails |
| `pypi` | uploads the *same* artifacts to PyPI | only runs when enabled (below) |

## Enabling PyPI (one time, no token)

Publishing uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/):
PyPI trusts this workflow's OpenID Connect identity, so there is no API token to
store, rotate or leak. Until you opt in, the `pypi` job is skipped.

1. On pypi.org: *Your account → Publishing → Add a new pending publisher*
   (a pending publisher also reserves the project name on the first upload):
   - PyPI project name: `auralite-ai`
   - Owner: `AlexanderNyr`, repository: `AuraLite-AI`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
2. On GitHub: *Settings → Environments → New environment* `pypi` (add required
   reviewers here if every upload should be approved by a person).
3. On GitHub: *Settings → Secrets and variables → Actions → Variables* →
   `PYPI_PUBLISH` = `true`.

The next pushed tag publishes. When this page was written (2026-10-12) the name
`auralite-ai` was still free on PyPI.

## When something goes wrong

- A failed job can be re-run from the Actions page; the tag stays.
- A bad tag can be removed before anything was published:
  `git push --delete origin v2.6.6 && git tag -d v2.6.6`.
- PyPI never accepts the same file name twice. If a broken version reached PyPI,
  *yank* it on pypi.org and release the next patch version — do not try to
  re-upload the same number.
