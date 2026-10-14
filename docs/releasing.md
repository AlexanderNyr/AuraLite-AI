# Releasing AuraLite

A release is a git tag. Everything after the tag is done by
[`.github/workflows/release.yml`](../.github/workflows/release.yml); nothing is
built or uploaded from a laptop.

## Checklist

1. **One version everywhere.** Bump `pyproject.toml` (`[project] version`),
   `auralite/__init__.py` (`__version__`) and add the `CHANGELOG.md` heading.
   The server imports the package version, so there is no third place to edit.

   ```bash
   python scripts/check_version_consistency.py          # all strings agree?
   python scripts/check_version_consistency.py v2.8.0   # ...and match the tag you will push?
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

3. **Dry run of the release workflow** (v2.6.6). *Actions → Release → Run
   workflow* on `main`, leave `tag` empty and `dry_run` ticked (the default). The
   run executes `verify`, `build` and the Docker image build exactly as a tag
   would, and publishes nothing: no GitHub Release, no image push, no PyPI upload.
   From the command line:

   ```bash
   gh workflow run release.yml -f dry_run=true
   ```

   A real release can also be started by hand for an existing tag:
   `gh workflow run release.yml -f tag=v2.8.0 -f dry_run=false` — the jobs then
   check out that tag, not the branch the button was pressed on.

4. **Tag and push.**

   ```bash
   git tag v2.8.0
   git push origin v2.8.0
   ```

## What the workflow does

| Job | What happens | Fails when |
|---|---|---|
| `verify` | version strings vs. the tag, ruff gate, fast test suite | the tag and the code disagree, lint or tests fail |
| `build` | sdist + wheel, `twine check`, clean-venv install + smoke script | metadata is invalid, a module is not shipped, `auralite --version` fails |
| `release` | GitHub Release with the artifacts and generated notes | — (skipped on a dry run) |
| `image` | `ghcr.io/<owner>/auralite-ai:<version>` and `:latest` (CPU runtime); built but not pushed on a dry run | the Docker build fails |
| `pypi` | uploads the *same* artifacts to PyPI | only runs when enabled (below), never on a dry run |

## Enabling PyPI (one time, no token)

> **Ready since 2.7.0.** Up to 2.6 the wheel installed 16 top-level modules with
> generic names (`server`, `agent`, `gui`, `export`, `kernels`, …); Hugging Face's
> [`kernels`](https://pypi.org/project/kernels/) package owns the import name
> `kernels`, so installing both would have overwritten files. 2.7.0 ships the one
> `auralite` package and nothing else — the CI `package` job and
> `tests/test_packaging.py` fail if anything else appears at the top level.

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
  `git push --delete origin v2.8.0 && git tag -d v2.8.0`.
- PyPI never accepts the same file name twice. If a broken version reached PyPI,
  *yank* it on pypi.org and release the next patch version — do not try to
  re-upload the same number.
