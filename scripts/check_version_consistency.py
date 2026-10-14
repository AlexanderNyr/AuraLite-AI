#!/usr/bin/env python3
"""Version consistency check (roadmap P0.1 / P2.14).

AuraLite ships version strings in several places; a release where `pyproject.toml`
says 2.6.5 while `model_engine.__version__` still said 2.6.4 is exactly the bug
that started the v2.6.5 roadmap. This script is the single place that knows all of
them, is used by the `release` workflow *before* anything is published, and is
covered by `tests/test_release.py`.

Usage:
    python scripts/check_version_consistency.py           # exits 1 on mismatch
    python scripts/check_version_consistency.py 2.6.6     # also require this tag
"""
from __future__ import annotations

import pathlib
import re
import sys

import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _pyproject_version(root: pathlib.Path = REPO_ROOT) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    return tomllib.loads(text)["project"]["version"]


def _module_version(root: pathlib.Path = REPO_ROOT) -> str:
    source = (root / "auralite" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', source, re.M)
    if not match:
        raise SystemExit("auralite/__init__.py has no __version__ assignment")
    return match.group(1)


def _imports_the_package_version(root: pathlib.Path, relative: str, literal_re: str) -> str:
    """Modules that report a version must import it, never carry their own literal."""
    source = (root / relative).read_text(encoding="utf-8")
    literal = re.search(literal_re, source, re.M)
    if literal:
        raise SystemExit(
            f"{relative} hardcodes the version {literal.group(1)!r}; "
            "import it from the auralite package instead")
    return _module_version(root)


def _server_version(root: pathlib.Path = REPO_ROOT) -> str:
    return _imports_the_package_version(
        root, "auralite/server/openai_server.py", r'^_AURALITE_VERSION\s*=\s*"([^"]+)"')


def _engine_version(root: pathlib.Path = REPO_ROOT) -> str:
    return _imports_the_package_version(
        root, "auralite/engine/__init__.py", r'^__version__\s*=\s*"([^"]+)"')


def _changelog_version(root: pathlib.Path = REPO_ROOT) -> str | None:
    source = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(r"^#\s*.*?v?(\d+\.\d+\.\d+)", source, re.M)
    return match.group(1) if match else None


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    expected_tag = args[0] if args else None
    if expected_tag:
        expected_tag = expected_tag.lstrip("v")

    versions = {
        "pyproject.toml": _pyproject_version(),
        "auralite/__init__.py": _module_version(),
        "auralite/engine/__init__.py (imported)": _engine_version(),
        "auralite/server/openai_server.py (imported)": _server_version(),
    }
    changelog = _changelog_version()
    if changelog:
        versions["CHANGELOG.md (top entry)"] = changelog

    width = max(len(name) for name in versions)
    for name, value in versions.items():
        print(f"{name:<{width}}  {value}")

    unique = set(versions.values())
    problems = []
    if len(unique) != 1:
        problems.append(f"version strings disagree: {sorted(unique)}")
    if expected_tag and expected_tag not in unique:
        problems.append(f"tag {expected_tag!r} does not match the code ({sorted(unique)})")

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    print("OK: all version strings agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
