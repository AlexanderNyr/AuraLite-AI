#!/usr/bin/env python3
"""Version consistency check (roadmap P0.1 / P2.14).

AuraLite ships version strings in several places; a release where `pyproject.toml`
says 2.6.5 while `model_engine.__version__` still says 2.6.4 is exactly the bug
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

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10 (still supported by requires-python)
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _pyproject_version(root: pathlib.Path = REPO_ROOT) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    if tomllib is not None:
        return tomllib.loads(text)["project"]["version"]
    # Last resort without a TOML parser: the `version = "..."` line of [project].
    section = re.search(r"^\[project\][ \t]*\n(.*?)(?=^\[|\Z)", text, re.M | re.S)
    match = section and re.search(r'^version\s*=\s*"([^"]+)"', section.group(1), re.M)
    if not match:
        raise SystemExit("pyproject.toml has no [project] version")
    return match.group(1)


def _module_version(root: pathlib.Path = REPO_ROOT) -> str:
    source = (root / "model_engine" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', source, re.M)
    if not match:
        raise SystemExit("model_engine/__init__.py has no __version__ assignment")
    return match.group(1)


def _server_version(root: pathlib.Path = REPO_ROOT) -> str:
    """The server must not carry its own literal version — it imports the package one."""
    source = (root / "server" / "openai_server.py").read_text(encoding="utf-8")
    literal = re.search(r'^_AURALITE_VERSION\s*=\s*"([^"]+)"', source, re.M)
    if literal:
        raise SystemExit(
            "server/openai_server.py hardcodes _AURALITE_VERSION="
            f"{literal.group(1)!r}; import it from model_engine instead")
    return _module_version(root)


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
        "model_engine/__init__.py": _module_version(),
        "server/openai_server.py (imported)": _server_version(),
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
