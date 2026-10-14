#!/usr/bin/env python3
"""Print the requirements of AuraLite's core and selected extras (v2.6.6).

`pyproject.toml` is the single source of truth for dependencies. This script
turns it into a requirements list, which is what the Docker images install in a
layer of their own (rebuilt when pyproject.toml changes, not on every source
edit) and what `tests/test_packaging.py` compares `requirements.txt` against.

Usage:
    python scripts/print_requirements.py                 # core dependencies only
    python scripts/print_requirements.py serve hf        # core + extras
    python scripts/print_requirements.py --pyproject path/to/pyproject.toml serve

Self-references such as ``auralite-ai[hf]`` inside an extra are expanded
recursively, so the output never asks pip to install AuraLite itself.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import tomllib  # Python >= 3.11, like AuraLite itself

HERE = pathlib.Path(__file__).resolve().parent
# Next to the script (how the Dockerfile copies the two files), else the repo root.
DEFAULT_PYPROJECT = next((path for path in (HERE / "pyproject.toml",
                                            HERE.parent / "pyproject.toml") if path.exists()),
                         HERE.parent / "pyproject.toml")


def requirements(project: dict, extras: list[str]) -> list[str]:
    """Core dependencies plus `extras`, de-duplicated, self-references expanded."""
    name = re.sub(r"[-_.]+", "-", project["name"]).lower()
    optional = project.get("optional-dependencies", {})
    result: list[str] = list(project.get("dependencies", []))
    seen_extras: set[str] = set()

    def add(extra: str) -> None:
        if extra in seen_extras:
            return
        if extra not in optional:
            raise SystemExit(f"unknown extra {extra!r}; known: {sorted(optional)}")
        seen_extras.add(extra)
        for req in optional[extra]:
            match = re.match(r"^\s*([A-Za-z0-9_.-]+)\s*\[([^\]]+)\]\s*$", req)
            if match and re.sub(r"[-_.]+", "-", match.group(1)).lower() == name:
                for nested in match.group(2).split(","):
                    add(nested.strip())
            elif req not in result:
                result.append(req)

    for extra in extras:
        add(extra)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("extras", nargs="*", help="optional-dependency groups to include")
    parser.add_argument("--pyproject", type=pathlib.Path, default=DEFAULT_PYPROJECT)
    args = parser.parse_args(argv)
    with args.pyproject.open("rb") as fh:
        project = tomllib.load(fh)["project"]
    sys.stdout.write("\n".join(requirements(project, args.extras)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
