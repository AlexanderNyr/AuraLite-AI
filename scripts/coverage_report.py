#!/usr/bin/env python3
"""Print the least-covered files under a path prefix (CI `coverage` job, advisory).

Usage (after a pytest-cov run has written `.coverage`):

    python -m coverage json -q -o coverage.json
    python scripts/coverage_report.py coverage.json --prefix gui/ --limit 12

Why JSON and not coverage.xml: with several ``--cov=<package>`` options the
Cobertura report stores each file relative to *its package*
(``<source>…/gui</source>`` + ``filename="app.py"``), so a filter on ``"gui/"``
never matched and the old inline step printed nothing on every CI run. The JSON
report keys files by their path relative to the working directory.
"""
from __future__ import annotations

import argparse
import json
import sys


def least_covered(files: dict, prefix: str, limit: int) -> list[tuple[str, int, int, float]]:
    """(path, covered, statements, percent) for files under `prefix`, worst first."""
    rows = []
    for name, entry in files.items():
        path = name.replace("\\", "/")
        if not path.startswith(prefix):
            continue
        summary = entry.get("summary", {})
        total = int(summary.get("num_statements", 0))
        if total == 0:
            continue                      # empty __init__.py / shim modules
        covered = int(summary.get("covered_lines", 0))
        rows.append((path, covered, total, 100.0 * covered / total))
    rows.sort(key=lambda row: (row[3], row[0]))
    return rows[:limit]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", help="coverage JSON report (python -m coverage json)")
    parser.add_argument("--prefix", default="gui/", help="only files under this path")
    parser.add_argument("--limit", type=int, default=12, help="how many files to list")
    args = parser.parse_args(argv)

    with open(args.report, encoding="utf-8") as fh:
        files = json.load(fh).get("files", {})
    rows = least_covered(files, args.prefix, args.limit)
    if not rows:
        print(f"no measured files under {args.prefix!r} in {args.report}", file=sys.stderr)
        return 1
    print(f"least-covered files under {args.prefix}:")
    for path, covered, total, percent in rows:
        print(f"  {path:42} {covered:5}/{total:<5} {percent:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
