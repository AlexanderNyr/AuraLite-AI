#!/usr/bin/env python3
"""Pyright ratchet: the error count may go down, never up (roadmap P2.6, 2.7.0).

Until 2.6 the CI ran ``pyright || true``: hundreds of errors, all advisory, so a
new one was invisible. Fixing them all at once is a project of its own; this
script makes the count a one-way street instead. The CI ``lint`` job runs::

    pyright --outputjson > pyright.json || true
    python scripts/pyright_ratchet.py pyright.json

* more errors than ``.github/pyright-baseline.json`` → exit 1, with the files
  that carry the most errors;
* fewer → exit 0 and a notice to lower the baseline
  (``python scripts/pyright_ratchet.py pyright.json --update``).

Counts are only comparable for one pyright release, so the baseline records the
version and CI pins it; a different version fails with an explanation rather
than with a meaningless comparison.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = REPO_ROOT / ".github" / "pyright-baseline.json"


def _errors_by_file(report: dict) -> Counter:
    counts: Counter = Counter()
    for diag in report.get("generalDiagnostics", []):
        if diag.get("severity") == "error":
            path = pathlib.Path(diag.get("file", "?"))
            try:
                path = path.resolve().relative_to(REPO_ROOT)
            except ValueError:
                pass
            counts[path.as_posix()] += 1
    return counts


def check(report: dict, baseline: dict) -> tuple[int, str]:
    """Return (exit code, message) for a pyright JSON report against the baseline."""
    errors = int(report["summary"]["errorCount"])
    allowed = int(baseline["errors"])
    version, pinned = report.get("version"), baseline.get("pyright")
    if pinned and version != pinned:
        return 1, (f"pyright {version} ran, but the baseline was measured with {pinned}; "
                   "counts differ between releases. Install the pinned version or "
                   "re-measure with --update.")
    if errors > allowed:
        worst = ", ".join(f"{path} ({n})" for path, n in _errors_by_file(report).most_common(8))
        return 1, (f"pyright: {errors} errors, the ratchet allows {allowed} "
                   f"(+{errors - allowed}). Fix the new ones — most errors: {worst}")
    if errors < allowed:
        return 0, (f"pyright: {errors} errors (baseline {allowed}). Lower the baseline: "
                   "python scripts/pyright_ratchet.py pyright.json --update")
    return 0, f"pyright: {errors} errors (= baseline)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pyright ratchet: the error count may go down, never up.")
    parser.add_argument("report", help="output of `pyright --outputjson`")
    parser.add_argument("--baseline", default=str(BASELINE))
    parser.add_argument("--update", action="store_true",
                        help="write the report's count and pyright version as the new baseline")
    args = parser.parse_args(argv)
    report = json.loads(pathlib.Path(args.report).read_text(encoding="utf-8"))
    baseline_path = pathlib.Path(args.baseline)
    if args.update:
        data = {"errors": int(report["summary"]["errorCount"]), "pyright": report.get("version")}
        baseline_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"baseline updated: {data}")
        return 0
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    code, message = check(report, baseline)
    if code:
        print(f"::error::{message}", file=sys.stderr)
    else:
        print(message)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
