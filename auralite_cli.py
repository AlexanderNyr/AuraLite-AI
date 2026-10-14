"""Deprecated alias of :mod:`auralite.cli` — removed in AuraLite 3.0.

``python auralite_cli.py …`` keeps working in a source checkout; an installed
AuraLite provides the ``auralite`` command (``python -m auralite``). Importing
``auralite_cli`` warns and returns ``auralite.cli`` itself (see
``auralite/_compat.py``).
"""
if __name__ == "__main__":
    from auralite.cli import main

    raise SystemExit(main())

from auralite._compat import alias_module  # noqa: E402

alias_module(__name__)
