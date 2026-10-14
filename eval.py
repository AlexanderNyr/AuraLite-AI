"""Deprecated alias of :mod:`auralite.eval` — removed in AuraLite 2.9.

``python eval.py --model ckpt.pt --test test.txt`` keeps working in a source
checkout; an installed AuraLite runs it as ``python -m auralite.eval``.
Importing ``eval`` warns and returns ``auralite.eval`` itself (see
``auralite/_compat.py``).
"""
if __name__ == "__main__":
    from auralite.eval import main

    raise SystemExit(main())

from auralite._compat import alias_module  # noqa: E402

alias_module(__name__)
