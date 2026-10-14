"""Start the AuraLite desktop GUI from a source checkout: ``python gui_app.py``.

An installed AuraLite provides the ``auralite-gui`` command (``python -m
auralite.gui``); the implementation lives in ``auralite/gui/``.

Importing this module (``from gui_app import AIApp``) is deprecated since 2.7
and stops working in 2.9 — import ``auralite.gui.app`` instead. Until then it
still mirrors every module-level name of ``auralite.gui.common`` (flags,
presets, ``messagebox``…), as it has since the GUI was split up in 2.6.5.
"""
if __name__ == "__main__":
    from auralite.gui.app import main

    main()
else:
    import warnings

    from auralite._compat import REMOVED_IN

    warnings.warn(
        "`gui_app` is deprecated: import `auralite.gui.app` instead (the alias "
        f"is removed in AuraLite {REMOVED_IN}). `python gui_app.py` keeps working.",
        DeprecationWarning, stacklevel=2)

    from auralite.gui import common as _common
    from auralite.gui.app import AIApp, main  # noqa: F401

    for _name in dir(_common):
        if not _name.startswith("__"):
            globals().setdefault(_name, getattr(_common, _name))
    del _name, _common

    __all__ = ["AIApp", "main"]
