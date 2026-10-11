"""gui_app.py — historical entry point for the AuraLite desktop GUI.

The implementation now lives in the `gui` package (`gui/app.py` core +
`gui/tabs/*` mixins); this module stays so `python gui_app.py` and
`from gui_app import AIApp` keep working (v2.6.5).
"""
from gui.app import AIApp, main  # noqa: F401

# Backwards compatibility: the historical `gui_app` module exposed every
# module-level name (imports, flags, presets, ConsoleRedirector) and tests —
# as well as user scripts — monkeypatch them as `gui_app.<name>`. The real
# definitions now live in `gui.common`, so they are mirrored here.
from gui import common as _common  # noqa: E402

for _name in dir(_common):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_common, _name))
del _name, _common

__all__ = ["AIApp", "main"]


if __name__ == "__main__":
    main()
