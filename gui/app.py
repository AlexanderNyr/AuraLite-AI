"""Compatibility wrapper around the historical gui_app.py AIApp."""
try:
    from gui_app import AIApp
except Exception:  # pragma: no cover
    AIApp = None
__all__ = ["AIApp"]
