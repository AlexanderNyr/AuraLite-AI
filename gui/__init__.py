"""Tkinter GUI package shim."""
try:
    from .app import AIApp
except Exception:  # pragma: no cover
    AIApp = None
__all__ = ["AIApp"]
