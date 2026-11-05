"""Light and dark looks of the Studio (Fusion style, one stylesheet), 2.19."""
from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

ACCENT = "#2f6fdb"

_LIGHT = {"window": "#f4f6f9", "base": "#ffffff", "alt": "#eef1f5", "text": "#1d2430",
          "muted": "#5f6b7a", "border": "#d5dbe3", "button": "#ffffff", "card": "#ffffff"}
_DARK = {"window": "#1e2228", "base": "#262b33", "alt": "#2c323b", "text": "#e6e9ee",
         "muted": "#9aa5b4", "border": "#3a414c", "button": "#2c323b", "card": "#262b33"}


def palette_colors(dark: bool) -> dict[str, str]:
    return dict(_DARK if dark else _LIGHT, accent=ACCENT)


def stylesheet(dark: bool) -> str:
    c = palette_colors(dark)
    return f"""
QWidget {{ font-size: 10pt; }}
QLabel#hint {{ color: {c['muted']}; }}
QLabel#title {{ font-size: 17pt; font-weight: bold; }}
QLabel#heading {{ font-size: 12pt; font-weight: bold; }}
QRadioButton#goal {{ font-weight: bold; }}
QToolButton#section {{ border: none; font-weight: bold; }}
QLabel#good {{ color: #1d8a4a; font-weight: bold; }}
QLabel#bad {{ color: #c0392b; font-weight: bold; }}
QFrame#card {{ background: {c['card']}; border: 1px solid {c['border']}; border-radius: 8px; }}
QGroupBox {{ border: 1px solid {c['border']}; border-radius: 8px; margin-top: 14px;
             padding: 10px 8px 8px 8px; background: {c['card']}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
                    font-weight: bold; }}
QPushButton {{ padding: 5px 12px; border: 1px solid {c['border']}; border-radius: 5px;
               background: {c['button']}; }}
QPushButton:hover {{ border-color: {c['accent']}; }}
QPushButton:disabled {{ color: {c['muted']}; }}
QPushButton#primary {{ background: {c['accent']}; color: white; border: none;
                       font-weight: bold; padding: 6px 18px; }}
QPushButton#primary:disabled {{ background: {c['border']}; color: {c['muted']}; }}
QPushButton#choice {{ text-align: left; padding: 10px 14px; font-size: 11pt; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ padding: 8px 18px; margin-right: 2px; border: 1px solid {c['border']};
                border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px;
                background: {c['alt']}; }}
QTabBar::tab:selected {{ background: {c['base']}; font-weight: bold; }}
QProgressBar {{ border: 1px solid {c['border']}; border-radius: 5px; text-align: center;
                background: {c['base']}; min-height: 18px; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 4px; }}
QTextBrowser, QPlainTextEdit, QLineEdit, QTableWidget {{
    border: 1px solid {c['border']}; border-radius: 4px; background: {c['base']};
    padding: 2px; }}
"""


def apply_theme(app: QApplication, dark: bool) -> None:
    """Fusion style with a light or dark palette and the Studio's stylesheet."""
    style = app.style()
    if style is None or style.name().lower() != "fusion":
        app.setStyle("Fusion")          # once: replacing a style live repolishes every widget
    c = palette_colors(dark)
    palette = QPalette()
    roles = {QPalette.ColorRole.Window: c["window"], QPalette.ColorRole.WindowText: c["text"],
             QPalette.ColorRole.Base: c["base"], QPalette.ColorRole.AlternateBase: c["alt"],
             QPalette.ColorRole.Text: c["text"], QPalette.ColorRole.Button: c["button"],
             QPalette.ColorRole.ButtonText: c["text"], QPalette.ColorRole.ToolTipBase: c["base"],
             QPalette.ColorRole.ToolTipText: c["text"], QPalette.ColorRole.Highlight: c["accent"],
             QPalette.ColorRole.HighlightedText: "#ffffff",
             QPalette.ColorRole.PlaceholderText: c["muted"], QPalette.ColorRole.Link: c["accent"]}
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(c["muted"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText,
                     QColor(c["muted"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText,
                     QColor(c["muted"]))
    app.setPalette(palette)
    app.setStyleSheet(stylesheet(dark))
