"""gui/config.py — persisted GUI settings and filesystem paths (v2.6.5).

Before v2.6.5 this module held a single constant (`SESSION_PATH`). The GUI had
no memory at all: window size, chosen language and the last used paths were
lost on every start.

Everything lives under ``$AURALITE_HOME`` (default ``~/.auralite``), so tests
and portable installs can point the whole thing at a temporary directory:

* ``gui_config.json``  — :class:`GuiConfig` (language, theme, geometry, …)
* ``gui_session.json`` — crash-safe session snapshot (see `gui.session`)

Reading is deliberately forgiving: a missing, empty or corrupt file yields the
defaults instead of an exception, because a broken config must never stop the
GUI from starting.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

__all__ = ["DEFAULT_CONFIG_PATH", "SESSION_PATH", "GuiConfig", "config_dir",
           "load_config", "save_config"]


def config_dir() -> Path:
    """Directory that holds the GUI's state files (``$AURALITE_HOME``)."""
    root = os.environ.get("AURALITE_HOME")
    return Path(root).expanduser() if root else Path.home() / ".auralite"


def _default_config_path() -> Path:
    override = os.environ.get("AURALITE_GUI_CONFIG")
    return Path(override).expanduser() if override else config_dir() / "gui_config.json"


def _default_session_path() -> Path:
    override = os.environ.get("AURALITE_GUI_SESSION")
    return Path(override).expanduser() if override else config_dir() / "gui_session.json"


DEFAULT_CONFIG_PATH = _default_config_path()
#: Kept under its historical name — `gui.session` imports it.
SESSION_PATH = _default_session_path()


@dataclass
class GuiConfig:
    """User-visible GUI settings, persisted between runs."""

    language: str = field(default_factory=lambda: os.environ.get("AURALITE_GUI_LANG", "en"))
    theme: str = "light"                 # "light" | "dark"
    geometry: str = ""                   # "1200x800", empty → computed from the screen
    active_tab: str = "train"
    restore_session: bool = True
    autosave_session: bool = True
    last_corpus: str = ""
    last_model: str = ""
    last_image_dir: str = ""
    attention_layer: int = 0
    plot_top_tokens: int = 20

    # ---- persistence --------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "GuiConfig":
        """Load the config, falling back to defaults for anything unusable."""
        target = Path(path) if path is not None else _default_config_path()
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: str | Path | None = None) -> Path:
        """Write the config atomically (tmp file + `replace`)."""
        target = Path(path) if path is not None else _default_config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(target)
        return target

    # ---- helpers ------------------------------------------------------
    def update(self, **changes) -> "GuiConfig":
        """Return a copy with the known fields replaced (unknown keys ignored)."""
        known = {f.name for f in fields(self)}
        return replace(self, **{k: v for k, v in changes.items() if k in known})

    def as_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None = None) -> GuiConfig:
    return GuiConfig.load(path)


def save_config(cfg: GuiConfig, path: str | Path | None = None) -> Path:
    return cfg.save(path)
