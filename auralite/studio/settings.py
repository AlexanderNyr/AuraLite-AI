"""The Studio's remembered settings: theme, recent models, last folders (2.19).

Stored as ``$AURALITE_HOME/studio.json``. Reading is forgiving - a missing or
broken file gives the defaults - because a settings file must never stop the
app from starting. Tokens are never stored: gated models use ``HF_TOKEN`` or
``huggingface-cli login``.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .hardware import auralite_home

__all__ = ["StudioSettings", "load_settings", "save_settings", "settings_path"]

MAX_RECENT = 8


def settings_path() -> Path:
    override = os.environ.get("AURALITE_STUDIO_SETTINGS")
    return Path(override).expanduser() if override else auralite_home() / "studio.json"


@dataclass
class StudioSettings:
    first_run_done: bool = False          # the wizard opens by itself until then
    dark: bool = False
    recent_models: list[str] = field(default_factory=list)
    last_data_dir: str = ""
    last_model_dir: str = ""
    geometry: str = ""                    # Qt's saveGeometry, hex
    check_updates: bool = True
    gflops: float = 0.0                   # the last speed test (0: none)
    gflops_threads: int = 0               # the threads it ran with
    last_tab: str = ""                    # 2.24: the tab open when the Studio closed
    language: str = ""                    # 2.24: the Studio's language ("": English)

    def add_recent(self, path: str) -> None:
        path = os.path.abspath(path)
        items = [p for p in self.recent_models if os.path.normcase(p) != os.path.normcase(path)]
        self.recent_models = [path] + items[:MAX_RECENT - 1]


def load_settings(path: Path | None = None) -> StudioSettings:
    path = path or settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return StudioSettings()
    if not isinstance(data, dict):
        return StudioSettings()
    known = {f.name: f for f in fields(StudioSettings)}
    values = {}
    defaults = StudioSettings()
    for name, value in data.items():
        if name not in known:
            continue
        default = getattr(defaults, name)
        if isinstance(default, bool):
            if isinstance(value, bool):
                values[name] = value
        elif isinstance(default, float):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[name] = float(value)
        elif isinstance(default, int):
            if isinstance(value, int) and not isinstance(value, bool):
                values[name] = value
        elif isinstance(default, list):
            if isinstance(value, list):
                values[name] = [str(v) for v in value if isinstance(v, str)][:MAX_RECENT]
        elif isinstance(value, str):
            values[name] = value
    return StudioSettings(**values)


def save_settings(settings: StudioSettings, path: Path | None = None) -> bool:
    """Write the settings atomically; False when the folder is not writable."""
    path = path or settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(asdict(settings), indent=1), encoding="utf-8")
        os.replace(temporary, path)
        return True
    except OSError:
        return False
