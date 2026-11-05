"""Is there a newer AuraLite on PyPI? (2.19)

One small request to ``pypi.org/pypi/auralite-ai/json`` with a short
timeout; no answer (offline, a proxy) is simply "unknown". Nothing is
downloaded or installed: the Studio shows the command.
"""
from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["UPDATE_URL", "UpdateInfo", "check_for_update", "version_tuple"]

UPDATE_URL = "https://pypi.org/pypi/auralite-ai/json"
RELEASES_URL = "https://github.com/AlexanderNyr/AuraLite-AI/releases"


def version_tuple(text: str) -> tuple[int, ...]:
    """``"2.19.0"`` -> ``(2, 19, 0)``; a pre-release (``"2.20.0rc1"``) counts as its base."""
    numbers = re.findall(r"\d+", str(text).split("+")[0].split("rc")[0].split("a")[0]
                         .split("b")[0])
    return tuple(int(n) for n in numbers[:4])


@dataclass
class UpdateInfo:
    current: str
    latest: str = ""                 # "" when unknown
    error: str = ""

    @property
    def newer(self) -> bool:
        return bool(self.latest) and version_tuple(self.latest) > version_tuple(self.current)

    def message(self) -> str:
        if self.error:
            return f"Could not check for updates ({self.error})."
        if self.newer:
            return (f"AuraLite {self.latest} is out (you have {self.current}).\n\n"
                    f'Update: pip install -U "auralite-ai[studio]"\n'
                    f"What is new: {RELEASES_URL}")
        return f"AuraLite {self.current} is the latest version."


def _fetch(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json",
                                                   "User-Agent": "auralite-studio"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_for_update(current: str | None = None, *, timeout: float = 5.0,
                     fetch: Callable[[str, float], dict] | None = None) -> UpdateInfo:
    """The latest version on PyPI compared with `current` (this AuraLite)."""
    if current is None:
        from auralite import __version__ as current
    info = UpdateInfo(current=str(current))
    try:
        data = (fetch or _fetch)(UPDATE_URL, timeout)
        latest = str(((data or {}).get("info") or {}).get("version") or "")
    except Exception as err:          # offline, proxy, a changed answer: just unknown
        info.error = type(err).__name__
        return info
    info.latest = latest
    return info
