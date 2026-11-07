"""Is there a newer AuraLite on PyPI? (2.19)

One small request to ``pypi.org/pypi/auralite-ai/json`` with a short
timeout; no answer (offline, a proxy) is simply "unknown". Nothing is
downloaded or installed: the Studio shows the command.

2.21: the app that needs no Python (:mod:`auralite.bundle`) asks GitHub for the
latest release instead and names its zip for this computer - pip cannot update
the app, and a release whose zip is not built yet is not offered as an update.
"""
from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from auralite import bundle

__all__ = ["APP_UPDATE_URL", "UPDATE_URL", "UpdateInfo", "check_for_update", "version_tuple"]

UPDATE_URL = "https://pypi.org/pypi/auralite-ai/json"
APP_UPDATE_URL = f"https://api.github.com/repos/{bundle.REPOSITORY}/releases/latest"
RELEASES_URL = bundle.RELEASES_URL


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
    app: bool = False                # 2.21: checked for the app (GitHub), not for pip
    download: str = ""               # 2.21: the app's zip for this computer ("" = not there)

    @property
    def newer(self) -> bool:
        return bool(self.latest) and version_tuple(self.latest) > version_tuple(self.current)

    def message(self) -> str:
        if self.error:
            return f"Could not check for updates ({self.error})."
        if self.newer:
            if not self.app:
                how = bundle.update_hint(app=False)
            elif self.download:
                how = bundle.update_hint(self.download, app=True)
            else:
                how = (f"Its {bundle.platform_tag()} app is not on the download page yet - "
                       f"check again later: {RELEASES_URL}")
            return (f"AuraLite {self.latest} is out (you have {self.current}).\n\n"
                    f"{how}\nWhat is new: {RELEASES_URL}")
        return f"AuraLite {self.current} is the latest version."


def _fetch(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json",
                                                   "User-Agent": "auralite-studio"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _app_download(release: dict, version: str) -> str:
    """The address of this computer's app zip in a GitHub `release` ("" when not there)."""
    wanted = bundle.zip_name(version)
    for asset in release.get("assets") or []:
        if isinstance(asset, dict) and asset.get("name") == wanted:
            return str(asset.get("browser_download_url") or "")
    return ""


def check_for_update(current: str | None = None, *, timeout: float = 5.0,
                     fetch: Callable[[str, float], dict] | None = None,
                     app: bool | None = None) -> UpdateInfo:
    """The latest version on PyPI compared with `current` (this AuraLite). 2.21: for the
    app (`app`, default: when this is the app) GitHub's latest release and its zip."""
    if current is None:
        from auralite import __version__ as current
    app = bundle.packaged() if app is None else app
    info = UpdateInfo(current=str(current), app=app)
    try:
        if app:
            release = (fetch or _fetch)(APP_UPDATE_URL, timeout) or {}
            latest = str(release.get("tag_name") or "").removeprefix("v")
            info.download = _app_download(release, latest) if latest else ""
        else:
            data = (fetch or _fetch)(UPDATE_URL, timeout)
            latest = str(((data or {}).get("info") or {}).get("version") or "")
    except Exception as err:          # offline, proxy, a changed answer: just unknown
        info.error = type(err).__name__
        return info
    info.latest = latest
    return info
