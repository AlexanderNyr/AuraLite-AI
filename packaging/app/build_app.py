"""Build the AuraLite app that needs no Python (2.21): one folder, zipped for a release.

    python packaging/app/build_app.py --zip     # dist/AuraLite-Studio-<version>-<system>.zip
    python packaging/app/build_app.py --check   # only check the build environment

Run it in a virtual environment made for it - ``build_exe.bat`` does all of this on
Windows::

    python -m venv .venv-app
    .venv-app\\Scripts\\activate
    pip install -r packaging/app/requirements.txt
    pip install --no-deps .
    python packaging/app/build_app.py --zip

The steps:

1. the environment: what is installed must be requirements.txt's pins, and AuraLite must be
   installed from this very folder (``pip install --no-deps .``: same files, not editable);
2. the icon (Windows): the Studio's own drawing, 16 to 256 pixels, and the programs' file
   properties (version, name);
3. PyInstaller with auralite_app.spec: ``dist/app/AuraLite-Studio-<version>/``;
4. README.txt, auralite-app.json and the licenses of everything bundled go into that folder,
   and a path too long to unzip on Windows (260 characters) fails the build;
5. ``--zip``: the zip, with the folder inside, and its SHA-256 next to it.

packaging/app/smoke_app.py then checks the result the way a user meets it.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import zipfile
from collections.abc import Callable, Iterable
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = HERE / "auralite_app.spec"
REQUIREMENTS = HERE / "requirements.txt"
#: The longest path inside the zip, its top folder included, that still unzips everywhere:
#: Windows allows 260 characters, and a deep destination such as
#: C:\Users\<a long name>\Downloads\AuraLite-Studio-2.28.1-windows-x64\ takes about 90.
PATH_BUDGET = 160
#: The distributions the app runs with (their dependencies follow); their licenses ship.
RUNTIME_ROOTS = ("auralite-ai", "torch", "numpy", "PySide6-Essentials", "safetensors",
                 "tokenizers", "zstandard", "psutil", "pillow")      # 2.26: Pillow
#: Required by a root but left out of the app (the spec's LEFT_OUT): no license to ship.
NOT_BUNDLED = {"matplotlib", "huggingface-hub"}
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


class BuildError(RuntimeError):
    """The build cannot go on; the message says why and what to do."""


def utf8_output() -> None:
    """Print UTF-8 whatever the console or pipe (2.21.1): Python on Windows writes a pipe - CI's
    log, a `> build.log` - in the ANSI code page, so a path with Russian letters in it (a Windows
    user name) would end the build with UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):   # a stream that cannot change: leave it
                pass


def github_error(title: str, message: str) -> str:
    """The annotation a failed build leaves on a GitHub Actions run (2.21.1): its page shows it,
    and the check run's annotations keep it - readable without signing in, unlike the log."""
    def escape(text: str) -> str:
        return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    title = escape(title).replace(":", "%3A").replace(",", "%2C")
    return f"::error title={title}::{escape(message)}"


# ----------------------------------------------------------------------
#  1. The environment
# ----------------------------------------------------------------------
def canonical(name: str) -> str:
    """A distribution name as pip compares them (``PySide6_Essentials`` = ``pyside6-essentials``)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path: Path = REQUIREMENTS) -> dict[str, str]:
    """``{name: version}`` of a requirements file whose every requirement is pinned (``==``)."""
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):           # comments, --extra-index-url
            continue
        name, sep, version = line.partition("==")
        if not sep or not version.strip() or any(c in name for c in "<>=!~;[ "):
            raise BuildError(f"{path.name}: every requirement must be pinned with ==: {raw!r}")
        pins[canonical(name)] = version.strip()
    return pins


def _installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def check_pins(pins: dict[str, str],
               installed: Callable[[str], str | None] = _installed_version) -> list[str]:
    """What differs between the pins and this environment ([] when nothing)."""
    problems = []
    for name, version in pins.items():
        have = installed(name)
        if have is None:
            problems.append(f"{name} {version} is not installed")
        elif have != version:
            problems.append(f"{name} is {have}, the build pins {version}")
    return problems


def source_version(root: Path = ROOT) -> str:
    """The version in this folder's pyproject.toml."""
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    found = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not found:
        raise BuildError("pyproject.toml has no version")
    return found.group(1)


def tree_digest(folder: Path) -> str:
    """One SHA-256 over the package's .py files (paths and contents), to compare two copies."""
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(folder).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def installed_package() -> Path:
    """The folder of the ``auralite`` package this Python imports."""
    import auralite
    return Path(auralite.__file__).resolve().parent


def check_environment(root: Path = ROOT) -> str:
    """Raise :class:`BuildError` unless this environment builds this folder's app exactly;
    returns the version."""
    problems = check_pins(read_pins())
    version = source_version(root)
    have = _installed_version("auralite-ai")
    package = installed_package()
    again = "pip install --no-deps ."
    if have is None:
        problems.append(f"AuraLite is not installed: {again}")
    elif have != version:
        problems.append(f"the installed AuraLite is {have}, this folder is {version}: {again}")
    elif package == (root / "auralite").resolve():
        problems.append("AuraLite runs from this folder (an editable install, or the current "
                        f"directory); install it as a package: {again}")
    elif tree_digest(package) != tree_digest(root / "auralite"):
        problems.append(f"the installed AuraLite's files differ from this folder's: {again}")
    if problems:
        raise BuildError("This environment does not build the app:\n"
                         + "\n".join(f"  - {p}" for p in problems)
                         + f"\nInstall the build's packages: pip install -r "
                           f"{REQUIREMENTS.relative_to(root).as_posix()}")
    return version


# ----------------------------------------------------------------------
#  2. The icon and the file properties (Windows)
# ----------------------------------------------------------------------
def ico_bytes(images: Iterable[tuple[int, bytes]]) -> bytes:
    """An .ico file holding PNG `images` ``(size, png bytes)`` - Windows Vista and newer read
    PNG icons of every size."""
    images = list(images)
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, data = b"", b""
    for size, png in images:
        side = 0 if size >= 256 else size          # 0 means 256
        entries += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(png),
                               offset + len(data))
        data += png
    return header + entries + data


def draw_icon(path: Path) -> Path:
    """Write the Studio's icon as an .ico (Qt draws it; no image file is kept anywhere)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QGuiApplication

    from auralite.studio.qt.theme import icon_pixmap
    app = QGuiApplication.instance() or QGuiApplication([sys.argv[0]])
    images = []
    for size in ICON_SIZES:
        data = QByteArray()
        buffer = QBuffer(data)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        if not icon_pixmap(size).save(buffer, "PNG"):
            raise BuildError(f"Qt could not draw the {size} px icon")
        buffer.close()
        images.append((size, bytes(data.data())))
    del app
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ico_bytes(images))
    return path


def version_info(version: str) -> str:
    """The programs' file properties (Windows' version resource), in PyInstaller's format."""
    parts = [int(n) for n in re.findall(r"\d+", version)[:3]]
    numbers = tuple(parts + [0] * (4 - len(parts)))
    strings = {"CompanyName": "AuraLite", "FileDescription": "AuraLite Studio",
               "FileVersion": version, "InternalName": "AuraLite Studio",
               "LegalCopyright": "Apache-2.0, https://github.com/AlexanderNyr/AuraLite-AI",
               "OriginalFilename": "AuraLite Studio.exe", "ProductName": "AuraLite Studio",
               "ProductVersion": version}
    table = ",\n          ".join(f"StringStruct({k!r}, {v!r})" for k, v in strings.items())
    return f"""# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(filevers={numbers}, prodvers={numbers}, mask=0x3f, flags=0x0,
                    OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
          {table}])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""


# ----------------------------------------------------------------------
#  4. What goes into the folder
# ----------------------------------------------------------------------
def runtime_distributions(roots: Iterable[str] = RUNTIME_ROOTS,
                          skip: Iterable[str] = NOT_BUNDLED) -> list[metadata.Distribution]:
    """The installed distributions the app runs with: `roots` and what they require (no
    extras, markers for this system), minus `skip` and what is not installed."""
    from packaging.requirements import Requirement
    skipped = {canonical(name) for name in skip}
    found: dict[str, metadata.Distribution] = {}
    todo = [canonical(name) for name in roots]
    while todo:
        name = todo.pop()
        if name in found or name in skipped:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        found[name] = dist
        for text in dist.requires or []:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                todo.append(canonical(requirement.name))
    return [found[name] for name in sorted(found)]


def _license_name(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    expression = meta.get("License-Expression") or ""
    if expression:
        return expression
    text = (meta.get("License") or "").strip()
    if text and "\n" not in text and len(text) < 80:
        return text
    classifiers = [c.split(" :: ")[-1] for c in meta.get_all("Classifier") or []
                   if c.startswith("License ::")]
    return ", ".join(classifiers) or "see its license file"


_LICENSE_FILE = re.compile(r"(licen[cs]e|copying|notice|authors)", re.IGNORECASE)


#: The GNU texts the LGPL-3.0 asks to accompany Qt with (PySide6's wheels carry none).
GNU_TEXTS = (HERE / "licenses" / "LGPL-3.0.txt", HERE / "licenses" / "GPL-3.0.txt")


def write_licenses(folder: Path, dists: Iterable[metadata.Distribution]) -> int:
    """``licenses/<name>/...`` for every bundled distribution (its license files),
    ``licenses/Qt/`` (the LGPL-3.0 and the GNU GPL it refers to) and
    ``licenses/THIRD-PARTY-NOTICES.txt``; returns how many files were copied."""
    dists = list(dists)
    target = folder / "licenses"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    lines = ["The AuraLite app bundles these packages; each keeps its own license - the",
             "files in the folder of its name. AuraLite itself is Apache-2.0 (auralite-ai/",
             "LICENSE), the license text too of the Apache-2.0 packages below that ship none.", ""]
    copied = 0
    qt = next((d for d in dists if canonical(d.metadata["Name"] or "") == "pyside6-essentials"),
              None)
    if qt is not None:
        (target / "Qt").mkdir()
        for text in GNU_TEXTS:
            shutil.copyfile(text, target / "Qt" / text.name)
            copied += 1
        version = qt.version
        series = ".".join(version.split(".")[:2])
        lines += [f"Qt {version} and Qt for Python (PySide6, shiboken6) {version} are used under",
                  "the GNU LGPL-3.0: Qt/LGPL-3.0.txt, with the GNU GPL it refers to in",
                  "Qt/GPL-3.0.txt. Their libraries are the Qt6* and pyside6* / shiboken6* files in",
                  "_internal/PySide6 and _internal/shiboken6, loaded at run time: you may replace",
                  "them with your own build of the same version. Their source:",
                  f"  https://download.qt.io/official_releases/qt/{series}/{version}/",
                  "  https://download.qt.io/official_releases/QtForPython/",
                  "  https://code.qt.io/cgit/pyside/pyside-setup.git/", ""]
    for dist in dists:
        name = dist.metadata["Name"] or "?"
        home = dist.metadata.get("Home-page") or ""
        for entry in dist.metadata.get_all("Project-URL") or []:
            label, _, url = entry.partition(",")
            if not home and label.strip().lower() in ("homepage", "source", "repository"):
                home = url.strip()
        lines.append(f"{name} {dist.version} - {_license_name(dist)}" + (f" - {home}" if home
                                                                           else ""))
        for file in dist.files or []:
            parts = Path(str(file)).parts
            if not parts or not parts[0].endswith(".dist-info") or len(parts) < 2:
                continue
            if not (_LICENSE_FILE.search(parts[-1]) or "licenses" in parts[1:-1]):
                continue
            source = Path(str(dist.locate_file(file)))
            inside = parts[2:] if parts[1].lower() == "licenses" and len(parts) > 2 else parts[1:]
            if source.is_file():
                destination = target / canonical(name) / Path(*inside)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                copied += 1
    python_license = _python_license()
    if python_license is not None:
        shutil.copyfile(python_license, target / "Python-LICENSE.txt")
        copied += 1
        lines.append(f"Python {sys.version.split()[0]} - PSF-2.0 (Python-LICENSE.txt)")
    (target / "THIRD-PARTY-NOTICES.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return copied


def _python_license() -> Path | None:
    base = Path(sys.base_prefix)
    for candidate in (base / "LICENSE.txt",
                      base / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}"
                      / "LICENSE.txt"):
        if candidate.is_file():
            return candidate
    return None


def app_info(version: str, tag: str, *, commit: str = "", built: str = "") -> dict:
    """auralite-app.json: which build this is (read by :mod:`auralite.bundle`)."""
    def have(name: str) -> str:
        return _installed_version(name) or ""
    return {"app": "AuraLite Studio", "version": version, "platform": tag,
            "python": sys.version.split()[0], "torch": have("torch"),
            "qt": have("PySide6-Essentials"), "pyinstaller": have("pyinstaller"),
            "built": built or _dt.datetime.now(_dt.timezone.utc).date().isoformat(),
            "commit": commit}


def readme_text(info: dict, system: str = sys.platform) -> str:
    """README.txt, the first thing in the folder."""
    from auralite import bundle
    windows = system == "win32"
    studio, cli = bundle.program_name("studio", system), bundle.program_name("cli", system)
    home = r"%USERPROFILE%\.auralite" if windows else "~/.auralite"
    logs = home + ("\\logs" if windows else "/logs")
    title = f"AuraLite Studio {info['version']} for {info['platform']} - no Python needed"
    underline = "=" * len(title)
    smartscreen = ("\n  Windows may say \"Windows protected your PC\": the app is not signed. "
                   "Choose\n  \"More info\", then \"Run anyway\".\n" if windows else "")
    # 2.21.2: typed with its folder - PowerShell runs nothing from the current folder by its
    # name alone (it looks on PATH: an AuraLite from pip there, or nothing)
    run = (".\\" if windows else "./") + cli
    howto = (f'The command line is "{cli}" in this folder. Open PowerShell or cmd here (type\n'
             "powershell in Explorer's address bar, press Enter) and type it with \".\\\" in\n"
             "front - PowerShell runs nothing from this folder without it:" if windows else
             f'The command line is "{cli}" in this folder. Open a terminal here and type it\n'
             'with "./" in front:')
    return f"""{title}
{underline}

Start: double-click "{studio}" in this folder. The first start takes longer: the
system checks the new files.{smartscreen}
Keep the whole folder together. It may live anywhere you can write to (not in
Program Files); your models, runs, logs and settings are kept in {home}
(set AURALITE_HOME to keep them elsewhere), never here.

{howto}
    {run} doctor --studio              check this computer and that the Studio starts
    {run} doctor --studio --out r.txt  the same, and the report in r.txt (for an issue)
    {run} runs                         the run library
    {run} bench                        how fast this computer trains (a few minutes)
    {run} train --help                 train from the command line

Update: download the new zip from
    {bundle.DOWNLOAD_URL}
and unzip it into a new folder; delete the old folder once the new one works.
Remove: delete this folder (and {home} with your models and runs).

If it does not start: the logs are in {logs}.
Run "{run} doctor --studio" in a terminal here.
The guide: {bundle.GUIDE_URL}

Inside: AuraLite {info['version']}, Python {info['python']}, PyTorch {info['torch']} (runs on
the processor; no GPU), Qt {info['qt']} (PySide6). The licenses are in the licenses
folder; AuraLite is Apache-2.0.
"""


#: The Visual C++ Redistributable's DLLs: the app must bring every one its binaries import.
VC_RUNTIME = re.compile(r"^(vcruntime|msvcp|concrt|vcomp|vccorlib)\d", re.IGNORECASE)


def pe_imports(path: Path) -> set[str]:
    """The DLLs a Windows binary imports, delay-loaded ones too (pefile, which PyInstaller
    installs on Windows)."""
    import pefile
    pe = pefile.PE(str(path), fast_load=True)
    try:
        pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"]])
        entries = (list(getattr(pe, "DIRECTORY_ENTRY_IMPORT", []))
                   + list(getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", [])))
        return {entry.dll.decode("ascii", "replace").lower() for entry in entries}
    finally:
        pe.close()


def missing_vc_runtime(folder: Path,
                       imports_of: Callable[[Path], set[str]] = pe_imports) -> dict[str, str]:
    """The Visual C++ runtime DLLs some binary of the app imports but the app does not bring:
    ``{dll: the first binary that needs it}``. The build machine's own copies hide them - and
    without the Redistributable the app would not start (PyTorch 2.13+'s
    vcruntime140_threads.dll, which PyInstaller does not collect, is the one that did)."""
    present = {path.name.lower() for path in folder.rglob("*") if path.suffix.lower() == ".dll"}
    missing: dict[str, str] = {}
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in (".dll", ".pyd", ".exe") or not path.is_file():
            continue
        try:
            names = imports_of(path)
        except Exception:                # not a binary pefile can read: nothing to check
            continue
        for name in sorted(names):
            if VC_RUNTIME.match(name) and name not in present:
                missing.setdefault(name, path.relative_to(folder).as_posix())
    return missing


def vc_runtime_lines(folder: Path) -> list[str]:
    """The Visual C++ runtime DLLs in the app, with their versions (for the build's log)."""
    from auralite.studio.msvc_runtime import file_version
    lines = []
    for path in sorted(folder.rglob("*.dll")):
        if VC_RUNTIME.match(path.name):
            version = file_version(path)
            text = ".".join(str(n) for n in version) if version else "?"
            lines.append(f"  {path.relative_to(folder).as_posix()}  {text}")
    return lines


def too_long(folder: Path, budget: int = PATH_BUDGET) -> list[tuple[int, str]]:
    """Paths inside the zip (top folder included) longer than `budget`, longest first."""
    found = []
    for path in folder.rglob("*"):
        inside = f"{folder.name}/{path.relative_to(folder).as_posix()}"
        if len(inside) > budget:
            found.append((len(inside), inside))
    return sorted(found, reverse=True)


def folder_stats(folder: Path) -> tuple[int, int]:
    """(files, bytes) in `folder`."""
    files = size = 0
    for path in folder.rglob("*"):
        if path.is_file():
            files += 1
            size += path.stat().st_size
    return files, size


def finish_folder(folder: Path, info: dict, system: str = sys.platform) -> None:
    """README.txt, auralite-app.json and the licenses."""
    newline = "\r\n" if system == "win32" else "\n"
    (folder / "README.txt").write_text(readme_text(info, system), encoding="utf-8",
                                       newline=newline)
    (folder / "auralite-app.json").write_text(json.dumps(info, indent=1) + "\n",
                                              encoding="utf-8")
    write_licenses(folder, runtime_distributions())


# ----------------------------------------------------------------------
#  5. The zip
# ----------------------------------------------------------------------
def make_zip(folder: Path, target: Path) -> Path:
    """Zip `folder` with the folder itself at the top (what unzipping gives back)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         strict_timestamps=False) as archive:
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            name = f"{folder.name}/{path.relative_to(folder).as_posix()}"
            entry = zipfile.ZipInfo.from_file(path, name, strict_timestamps=False)
            entry.compress_type = zipfile.ZIP_DEFLATED
            with open(path, "rb") as source, \
                    archive.open(entry, "w", force_zip64=entry.file_size > 0x7FFF0000) as out:
                shutil.copyfileobj(source, out, 1 << 20)
    partial.replace(target)
    return target


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksum(path: Path) -> Path:
    """``<zip>.sha256`` in sha256sum's format (``sha256sum -c`` and ``Get-FileHash`` check it)."""
    checksum = path.with_name(path.name + ".sha256")
    # "\n" on Windows too: sha256sum on Linux would take a "\r" for part of the name
    checksum.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8", newline="\n")
    return checksum


# ----------------------------------------------------------------------
#  The build
# ----------------------------------------------------------------------
def _commit(root: Path) -> str:
    sha = os.environ.get("GITHUB_SHA", "")
    if sha:
        return sha[:7]
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=30, check=False)
    except OSError:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def build(dist: Path, *, make_archive: bool = False, clean: bool = True,
          quiet: bool = False) -> tuple[Path, Path | None]:
    """Build the app into ``dist/app/AuraLite-Studio-<version>``; with `make_archive` also the
    zip and its checksum in `dist`. Returns (folder, zip or None)."""
    from auralite import bundle
    version = check_environment()
    tag = bundle.platform_tag()
    name = f"AuraLite-Studio-{version}"
    work = ROOT / "build" / "app"
    env = dict(os.environ, AURALITE_APP_FOLDER=name)
    for variable in ("PYTHONPATH", "PYTHONHOME"):   # only the installed packages count
        env.pop(variable, None)
    if sys.platform == "win32":
        env["AURALITE_APP_ICON"] = str(_child_draw_icon(work / "auralite.ico"))
        info_file = work / "version-info.txt"
        info_file.parent.mkdir(parents=True, exist_ok=True)
        info_file.write_text(version_info(version), encoding="utf-8")
        env["AURALITE_APP_VERSION_INFO"] = str(info_file)
    command = [sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm",
               "--distpath", str(dist / "app"), "--workpath", str(work)]
    command += ["--clean"] if clean else []
    command += ["--log-level", "WARN"] if quiet else []
    print("$ " + " ".join(command), flush=True)
    work.mkdir(parents=True, exist_ok=True)
    # Not from this folder: `python -m` puts the current directory first on sys.path, and the
    # build would take AuraLite (and the *.egg-info setuptools leaves here) from the source.
    if subprocess.call(command, cwd=work, env=env) != 0:
        raise BuildError("PyInstaller failed (see above)")
    folder = dist / "app" / name
    info = app_info(version, tag, commit=_commit(ROOT))
    finish_folder(folder, info)
    if sys.platform == "win32":
        print("Visual C++ runtime in the app:\n" + "\n".join(vc_runtime_lines(folder)), flush=True)
        missing = missing_vc_runtime(folder)
        if missing:
            needed = "\n".join(f"  {dll} (imported by {who})" for dll, who in missing.items())
            raise BuildError("the app does not bring these Visual C++ runtime DLLs - without the "
                             f"Visual C++ Redistributable it would not start:\n{needed}\n"
                             "Install the latest Visual C++ Redistributable (x64) on this "
                             "computer and add them to the Visual C++ runtime list in "
                             "packaging/app/auralite_app.spec (a data file, not `binaries`: "
                             "PyInstaller drops Windows' own DLLs from those)")
    long_paths = too_long(folder)
    if long_paths:
        shown = "\n".join(f"  {n}: {p}" for n, p in long_paths[:20])
        raise BuildError(f"{len(long_paths)} paths inside the zip are longer than "
                         f"{PATH_BUDGET} characters - Windows could not unzip them into a "
                         f"deep folder:\n{shown}")
    files, size = folder_stats(folder)
    print(f"\nThe app: {folder} - {files:,} files, {size / 1e6:,.0f} MB", flush=True)
    archive = None
    if make_archive:
        archive = make_zip(folder, dist / bundle.zip_name(version, tag))
        checksum = write_checksum(archive)
        print(f"The zip: {archive} - {archive.stat().st_size / 1e6:,.0f} MB "
              f"(SHA-256 in {checksum.name})", flush=True)
    return folder, archive


def _child_draw_icon(path: Path) -> Path:
    """Draw the icon in a child process: Qt stays out of the process that runs PyInstaller."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if subprocess.call([sys.executable, str(Path(__file__).resolve()), "--draw-icon",
                        str(path)], cwd=path.parent) != 0 or not path.is_file():
        raise BuildError("could not draw the icon (see above)")
    return path


def main(argv: list[str] | None = None) -> int:
    utf8_output()
    parser = argparse.ArgumentParser(description="Build the AuraLite app that needs no Python.")
    parser.add_argument("--zip", action="store_true", help="also make the release's zip")
    parser.add_argument("--check", action="store_true",
                        help="only check that this environment builds the app")
    parser.add_argument("--dist", type=Path, default=ROOT / "dist",
                        help="where the app and the zip go (default: dist)")
    parser.add_argument("--no-clean", action="store_true",
                        help="reuse PyInstaller's cache from the last build")
    parser.add_argument("--quiet", action="store_true", help="PyInstaller's warnings only")
    parser.add_argument("--draw-icon", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.draw_icon:
            draw_icon(args.draw_icon)
            return 0
        if args.check:
            print(f"This environment builds AuraLite {check_environment()}'s app.")
            return 0
        build(args.dist.resolve(), make_archive=args.zip, clean=not args.no_clean,
              quiet=args.quiet)
    except BuildError as err:
        print(f"\nbuild_app: {err}", file=sys.stderr)
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print(github_error("build_app.py", str(err)), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
