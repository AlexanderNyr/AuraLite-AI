# -*- mode: python -*-
# PyInstaller spec of the AuraLite app that needs no Python (2.21). Build it with
# packaging/app/build_app.py: that checks the pins, draws the icon and runs this, then adds
# README.txt, auralite-app.json and the licenses and makes the release's zip.
#
# One folder, two programs that share everything else (Python, PyTorch, Qt, AuraLite):
#   AuraLite Studio.exe   the Studio, no console        entry_studio.py   (auralite-studio
#   auralite.exe          the auralite command, console  entry_cli.py      elsewhere)
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

from auralite import bundle

HERE = Path(SPECPATH)  # noqa: F821 - PyInstaller defines SPECPATH
FOLDER = os.environ.get("AURALITE_APP_FOLDER") or "AuraLite-Studio"
ICON = os.environ.get("AURALITE_APP_ICON") or None
VERSION_INFO = os.environ.get("AURALITE_APP_VERSION_INFO") or None   # Windows: file properties

# What the app leaves out (docs/windows-app.md, "What the app leaves out"): the classic
# interface and the optional parts that come with pip's extras. Excluded even when the build
# environment has them installed, so every build holds the same parts.
LEFT_OUT = [
    "auralite.gui", "tkinter", "_tkinter", "matplotlib",
    "transformers", "peft", "accelerate", "bitsandbytes", "sentencepiece", "tiktoken",
    "datasets", "pyarrow", "pandas", "huggingface_hub",
    "llama_cpp", "onnx", "onnxruntime", "onnxscript", "torchao",
    "fastapi", "uvicorn", "starlette", "pydantic", "httpx", "openai",
    "sentence_transformers", "faiss", "vllm",
    # never needed when the app runs (torch.testing._internal is: torch.utils.checkpoint
    # imports it - smoke_app.py found that out)
    "triton", "torch.utils.tensorboard", "tensorboard",
    "pytest", "hypothesis", "IPython", "jupyter", "notebook",
]

# Every AuraLite module but the classic interface: commands import theirs when they run.
hiddenimports = collect_submodules("auralite", filter=lambda name: not name.startswith(
    "auralite.gui"))

# The metadata `auralite doctor`, the self-test and the libraries themselves read.
datas = []
for dist in ("auralite-ai", "torch", "numpy", "PySide6-Essentials", "shiboken6",
             "safetensors", "tokenizers", "zstandard", "psutil", "pillow"):
    try:
        datas += copy_metadata(dist)
    except Exception:              # an optional part this environment does not have
        pass


# 2.28: the interfaces' catalogues (Russian): data files of the package, which PyInstaller
# does not take by itself; the self-test names them, smoke_app.py checks that.
datas.append((str(Path(bundle.__file__).resolve().parent / "locale"), "auralite/locale"))


def license_copy(dest):
    """A license file inside a *.dist-info folder: the app has them all in its licenses
    folder already, and PyTorch's go deeper than a path Windows can unzip."""
    parts = Path(dest).parts
    return (len(parts) > 2 and parts[0].endswith((".dist-info", ".egg-info"))
            and parts[1].lower() == "licenses")


# The Visual C++ runtime PyTorch 2.13+ links that PyInstaller does not take from Windows'
# folder (the rest of it - msvcp140, vcruntime140, vcruntime140_1 - it does): the build
# machine's copy into the app's folder, where rthook_vcruntime.py loads it first. A data
# file, so that no filter for Windows' own DLLs drops it; build_app.py checks it is there.
if sys.platform == "win32":
    system32 = Path(os.environ.get("SystemRoot") or r"C:\Windows") / "System32"
    for name in ("vcruntime140_threads.dll",):
        if (system32 / name).is_file():
            datas.append((str(system32 / name), "."))

a = Analysis(  # noqa: F821 - PyInstaller's spec globals
    [str(HERE / "entry_studio.py"), str(HERE / "entry_cli.py")],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(HERE / "rthook_vcruntime.py")],
    excludes=LEFT_OUT,
    noarchive=False,
    optimize=0,                    # PyTorch reads docstrings and asserts
)
# PyInstaller also collects whole *.dist-info folders by itself, for code that asks
# importlib.metadata about a distribution (PyTorch's own code asks about PyTorch).
a.datas = [entry for entry in a.datas if not license_copy(entry[0])]
pyz = PYZ(a.pure)  # noqa: F821
# UTF-8 for what the programs print and read: a frozen Python ignores PYTHONUTF8 and
# PYTHONIOENCODING, and Windows' code page would fail on the first character outside it.
OPTIONS = [("X utf8", None, "OPTION")]


def scripts_for(entry):
    """PyInstaller's start-up scripts, and of the two programs' scripts only `entry`."""
    other = {"entry_studio", "entry_cli"} - {entry}
    return [item for item in a.scripts if item[0] not in other]


studio = EXE(  # noqa: F821
    pyz, scripts_for("entry_studio"), OPTIONS,
    exclude_binaries=True,
    name=bundle.program_name("studio").removesuffix(".exe"),
    console=False,                 # a window, no console: the Studio starts its own child
    icon=ICON,
    version=VERSION_INFO,
    upx=False,                     # packed programs look suspicious to antivirus software
)
cli = EXE(  # noqa: F821
    pyz, scripts_for("entry_cli"), OPTIONS,
    exclude_binaries=True,
    name=bundle.program_name("cli").removesuffix(".exe"),
    console=True,
    icon=ICON,
    version=VERSION_INFO,
    upx=False,
)
coll = COLLECT(studio, cli, a.binaries, a.datas, strip=False, upx=False, name=FOLDER)  # noqa: F821
