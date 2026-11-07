"""2.21: AuraLite as an app that needs no Python.

* ``auralite.bundle`` - the facts of the app (its programs, its note, the hints it gives);
* the code that asks it: the Studio's start, its errors and updates, the processor note,
  ``auralite doctor --studio``, ``auralite studio --self-test``, the CLI's errors in the app,
  the Visual C++ runtime the app brings;
* the plain AuraLiteConfig, without which ``auralite train`` could not run without pydantic;
* packaging/app/ - the build script, the spec, the entry scripts, the smoke test - and the
  CI and Release jobs that build the Windows app.

The app itself is built and checked by CI (``windows-app``) and by
packaging/app/smoke_app.py; here this process only plays the app (``sys.frozen``).
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import struct
import sys
import zipfile
from pathlib import Path

import pytest

from auralite import bundle

REPO = Path(__file__).resolve().parents[1]
APP_DIR = REPO / "packaging" / "app"
INFO = {"app": "AuraLite Studio", "version": "2.21.0", "platform": "windows-x64",
        "python": "3.13.9", "torch": "2.15.0+cpu", "qt": "6.9.3", "built": "2026-11-08",
        "commit": "abc1234"}


def _load(name: str):
    """A script of packaging/app as a module (the folder is not a package)."""
    spec = importlib.util.spec_from_file_location(f"_auralite_app_{name}", APP_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_app(tmp_path, monkeypatch):
    """This process as the app: frozen, its programs in tmp_path/app, with its note."""
    folder = tmp_path / "app"
    folder.mkdir()
    (folder / bundle.APP_INFO).write_text(json.dumps(INFO), encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(folder / bundle.program_name("studio")))
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    bundle._read_info.cache_clear()
    yield folder.resolve()
    bundle._read_info.cache_clear()


# ----------------------------------------------------------------------
#  auralite.bundle
# ----------------------------------------------------------------------
def test_outside_the_app_everything_is_as_before():
    assert not bundle.packaged() and bundle.app_folder() is None and bundle.app_info() is None
    assert bundle.studio_program() is None and bundle.cli_program() is None
    assert bundle.studio_command("--self-test") == [sys.executable, "-m", "auralite.studio",
                                                    "--self-test"]
    assert bundle.cli_command("doctor") == [sys.executable, "-m", "auralite", "doctor"]
    assert bundle.update_hint() == 'Update: pip install -U "auralite-ai[studio]"'
    assert bundle.missing_part("onnx", "export") == ['pip install "auralite-ai[export]"',
                                                     "then start AuraLite again"]
    assert bundle.missing_part("foo") == ["pip install foo", "then start AuraLite again"]


def test_program_names_platforms_and_the_zip():
    assert bundle.program_name("studio", "win32") == "AuraLite Studio.exe"
    assert bundle.program_name("cli", "win32") == "auralite.exe"
    assert bundle.program_name("studio", "linux") == "auralite-studio"
    assert bundle.program_name("cli", "darwin") == "auralite"
    with pytest.raises(ValueError):
        bundle.program_name("gui", "win32")
    assert bundle.platform_tag("win32", "AMD64") == "windows-x64"
    assert bundle.platform_tag("linux", "x86_64") == "linux-x64"
    assert bundle.platform_tag("linux", "aarch64") == "linux-arm64"
    assert bundle.platform_tag("darwin", "arm64") == "macos-arm64"
    assert bundle.zip_name("2.21.0", "windows-x64") == "AuraLite-Studio-2.21.0-windows-x64.zip"
    assert bundle.quoted(["C:\\AuraLite Studio\\auralite.exe", "doctor", ""]) == \
        '"C:\\AuraLite Studio\\auralite.exe" doctor ""'


def test_inside_the_app_the_programs_are_its_own(fake_app):
    assert bundle.packaged() and bundle.app_folder() == fake_app
    studio, cli = bundle.program_name("studio"), bundle.program_name("cli")
    assert bundle.studio_command("--studio-child") == [str(fake_app / studio), "--studio-child"]
    assert bundle.cli_command("studio", "--self-test") == [str(fake_app / cli), "studio",
                                                           "--self-test"]
    info = bundle.app_info()
    assert info is not None and info.version == "2.21.0" and info.qt == "6.9.3"
    assert info.folder == str(fake_app)
    text = bundle.describe()
    assert text == (f"the AuraLite Studio app 2.21.0 for windows-x64 (built 2026-11-08, "
                    f"abc1234), in {fake_app}")


def test_an_app_without_its_note_still_says_what_it_is(fake_app):
    from auralite import __version__
    (fake_app / bundle.APP_INFO).write_text("not json", encoding="utf-8")
    bundle._read_info.cache_clear()
    assert bundle.app_info() is None
    assert bundle.describe() == (f"a packaged AuraLite {__version__} without "
                                 f"auralite-app.json, in {fake_app}")


def test_the_app_names_no_pip_command_it_cannot_run(fake_app, tmp_path):
    hints = bundle.missing_part("onnx", "export")
    assert "does not include «onnx»" in hints[0] and 'pip install "auralite-ai[export]"' in hints[0]
    assert hints[1].endswith("docs/windows-app.md#what-the-app-leaves-out")
    hint = bundle.update_hint("https://example.org/a.zip")
    assert hint.startswith("Download the new app: https://example.org/a.zip")
    assert str(tmp_path / "home") in hint and "pip install" not in hint
    assert bundle.update_hint(app=False).startswith("Update: pip install")


# ----------------------------------------------------------------------
#  The Studio's start in the app
# ----------------------------------------------------------------------
def test_the_installation_is_the_app(fake_app):
    from auralite.studio import app
    package, stale = app.installation()
    assert package == bundle.describe() and stale == ""


def test_failure_and_crash_texts_in_the_app_name_its_programs(fake_app, tmp_path):
    from auralite.studio import app
    cli = bundle.quoted(bundle.cli_command())
    text = app.failure_text(RuntimeError("boom"))
    assert f"{cli} studio" in text and f"{cli} doctor --studio" in text
    assert "python -m" not in text
    output, faults = tmp_path / "out.log", tmp_path / "faults.log"
    for seen in ("This application failed to start because no Qt platform plugin could be "
                 "initialized.", "ImportError: DLL load failed while importing QtCore",
                 "Incompatible processor. This Qt build requires sse4.2"):
        output.write_text(seen + "\n", encoding="utf-8")
        crash = app.crash_text(0xC0000409, 1.0, output, faults)
        assert "Unzip the app again" in crash and bundle.DOWNLOAD_URL in crash, seen
        assert "pip install" not in crash and f"{cli} studio" in crash


def test_outside_the_app_the_texts_name_pip_and_python(tmp_path):
    from auralite.studio import app
    output, faults = tmp_path / "out.log", tmp_path / "faults.log"
    output.write_text("DLL load failed while importing QtCore\n", encoding="utf-8")
    crash = app.crash_text(0xC0000409, 1.0, output, faults)
    assert "-m pip install -U" in crash and "python -m auralite.studio" in crash
    assert "update Qt for Python" in crash
    assert "python -m auralite.studio" in app.failure_text(RuntimeError("boom"))


def test_an_app_without_qt_says_it_is_incomplete(fake_app, monkeypatch):
    from auralite.studio import app
    monkeypatch.setattr(app, "_qt_version", lambda: "")
    monkeypatch.setitem(sys.modules, "PySide6", None)          # import PySide6 fails
    problem = app.qt_problem()
    assert problem.startswith("This copy of AuraLite Studio has no Qt")
    assert "Unzip the app again" in problem and "pip" not in problem


def test_the_self_test_leaves_its_report_in_the_logs(tmp_path, monkeypatch, capsys):
    from auralite.studio import app
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path))
    monkeypatch.setattr(app, "qt_problem", lambda: "AuraLite Studio needs PySide6 (Qt 6).")
    assert app.self_test() == 2
    report = tmp_path / "logs" / "studio-self-test.txt"
    assert app.report_path() == report
    text = report.read_text(encoding="utf-8")
    assert text == capsys.readouterr().out                      # the same, word for word
    assert text.startswith("AuraLite Studio self-test: AuraLite ")
    assert text.rstrip().endswith("RESULT: AuraLite Studio cannot start here (see above)")

    def broken() -> tuple[str, str]:
        raise RuntimeError("the metadata is broken")

    monkeypatch.setattr(app, "installation", broken)
    assert app.self_test() == 1
    text = report.read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in text and "the metadata is broken" in text
    assert text.rstrip().endswith("RESULT: FAILED")


def test_without_a_console_the_app_watches_its_own_program(fake_app):
    from auralite.studio import app
    assert app._child_command(["--x"]) == [sys.executable, app.CHILD, "--x"]
    assert app._child_command([])[0] == str(fake_app / bundle.program_name("studio"))


# ----------------------------------------------------------------------
#  Errors, updates, the processor and the C++ runtime in the app
# ----------------------------------------------------------------------
def test_a_missing_package_in_the_app_is_one_it_leaves_out(fake_app):
    from auralite.studio.errors import explain_error
    err = ModuleNotFoundError("No module named 'onnx'", name="onnx")
    friendly = explain_error(err)
    assert friendly.kind == "missing_package" and "does not include «onnx»" in friendly.hints[0]


def test_a_missing_package_outside_the_app_is_a_pip_command():
    from auralite.studio.errors import explain_error
    friendly = explain_error(ModuleNotFoundError("No module named 'onnx'", name="onnx"))
    assert friendly.hints == ['pip install "auralite-ai[export]"', "then start AuraLite again"]


def test_a_library_that_does_not_load_in_the_app(fake_app):
    from auralite.studio.errors import explain_error
    err = OSError("[WinError 1114] A dynamic link library (DLL) initialization routine "
                  "failed. Error loading c10.dll")
    err.winerror = 1114                                         # type: ignore[attr-defined]
    friendly = explain_error(err)
    assert friendly.kind == "library"
    assert any("Unzip the app again" in hint for hint in friendly.hints)
    assert not any("pip install" in hint for hint in friendly.hints)


def _release(version: str, *names: str) -> dict:
    return {"tag_name": f"v{version}",
            "assets": [{"name": n, "browser_download_url": f"https://dl.example/{n}"}
                       for n in names]}


def test_the_app_updates_from_the_release_page():
    from auralite.studio.updates import APP_UPDATE_URL, check_for_update
    asked: list[str] = []
    wanted = bundle.zip_name("2.22.0")

    def fetch(url: str, timeout: float) -> dict:
        asked.append(url)
        return _release("2.22.0", "auralite_ai-2.22.0-py3-none-any.whl", wanted)

    info = check_for_update("2.21.0", fetch=fetch, app=True)
    assert asked == [APP_UPDATE_URL] and info.newer and info.app
    assert info.download == f"https://dl.example/{wanted}"
    message = info.message()
    assert "AuraLite 2.22.0 is out (you have 2.21.0)" in message
    assert f"Download the new app: https://dl.example/{wanted}" in message
    assert "pip" not in message


def test_a_release_without_this_computers_app_is_not_offered_as_one():
    from auralite.studio.updates import check_for_update
    info = check_for_update("2.21.0", app=True,
                            fetch=lambda url, t: _release("2.22.0", "something-else.zip"))
    assert info.newer and info.download == ""
    assert "app is not on the download page yet" in info.message()
    same = check_for_update("2.21.0", app=True, fetch=lambda url, t: _release("2.21.0"))
    assert not same.newer and "latest version" in same.message()


def test_outside_the_app_updates_still_come_from_pypi():
    from auralite.studio.updates import UPDATE_URL, check_for_update
    asked: list[str] = []

    def fetch(url: str, timeout: float) -> dict:
        asked.append(url)
        return {"info": {"version": "2.22.0"}}

    info = check_for_update("2.21.0", fetch=fetch)
    assert asked == [UPDATE_URL] and not info.app
    assert 'Update: pip install -U "auralite-ai[studio]"' in info.message()


def test_the_processor_note_in_the_app_names_its_own_qt(fake_app, monkeypatch):
    from auralite.studio import hardware
    monkeypatch.setattr(hardware, "_simd", lambda cpuinfo: ("SSE2", False))
    notes = hardware.detect_hardware().notes
    assert any("the app's Qt 6.9 runs here" in note for note in notes)
    assert not any("pip install" in note for note in notes)


def _fake_dll(path: Path, version: tuple[int, int, int, int]) -> None:
    major, minor, build, private = version
    path.write_bytes(b"MZ" + b"\0" * 64 + struct.pack("<IIII", 0xFEEF04BD, 0x00010000,
                                                    (major << 16) | minor,
                                                    (build << 16) | private))


def test_the_runtime_copies_include_the_apps_own(fake_app, tmp_path, monkeypatch):
    from auralite.studio import msvc_runtime
    internal, system = fake_app / "_internal", tmp_path / "System32"
    internal.mkdir()
    system.mkdir()
    _fake_dll(internal / "msvcp140.dll", (14, 99, 1, 0))        # newer than any real one
    _fake_dll(system / "msvcp140.dll", (14, 29, 30133, 0))
    monkeypatch.setattr(sys, "_MEIPASS", str(internal), raising=False)
    copies = msvc_runtime.runtime_copies(system)
    origins = {copy.origin: copy.version for copy in copies}
    assert origins["app"] == (14, 99, 1, 0) and origins["system"] == (14, 29, 30133, 0)
    best = msvc_runtime.choose_runtime(copies)
    assert best is not None and best.origin == "app"            # the newest is loaded first


def test_the_app_runs_on_its_own_runtime_when_it_is_new_enough():
    from auralite.studio.msvc_runtime import RuntimeCopy, choose_runtime, loaded_module
    system = RuntimeCopy("C:/Windows/System32/msvcp140.dll", (14, 50, 1, 0), "system")
    app = RuntimeCopy("C:/App/_internal/msvcp140.dll", (14, 44, 35211, 0), "app")
    old_app = RuntimeCopy("C:/App/_internal/msvcp140.dll", (14, 39, 1, 0), "app")
    assert choose_runtime([system, app]) is app                 # the one its build checked
    assert choose_runtime([system, old_app]) is system          # too old for PyTorch: newest
    assert choose_runtime([system, app], loaded="C:/x/msvcp140.dll") is None
    if sys.platform != "win32":
        assert loaded_module("vcruntime140_threads.dll") == ""


# ----------------------------------------------------------------------
#  The CLI in the app
# ----------------------------------------------------------------------
def test_doctor_checks_the_studio_with_the_apps_console_program(fake_app):
    from auralite import cli
    assert cli._studio_check_command() == [str(fake_app / bundle.program_name("cli")),
                                           "studio", "--self-test"]


def test_doctor_checks_the_studio_with_python_outside_the_app():
    from auralite import cli
    assert cli._studio_check_command() == [sys.executable, "-m", "auralite.studio",
                                           "--self-test"]


def test_auralite_studio_runs_the_self_test(monkeypatch):
    from auralite import cli
    from auralite.studio import app
    seen: list[list[str]] = []
    monkeypatch.setattr(app, "main", lambda argv=None: seen.append(list(argv or [])) or 0)
    assert cli.main(["studio", "--self-test"]) == 0 and seen[-1] == [app.SELF_TEST]
    assert cli.main(["studio"]) == 0 and seen[-1] == []


def _needs_onnx(args) -> int:
    raise ModuleNotFoundError("No module named 'onnx'", name="onnx")


def test_a_part_the_app_leaves_out_is_named_not_a_traceback(fake_app, monkeypatch, capsys):
    from auralite import cli
    monkeypatch.setattr(cli, "cmd_runs", _needs_onnx)       # build_parser runs in main()
    assert cli.main(["runs"]) == 2
    err = capsys.readouterr().err
    assert "No module named 'onnx'" in err and "does not include «onnx»" in err


def test_outside_the_app_a_missing_module_still_raises(monkeypatch):
    from auralite import cli
    monkeypatch.setattr(cli, "cmd_runs", _needs_onnx)
    with pytest.raises(ModuleNotFoundError):
        cli.main(["runs"])


def test_serve_in_the_app_says_the_server_is_left_out(fake_app, monkeypatch, capsys):
    from auralite import cli
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    assert cli.main(["serve", "--model", "m.pt"]) == 2
    err = capsys.readouterr().err
    assert "the server needs FastAPI and uvicorn" in err and "«uvicorn»" in err


def test_doctor_names_the_app(fake_app, capsys):
    from auralite import cli
    assert cli.main(["doctor", "--no-speed"]) == 0
    out = capsys.readouterr().out
    assert f"App: {bundle.describe()}" in out
    assert (f"Or in the desktop app: {bundle.program_name('studio')} in {fake_app}"
            in out)


# ----------------------------------------------------------------------
#  AuraLiteConfig without pydantic (auralite train could not run without it)
# ----------------------------------------------------------------------
def test_the_plain_config_keeps_other_keys_like_pydantic():
    from auralite.engine.config import _PlainConfig
    config = _PlainConfig(d_model=32, seq_length=16, epochs=1)
    assert config.d_model == 32 and config.n_heads == 4 and config.max_seq_len == 4096
    assert config.seq_length == 16 and config.extra == {"seq_length": 16, "epochs": 1}
    assert "seq_length" in repr(config)
    assert _PlainConfig(n_heads=6, n_kv_heads=3).n_kv_heads == 3


@pytest.mark.parametrize("values", [{"d_model": 0}, {"n_layers": -1}, {"d_ff": 2.5},
                                    {"dropout": 1.5}, {"dropout": -0.1}, {"dropout": "x"},
                                    {"vocab_size": -1}, {"seed": -3}, {"d_model": True},
                                    {"n_heads": 6, "n_kv_heads": 4},
                                    {"n_heads": 6, "n_kv_heads": 0}])
def test_the_plain_config_refuses_what_pydantic_refuses(values):
    from auralite.engine.config import _PlainConfig
    with pytest.raises(ValueError):
        _PlainConfig(**values)


def test_auralite_train_runs_without_pydantic(tmp_path, monkeypatch, capsys):
    from auralite import cli
    from auralite.engine import config
    monkeypatch.setattr(config, "AuraLiteConfig", config._PlainConfig)
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    data, params, model = tmp_path / "t.txt", tmp_path / "p.json", tmp_path / "m.pt"
    data.write_text("The quick brown fox jumps over the lazy dog. 0123456789\n" * 60,
                    encoding="utf-8")
    params.write_text(json.dumps({"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64,
                                  "seq_length": 16, "batch_size": 8, "epochs": 1,
                                  "bpe_vocab_size": 120}), encoding="utf-8")
    code = cli.main(["train", "--data", str(data), "--params", str(params), "--model",
                     str(model), "--quiet", "--no-record"])
    assert code == 0, capsys.readouterr()
    assert model.is_file() and "saved:" in capsys.readouterr().out


# ----------------------------------------------------------------------
#  The Studio's window in the app (Qt)
# ----------------------------------------------------------------------
def _qt():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from tests._integration import require
    widgets = require("PySide6.QtWidgets")
    return widgets.QApplication.instance() or widgets.QApplication([])


def _window(monkeypatch, tmp_path):
    from auralite.studio.qt.window import MainWindow
    from auralite.studio.settings import StudioSettings
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    return MainWindow(settings=StudioSettings(first_run_done=True), detect=False,
                      capture_streams=False, save_on_close=False, auto_speed_test=False)


def _close(window) -> None:
    from PySide6.QtCore import QCoreApplication, QEvent
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _actions(window) -> list[str]:
    texts: list[str] = []
    for top in window.menuBar().actions():
        menu = top.menu()
        if menu is not None:
            texts += [action.text() for action in menu.actions()]
    return texts


def test_the_app_has_no_classic_interface_to_open(monkeypatch, tmp_path):
    _qt()
    window = _window(monkeypatch, tmp_path)
    try:
        assert "Classic interface (tkinter)" in _actions(window)
    finally:
        _close(window)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    window = _window(monkeypatch, tmp_path)
    try:
        assert "Classic interface (tkinter)" not in _actions(window)
    finally:
        _close(window)


def test_the_icon_is_drawn_at_every_size():
    _qt()
    from auralite.studio.qt.theme import icon_pixmap
    from auralite.studio.qt.window import studio_icon
    for size in (16, 48, 256):
        pixmap = icon_pixmap(size)
        assert pixmap.width() == size and pixmap.height() == size and not pixmap.isNull()
    sizes = {s.width() for s in studio_icon().availableSizes()}
    assert {16, 32, 256} <= sizes


def test_a_smoke_start_closes_the_window_and_what_it_opened(monkeypatch, tmp_path):
    app = _qt()
    from PySide6.QtWidgets import QDialog

    from auralite.studio.qt import window as window_module
    assert window_module._smoke_seconds() == 0.0
    for text, seconds in (("8", 8.0), ("0.5", 0.5), ("x", 0.0), ("-3", 0.0), ("", 0.0)):
        monkeypatch.setenv(window_module.SMOKE_ENV, text)
        assert window_module._smoke_seconds() == seconds, text
    assert window_module.SMOKE_ENV == _load("smoke_app").SMOKE_ENV
    window = _window(monkeypatch, tmp_path)
    window.show()
    dialog = QDialog(window)                     # the setup wizard of a first start
    dialog.show()
    try:
        window_module._smoke_close(window)
        app.processEvents()
        assert not dialog.isVisible() and not window.isVisible()
        assert dialog.result() == QDialog.DialogCode.Rejected
    finally:
        _close(window)                           # its quit timer goes with it


# ----------------------------------------------------------------------
#  packaging/app: the build script
# ----------------------------------------------------------------------
def test_the_build_pins_every_package_and_the_qt_the_studio_needs():
    build = _load("build_app")
    pins = build.read_pins()
    from auralite.studio.app import QT_MINIMUM
    assert pins["pyside6-essentials"] == QT_MINIMUM == "6.9.3"   # no SSE4.2 needed
    assert pins["shiboken6"] == pins["pyside6-essentials"]
    assert pins["torch"].endswith("+cpu")
    assert {"numpy", "safetensors", "tokenizers", "zstandard", "psutil", "pyinstaller",
            "pyinstaller-hooks-contrib"} <= set(pins)
    text = build.REQUIREMENTS.read_text(encoding="utf-8")
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in text
    assert not set(pins) & {"matplotlib", "pydantic", "fastapi", "transformers"}


def test_unpinned_requirements_and_a_different_environment_are_refused(tmp_path):
    build = _load("build_app")
    loose = tmp_path / "requirements.txt"
    loose.write_text("# x\n--extra-index-url https://example.org\ntorch>=2.1\n",
                     encoding="utf-8")
    with pytest.raises(build.BuildError, match="pinned"):
        build.read_pins(loose)
    installed = {"torch": "2.15.0+cpu", "numpy": "2.4.0"}
    problems = build.check_pins({"torch": "2.15.0+cpu", "numpy": "2.5.4", "psutil": "7.2.2"},
                                installed.get)
    assert problems == ["numpy is 2.4.0, the build pins 2.5.4", "psutil 7.2.2 is not installed"]
    assert build.canonical("PySide6_Essentials") == "pyside6-essentials"


def test_this_test_environment_does_not_build_the_app(capsys):
    build = _load("build_app")
    assert build.source_version() == __import__("auralite").__version__
    if not build.check_pins(build.read_pins()):
        pytest.skip("this environment has the build's packages")
    assert build.main(["--check"]) == 1
    err = capsys.readouterr().err
    assert "does not build the app" in err and "pip install -r packaging/app/requirements.txt" \
        in err


def test_the_tree_digest_sees_a_changed_file(tmp_path):
    build = _load("build_app")
    (tmp_path / "pkg" / "__pycache__").mkdir(parents=True)
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n", encoding="utf-8")
    first = build.tree_digest(tmp_path / "pkg")
    (tmp_path / "pkg" / "__pycache__" / "a.cpython-313.py").write_text("y", encoding="utf-8")
    assert build.tree_digest(tmp_path / "pkg") == first
    (tmp_path / "pkg" / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert build.tree_digest(tmp_path / "pkg") != first


def test_the_icon_file_holds_png_images_of_every_size():
    build = _load("build_app")
    images = [(16, b"\x89PNG-16"), (256, b"\x89PNG-256-bytes")]
    data = build.ico_bytes(images)
    reserved, kind, count = struct.unpack_from("<HHH", data)
    assert (reserved, kind, count) == (0, 1, 2)
    for index, (size, png) in enumerate(images):
        width, height, _c, _r, planes, bits, length, offset = struct.unpack_from(
            "<BBBBHHII", data, 6 + 16 * index)
        assert width == height == (0 if size == 256 else size) and (planes, bits) == (1, 32)
        assert data[offset:offset + length] == png


def test_the_file_properties_are_pyinstallers_version_resource():
    build = _load("build_app")
    text = build.version_info("2.21.0")
    seen: dict = {}

    def record(name):
        def make(*args, **kwargs):
            seen.setdefault(name, []).append((args, kwargs))
            return name
        return make

    names = ("VSVersionInfo", "FixedFileInfo", "StringFileInfo", "StringTable",
             "StringStruct", "VarFileInfo", "VarStruct")
    eval(compile(text, "version-info.txt", "eval"), {n: record(n) for n in names})  # noqa: S307
    assert seen["FixedFileInfo"][0][1]["filevers"] == (2, 21, 0, 0)
    strings = {args[0]: args[1] for args, _kw in seen["StringStruct"]}
    assert strings["ProductVersion"] == "2.21.0" and strings["ProductName"] == "AuraLite Studio"


def test_the_readme_says_how_to_start_update_and_remove():
    build = _load("build_app")
    info = dict(INFO)
    windows = build.readme_text(info, "win32")
    assert '"AuraLite Studio.exe"' in windows and "auralite.exe doctor --studio" in windows
    assert "Run anyway" in windows and r"%USERPROFILE%\.auralite\logs" in windows
    assert bundle.DOWNLOAD_URL in windows and "Qt 6.9.3" in windows
    linux = build.readme_text(info, "linux")
    assert '"auralite-studio"' in linux and "~/.auralite/logs" in linux
    assert "Run anyway" not in linux


def test_the_build_finds_a_visual_cpp_runtime_dll_the_app_does_not_bring(tmp_path):
    build = _load("build_app")
    folder = tmp_path / "AuraLite-Studio-2.21.0"
    (folder / "_internal" / "torch" / "lib").mkdir(parents=True)
    for name in ("_internal/torch/lib/torch_cpu.dll", "_internal/torch/lib/c10.dll",
                 "_internal/msvcp140.dll", "_internal/vcruntime140.dll", "auralite.exe",
                 "_internal/notes.txt"):
        (folder / name).write_bytes(b"MZ")
    imports = {"torch_cpu.dll": {"kernel32.dll", "vcruntime140.dll", "vcruntime140_threads.dll",
                                 "api-ms-win-crt-runtime-l1-1-0.dll"},
               "c10.dll": {"msvcp140.dll", "vcruntime140_1.dll"}, "msvcp140.dll": set(),
               "vcruntime140.dll": set(), "auralite.exe": {"kernel32.dll"}}

    def imports_of(path):
        if path.suffix == ".txt":
            raise AssertionError("only binaries are read")
        return imports[path.name]

    missing = build.missing_vc_runtime(folder, imports_of)
    assert missing == {"vcruntime140_1.dll": "_internal/torch/lib/c10.dll",
                       "vcruntime140_threads.dll": "_internal/torch/lib/torch_cpu.dll"}
    for name in ("vcruntime140_1.dll", "vcruntime140_threads.dll"):
        (folder / "_internal" / name).write_bytes(b"MZ")
    assert build.missing_vc_runtime(folder, imports_of) == {}
    assert build.VC_RUNTIME.match("MSVCP140_ATOMIC_WAIT.dll") and not build.VC_RUNTIME.match(
        "ucrtbase.dll")

    def unreadable(path):
        raise ValueError("not a PE file")

    assert build.missing_vc_runtime(folder, unreadable) == {}
    _fake_dll(folder / "_internal" / "msvcp140.dll", (14, 44, 35211, 0))
    lines = build.vc_runtime_lines(folder)
    assert "  _internal/msvcp140.dll  14.44.35211.0" in lines and len(lines) == 4


def test_paths_too_long_for_windows_are_found(tmp_path):
    build = _load("build_app")
    folder = tmp_path / "AuraLite-Studio-2.21.0"
    deep = folder / "_internal" / ("x" * 60) / ("y" * 60)
    deep.mkdir(parents=True)
    (deep / "file.txt").write_text("z", encoding="utf-8")
    found = build.too_long(folder)
    assert found and found[0][1].endswith("/file.txt") and found[0][0] > build.PATH_BUDGET
    assert build.too_long(folder, budget=1000) == []
    assert build.folder_stats(folder) == (1, 1)


def test_the_zip_holds_the_folder_and_its_checksum_unzips_anywhere(tmp_path):
    build = _load("build_app")
    smoke = _load("smoke_app")
    folder = tmp_path / "AuraLite-Studio-2.21.0"
    (folder / "_internal").mkdir(parents=True)
    (folder / "auralite-app.json").write_text(json.dumps(INFO), encoding="utf-8")
    program = folder / "auralite"
    program.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    program.chmod(0o755)
    (folder / "_internal" / "lib.bin").write_bytes(os.urandom(4096))
    archive = build.make_zip(folder, tmp_path / "out" / "app.zip")
    with zipfile.ZipFile(archive) as zf:
        assert sorted(zf.namelist()) == ["AuraLite-Studio-2.21.0/_internal/lib.bin",
                                         "AuraLite-Studio-2.21.0/auralite",
                                         "AuraLite-Studio-2.21.0/auralite-app.json"]
    checksum = build.write_checksum(archive)
    raw = checksum.read_bytes()
    assert raw.endswith(b"  app.zip\n") and b"\r" not in raw and len(raw.split()[0]) == 64
    assert raw.split()[0].decode() == build.sha256_file(archive)
    unzipped = smoke.unzip(archive, tmp_path / smoke.ODD_FOLDER)
    assert unzipped == tmp_path / smoke.ODD_FOLDER / "AuraLite-Studio-2.21.0"
    assert (unzipped / "_internal" / "lib.bin").read_bytes() == \
        (folder / "_internal" / "lib.bin").read_bytes()
    if os.name == "posix":
        assert os.access(unzipped / "auralite", os.X_OK)        # the program stays runnable


def test_a_zip_with_more_than_one_folder_is_refused(tmp_path):
    smoke = _load("smoke_app")
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a/x.txt", "1")
        zf.writestr("b/y.txt", "2")
    with pytest.raises(SystemExit, match="one folder"):
        smoke.unzip(archive, tmp_path / "out")


def test_the_licenses_of_what_the_app_bundles(tmp_path):
    build = _load("build_app")
    dists = build.runtime_distributions(roots=("pytest",), skip=("iniconfig",))
    names = {build.canonical(d.metadata["Name"]) for d in dists}
    assert "pytest" in names and "pluggy" in names and "iniconfig" not in names
    folder = tmp_path / "app"
    folder.mkdir()
    copied = build.write_licenses(folder, dists)
    notices = (folder / "licenses" / "THIRD-PARTY-NOTICES.txt").read_text(encoding="utf-8")
    assert "Apache-2.0" in notices and "LGPL" not in notices      # no Qt among these
    assert any(line.startswith("pytest ") for line in notices.splitlines())
    assert copied >= 1 and (folder / "licenses" / "pytest").is_dir()
    assert not (folder / "licenses" / "Qt").exists()


def test_qt_comes_with_the_lgpl_the_gpl_and_where_its_source_is(tmp_path):
    """PySide6's wheels carry no license file; the LGPL-3.0 asks for its text and the GNU
    GPL's to go with the libraries, and for where their source is."""
    from importlib import metadata
    build = _load("build_app")
    info = tmp_path / "pyside6_essentials-6.9.3.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: PySide6_Essentials\n"
                                   "Version: 6.9.3\nLicense-Expression: LGPL-3.0-only OR "
                                   "GPL-2.0-only OR GPL-3.0-only\n", encoding="utf-8")
    folder = tmp_path / "app"
    folder.mkdir()
    build.write_licenses(folder, [metadata.PathDistribution(info)])
    qt = folder / "licenses" / "Qt"
    lgpl = (qt / "LGPL-3.0.txt").read_text(encoding="utf-8")
    gpl = (qt / "GPL-3.0.txt").read_text(encoding="utf-8")
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in lgpl and "Version 3, 29 June 2007" in lgpl
    assert "GNU GENERAL PUBLIC LICENSE" in gpl and "TERMS AND CONDITIONS" in gpl
    notices = (folder / "licenses" / "THIRD-PARTY-NOTICES.txt").read_text(encoding="utf-8")
    assert "Qt 6.9.3 and Qt for Python (PySide6, shiboken6) 6.9.3" in notices
    assert "https://download.qt.io/official_releases/qt/6.9/6.9.3/" in notices
    assert "PySide6_Essentials 6.9.3 - LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only" in notices


def test_the_apps_note_says_which_build_it_is():
    build = _load("build_app")
    info = build.app_info("2.21.0", "windows-x64", commit="abc1234", built="2026-11-08")
    assert info["version"] == "2.21.0" and info["platform"] == "windows-x64"
    assert info["commit"] == "abc1234" and info["built"] == "2026-11-08"
    assert info["python"] == sys.version.split()[0]
    assert bundle.AppInfo.from_dict(info).version == "2.21.0"


# ----------------------------------------------------------------------
#  packaging/app: the spec, the entry scripts, the smoke test
# ----------------------------------------------------------------------
def _spec() -> str:
    return (APP_DIR / "auralite_app.spec").read_text(encoding="utf-8")


def test_the_spec_builds_two_programs_from_the_entry_scripts():
    text = _spec()
    ast.parse(text)                                             # valid Python
    assert '"entry_studio.py"' in text and '"entry_cli.py"' in text
    assert (APP_DIR / "entry_studio.py").is_file() and (APP_DIR / "entry_cli.py").is_file()
    tree = ast.parse(text)
    calls = {node.func.id: node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert {"Analysis", "PYZ", "EXE", "COLLECT"} <= set(calls)
    exes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "EXE"]
    consoles = sorted(next(k.value.value for k in exe.keywords if k.arg == "console")
                      for exe in exes)
    assert consoles == [False, True]                            # the Studio has no console
    for exe in exes:
        assert next(k.value.value for k in exe.keywords if k.arg == "upx") is False
    assert '("X utf8", None, "OPTION")' in text                 # UTF-8 whatever the code page


def test_the_spec_leaves_out_what_the_docs_say_it_leaves_out():
    text = _spec()
    tree = ast.parse(text)
    left_out = next(ast.literal_eval(node.value) for node in ast.walk(tree)
                    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "")
                    == "LEFT_OUT")
    assert {"auralite.gui", "tkinter", "matplotlib", "transformers", "fastapi", "onnx",
            "datasets", "pyarrow"} <= set(left_out)
    assert "torch.testing._internal" not in left_out            # torch.utils.checkpoint needs it
    for name in bundle.INCLUDED:
        assert name not in left_out
    guide = (REPO / "docs" / "windows-app.md").read_text(encoding="utf-8")
    for part in ("classic interface", "transformers", "server", "ONNX", "Parquet"):
        assert part in guide, part


def test_the_app_brings_and_loads_pytorchs_threads_runtime_first():
    text = _spec()
    assert 'runtime_hooks=[str(HERE / "rthook_vcruntime.py")]' in text
    assert '"vcruntime140_threads.dll"' in text and 'datas.append((str(system32 / name), "."))' \
        in text
    hook = (APP_DIR / "rthook_vcruntime.py").read_text(encoding="utf-8")
    ast.parse(hook)
    assert 'sys.platform == "win32" and getattr(sys, "frozen", False)' in hook
    assert 'os.path.join(getattr(sys, "_MEIPASS", ""), _name)' in hook
    assert "ctypes.WinDLL(_path)" in hook                      # by its full path


def test_the_entry_scripts_turn_worker_processes_into_workers_first():
    for name, main in (("entry_studio.py", "auralite.studio.app"), ("entry_cli.py", "auralite.cli")):
        text = (APP_DIR / name).read_text(encoding="utf-8")
        assert text.index("multiprocessing.freeze_support()") < text.index(f"from {main} import")


def test_the_smoke_test_checks_what_a_user_meets():
    smoke = _load("smoke_app")
    assert smoke.STEPS == ("version", "doctor", "selftest", "start", "train", "workers")
    assert smoke.SMOKE_ENV == "AURALITE_STUDIO_SMOKE_SECONDS"
    assert " " in smoke.ODD_FOLDER and not smoke.ODD_FOLDER.isascii()
    for name in smoke.STEPS:
        assert callable(getattr(smoke.Smoke, f"step_{name}"))


def test_the_smoke_tests_double_click_hands_the_program_no_streams(tmp_path, monkeypatch):
    """A GUI program started with any redirected stream gets this process' other ones too
    (STARTF_USESTDHANDLES) - it would not start the way a double-click starts it."""
    import subprocess
    smoke = _load("smoke_app")
    folder = tmp_path / "app"
    folder.mkdir()
    (folder / "auralite-app.json").write_text(json.dumps(INFO), encoding="utf-8")
    seen: dict = {}

    def fake_run(command, timeout, check, **kwargs):
        seen.clear()
        seen.update(kwargs)
        captured = kwargs.get("stdout") == subprocess.PIPE
        return subprocess.CompletedProcess(command, 0, b"out" if captured else None, None)

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    checker = smoke.Smoke(folder, tmp_path, "")
    monkeypatch.setattr(smoke, "WINDOWS", True)
    assert checker.run(["studio"], no_console=True) == (0, "")
    assert not {"stdin", "stdout", "stderr"} & set(seen)
    monkeypatch.setattr(smoke, "WINDOWS", False)
    checker.run(["studio"], no_console=True)
    assert seen["stdout"] == seen["stderr"] == subprocess.DEVNULL
    assert seen["preexec_fn"] is smoke._close_std
    assert checker.run(["auralite"]) == (0, "out") and seen["stdout"] == subprocess.PIPE


@pytest.mark.skipif(os.name != "posix", reason="the fake app's programs are shell scripts")
def test_the_smoke_test_passes_a_working_app_and_fails_a_broken_one(tmp_path):
    smoke = _load("smoke_app")
    folder = tmp_path / "AuraLite-Studio-2.21.0"
    folder.mkdir()
    (folder / "auralite-app.json").write_text(json.dumps(INFO), encoding="utf-8")
    cli = folder / "auralite"
    cli.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then echo "auralite 2.21.0"; exit 0; fi\n'
                   'echo "App: the AuraLite Studio app 2.21.0"\necho "    RESULT: OK - built"\n'
                   'echo "  => the window starts"\n', encoding="utf-8")
    cli.chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    checker = smoke.Smoke(folder, work, "offscreen")
    assert checker.check(["version", "doctor"])
    assert all(ok for _n, ok, _s, _t in checker.results)
    cli.write_text("#!/bin/sh\necho broken\nexit 1\n", encoding="utf-8")
    broken = smoke.Smoke(folder, work, "offscreen")
    assert not broken.check(["version", "doctor"])
    assert [ok for _n, ok, _s, _t in broken.results] == [False, False]     # each on its own


# ----------------------------------------------------------------------
#  CI and the release
# ----------------------------------------------------------------------
def _workflow(name: str) -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((REPO / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def _runs(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in job["steps"])


def test_ci_builds_and_checks_the_windows_app_on_every_push():
    ci = _workflow("ci.yml")
    assert "pyinstaller-smoke" not in ci["jobs"]
    job = ci["jobs"]["windows-app"]
    assert job["runs-on"] == "windows-latest" and "if" not in job
    triggers = ci[True]                                          # YAML reads `on:` as True
    assert "push" in triggers and "pull_request" in triggers and not triggers["workflow_dispatch"]
    runs = _runs(job)
    assert "pip install -r packaging/app/requirements.txt" in runs
    assert "pip install --no-deps ." in runs
    assert "python packaging/app/build_app.py --zip" in runs
    assert "python packaging/app/smoke_app.py dist/AuraLite-Studio-*-windows-x64.zip" in runs
    python = next(step["with"]["python-version"] for step in job["steps"]
                  if "setup-python" in str(step.get("uses", "")))
    assert python == "3.13"


def test_ci_coverage_measures_the_studios_qt_code():
    job = _workflow("ci.yml")["jobs"]["coverage"]
    assert job["env"]["QT_QPA_PLATFORM"] == "offscreen"
    runs = _runs(job)
    assert "'.[test,studio]'" in runs and "libxkbcommon0" in runs
    assert "--cov-fail-under=85" in runs


def test_the_release_attaches_the_windows_app_without_waiting_for_it():
    release = _workflow("release.yml")
    jobs = release["jobs"]
    assert jobs["windows-app"]["needs"] == "verify"
    assert jobs["windows-app"]["runs-on"] == "windows-latest"
    assert "smoke_app.py" in _runs(jobs["windows-app"])
    publish = jobs["windows-app-publish"]
    assert publish["needs"] == ["verify", "release", "windows-app"]
    assert publish["permissions"]["contents"] == "write"
    assert publish["permissions"]["attestations"] == "write"
    assert "gh release upload" in _runs(publish) and "sha256sum -c" in _runs(publish)
    assert any("actions/attest" in str(step.get("uses", "")) for step in publish["steps"])
    assert "windows-app" not in jobs["release"]["needs"]        # the wheel does not wait
    body = next(step["with"]["body"] for step in jobs["release"]["steps"]
                if "action-gh-release" in str(step.get("uses", "")))
    assert "-windows-x64.zip" in body and "AuraLite Studio.exe" in body


def test_build_exe_bat_builds_the_app_the_way_ci_does():
    text = (REPO / "build_exe.bat").read_text(encoding="utf-8")
    assert "packaging\\app\\requirements.txt" in text
    assert "packaging\\app\\build_app.py" in text and "--zip" in text
    assert "-m pip install --no-deps --force-reinstall ." in text   # this folder's code
    assert "gui_app.py" not in text and "--exclude-module PySide6" not in text
