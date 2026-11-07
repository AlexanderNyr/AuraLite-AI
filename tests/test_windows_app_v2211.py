"""2.21.1: the check of the Windows app runs to its end, and runs the app as a Russian user.

2.21.0's check (packaging/app/smoke_app.py) ended in its first second on CI's Windows runner:
it printed the folder it unzips the app into - a name with Russian letters, on purpose - and
Python on Windows writes a pipe in the ANSI code page, which has none (UnicodeEncodeError).
So the Release workflow attached no zip to 2.21.0. Here:

* the check run the way that runner runs it (a pipe in cp1252), on a zip;
* the app's data, temporary files and training files under a Russian Windows user name;
* the annotations a failed check (or build) leaves on a GitHub Actions run;
* build_app.py's output in the same code page.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
APP_DIR = REPO / "packaging" / "app"
VERSION = "2.21.1"
INFO = {"app": "AuraLite Studio", "version": VERSION, "platform": "linux-x64",
        "python": "3.13.9", "torch": "2.15.0+cpu", "qt": "6.9.3", "built": "2026-11-07",
        "commit": "abc1234"}
#: "User" in Russian, as the smoke test names the user's folder
USER = "\u041f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c"


def _load(name: str):
    """A script of packaging/app as a module (the folder is not a package)."""
    spec = importlib.util.spec_from_file_location(f"_auralite_app_v2211_{name}",
                                                  APP_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _app_folder(parent: Path, cli_script: str) -> Path:
    folder = parent / f"AuraLite-Studio-{VERSION}"
    folder.mkdir(parents=True)
    (folder / "auralite-app.json").write_text(json.dumps(INFO), encoding="utf-8")
    cli = folder / "auralite"
    cli.write_text(cli_script, encoding="utf-8")
    cli.chmod(0o755)
    return folder


#: A console program that passes `version` and `doctor`, and says where it was told to keep
#: its data and temporary files.
WORKING_CLI = (
    "#!/bin/sh\n"
    f'if [ "$1" = "--version" ]; then echo "auralite {VERSION}"; exit 0; fi\n'
    f'echo "App: the AuraLite Studio app {VERSION}"\n'
    'echo "home: $AURALITE_HOME"\n'
    'echo "temp: $TMPDIR"\n'
    'echo "    RESULT: OK - built"\n'
    'echo "  => the window starts"\n'
)


def _zip(folder: Path, archive: Path) -> Path:
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            zf.write(path, path.relative_to(folder.parent).as_posix())   # keeps the mode
    return archive


# ----------------------------------------------------------------------
#  The check as CI's Windows runner runs it
# ----------------------------------------------------------------------
@pytest.mark.skipif(os.name != "posix", reason="the fake app's programs are shell scripts")
def test_the_check_runs_to_its_end_through_a_windows_pipe(tmp_path):
    """2.21.0's check died on its first line here: stdout a pipe in cp1252, as Python on
    Windows has it, and the folder it unzips into named with Russian letters."""
    smoke = _load("smoke_app")
    archive = _zip(_app_folder(tmp_path / "src", WORKING_CLI),
                   tmp_path / f"AuraLite-Studio-{VERSION}-linux-x64.zip")
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_ACTIONS"}
    env["PYTHONIOENCODING"] = "cp1252"
    proc = subprocess.run([sys.executable, str(APP_DIR / "smoke_app.py"), str(archive),
                           "--only", "version", "doctor"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                          timeout=120, check=False)
    out = proc.stdout.decode("utf-8")              # what the runner's log receives
    assert proc.returncode == 0, out
    assert "UnicodeEncodeError" not in out
    assert f"Unzipping AuraLite-Studio-{VERSION}-linux-x64.zip into " in out
    assert smoke.ODD_FOLDER in out
    assert "PASS  version" in out and "PASS  doctor" in out and "RESULT: the app works" in out
    # the app was told to keep its data and temporary files under the Russian user name
    home = next(line for line in out.splitlines() if "home: " in line)
    temp = next(line for line in out.splitlines() if "temp: " in line)
    assert f"{USER}{os.sep}.auralite" in home
    assert USER in temp and temp.rstrip().endswith("Temp")
    assert "::error" not in out                    # not on GitHub Actions: no annotations


def test_utf8_output_survives_the_ansi_code_page(monkeypatch):
    for name in ("smoke_app", "build_app"):
        module = _load(name)
        out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        err = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        module.utf8_output()
        print(f"C:\\Users\\{USER}")
        print(USER, file=sys.stderr)
        out.flush()
        err.flush()
        assert out.buffer.getvalue().decode("utf-8").strip() == f"C:\\Users\\{USER}", name
        assert err.buffer.getvalue().decode("utf-8").strip() == USER, name
        # a stream that cannot change its encoding is left alone
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        module.utf8_output()
        monkeypatch.undo()


def test_the_check_switches_to_utf8_before_anything_else(monkeypatch, tmp_path):
    """main() itself, not only the script: a broken check still reports in UTF-8."""
    smoke = _load("smoke_app")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", out)
    missing = tmp_path / f"{USER}.zip"             # unzipping it fails - after the first line
    with pytest.raises(FileNotFoundError):
        smoke.main([str(missing)])
    out.flush()
    assert f"Unzipping {USER}.zip into" in out.buffer.getvalue().decode("utf-8")


# ----------------------------------------------------------------------
#  The app under a Russian Windows user name
# ----------------------------------------------------------------------
def test_the_app_keeps_its_data_and_temporary_files_under_a_russian_user_name(tmp_path):
    smoke = _load("smoke_app")
    assert smoke.USER_FOLDER == USER and not USER.isascii()
    folder = tmp_path / f"AuraLite-Studio-{VERSION}"
    folder.mkdir()
    (folder / "auralite-app.json").write_text(json.dumps(INFO), encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    checker = smoke.Smoke(folder, work, "")
    user = work / USER
    assert checker.env["AURALITE_HOME"] == str(user / ".auralite")
    temp = user / "AppData" / "Local" / "Temp"
    assert checker.env["TEMP"] == checker.env["TMP"] == checker.env["TMPDIR"] == str(temp)
    assert temp.is_dir() and checker.files == user / "Documents" and checker.files.is_dir()
    # the files the training checks read and write are the user's documents
    assert checker._text().parent == checker.files
    assert checker._text().read_text(encoding="utf-8").startswith("0: The quick brown fox")


@pytest.mark.skipif(os.name != "posix", reason="the fake app's programs are shell scripts")
def test_the_training_checks_use_the_users_documents(tmp_path):
    smoke = _load("smoke_app")
    script = ("#!/bin/sh\n"
              'echo "args: $*"\n'
              'for a in "$@"; do case "$a" in *.pt) echo saved > "$a";; esac; done\n'
              'echo "saved: model"\n'
              'echo "20261107-000000-train  finished  Train"\n')
    folder = _app_folder(tmp_path, script)
    work = tmp_path / "work"
    work.mkdir()
    checker = smoke.Smoke(folder, work, "offscreen")
    ok, text = checker.step_train()
    documents = str(work / USER / "Documents")
    assert ok, text
    assert f"--data {documents}{os.sep}text.txt" in text
    assert f"--model {documents}{os.sep}model.pt" in text
    _ok, text = checker.step_workers()
    assert f"--out {documents}{os.sep}shards" in text


# ----------------------------------------------------------------------
#  Annotations on a GitHub Actions run
# ----------------------------------------------------------------------
def test_annotations_escape_what_github_reads():
    smoke = _load("smoke_app")
    build = _load("build_app")
    assert smoke.annotation("error", "smoke: a,b", "50%\r\nnext") == \
        "::error title=smoke%3A a%2Cb::50%25%0D%0Anext"
    assert build.github_error("build_app.py", "no torch:\n  pip install") == \
        "::error title=build_app.py::no torch:%0A  pip install"
    text = "\n".join(f"line {i}" for i in range(100))
    assert smoke.tail(text, lines=3) == "line 97\nline 98\nline 99"
    assert smoke.tail(text, lines=3, chars=10) == "98\nline 99"
    assert smoke.tail(text) == "\n".join(f"line {i}" for i in range(40, 100))


def _checker(smoke, tmp_path):
    folder = tmp_path / f"AuraLite-Studio-{VERSION}"
    folder.mkdir(parents=True)
    (folder / "auralite-app.json").write_text(json.dumps(INFO), encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    return smoke.Smoke(folder, work, "")


def test_a_failed_check_is_an_annotation_on_github_only(tmp_path, monkeypatch, capsys):
    smoke = _load("smoke_app")
    monkeypatch.setattr(smoke.Smoke, "step_version", lambda self: (True, "auralite 2.21.1"))
    monkeypatch.setattr(smoke.Smoke, "step_doctor",
                        lambda self: (False, "Processor: 2 cores\nRESULT: FAILED - no Qt"))
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert not _checker(smoke, tmp_path / "a").check(["version", "doctor"])
    assert "::error" not in capsys.readouterr().out
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert not _checker(smoke, tmp_path / "b").check(["version", "doctor"])
    errors = [line for line in capsys.readouterr().out.splitlines() if line.startswith("::")]
    assert errors == ["::error title=smoke%3A doctor::Processor: 2 cores%0ARESULT: FAILED - no Qt"]


def test_a_crash_of_the_check_itself_is_an_annotation(monkeypatch, capsys):
    smoke = _load("smoke_app")

    def broken(argv):
        raise RuntimeError("the check broke")

    monkeypatch.setattr(smoke, "_main", broken)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with pytest.raises(RuntimeError):
        smoke.main([])
    out = capsys.readouterr().out
    assert out.startswith("::error title=smoke_app.py::Traceback")
    assert "RuntimeError: the check broke" in out
    monkeypatch.delenv("GITHUB_ACTIONS")
    with pytest.raises(RuntimeError):
        smoke.main([])
    assert "::error" not in capsys.readouterr().out


def test_a_failed_build_is_an_annotation_on_github(monkeypatch, capsys):
    build = _load("build_app")

    def fails(*args, **kwargs):
        raise build.BuildError(f"C:\\Users\\{USER}\\AuraLite-AI: torch 2.14.0 is not the pin")

    monkeypatch.setattr(build, "build", fails)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert build.main(["--zip"]) == 1
    captured = capsys.readouterr()
    assert f"build_app: C:\\Users\\{USER}\\AuraLite-AI: torch 2.14.0" in captured.err
    assert captured.out.strip() == ("::error title=build_app.py::"
                                    f"C:\\Users\\{USER}\\AuraLite-AI: torch 2.14.0 is not the pin")
    monkeypatch.delenv("GITHUB_ACTIONS")
    assert build.main(["--zip"]) == 1
    assert "::error" not in capsys.readouterr().out


def test_a_failed_build_reports_through_a_windows_pipe(monkeypatch):
    """The message names a path with a Russian user name; stderr is a pipe in cp1252."""
    build = _load("build_app")

    def fails(*args, **kwargs):
        raise build.BuildError(f"C:\\Users\\{USER}\\AuraLite-AI is not where AuraLite is from")

    monkeypatch.setattr(build, "build", fails)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    err = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stderr", err)
    assert build.main(["--zip"]) == 1
    err.flush()
    assert USER in err.buffer.getvalue().decode("utf-8")


# ----------------------------------------------------------------------
#  The docs say what the check covers
# ----------------------------------------------------------------------
def test_the_docs_describe_the_check_and_the_size():
    guide = (REPO / "docs" / "windows-app.md").read_text(encoding="utf-8")
    assert "under a user name in Cyrillic letters" in guide
    assert "(about 250 MB)" in guide and "annotation" in guide
    releasing = (REPO / "docs" / "releasing.md").read_text(encoding="utf-8")
    assert "A fix in the code comes with the next patch\nrelease" in releasing
