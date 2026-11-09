"""Check a built AuraLite app the way a user meets it (2.21) - CI runs it after every build.

    python packaging/app/smoke_app.py dist/AuraLite-Studio-2.26.0-windows-x64.zip
    python packaging/app/smoke_app.py dist/app/AuraLite-Studio-2.26.0

A zip is unzipped first, into a folder whose path has a space and letters outside ASCII.
Everything runs as under a Windows user name in Russian (2.21.1): the app's data
(AURALITE_HOME), its temporary files (TEMP, TMP) and the files it trains on are in a folder
with that name, as in C:\\Users\\<name>. Every check runs even after one has failed, so one CI
run tells everything:

1. ``version``  - ``auralite --version``: the console program starts and knows its version;
2. ``doctor``   - ``auralite doctor --no-speed --studio``: this computer, then the Studio's
   self-test in a fresh process ("the window starts");
3. ``selftest`` - the Studio's own program with ``--self-test``; it has no console on
   Windows, so its report file is read;
4. ``start``    - the Studio's program with no arguments and no console, as a double-click
   starts it; it closes itself after a few seconds (AURALITE_STUDIO_SMOKE_SECONDS): exit
   code 0 and "the window is open" in its log;
5. ``train``    - ``auralite train`` on a small text (PyTorch trains inside the app), and the
   run is in ``auralite runs``;
6. ``workers``  - ``auralite data prepare --workers 2 --no-fast``: worker processes of a
   frozen program start the program again, and must turn into workers;
7. ``bench``    - ``auralite bench --quick``: a model trains and writes, the report is saved;
8. ``images``   - ``auralite image-train`` on a folder of pictures (2.26: the app brings
   Pillow, which reads them and writes the samples).

Only the standard library: this runs with any Python, outside the app. It prints UTF-8 to any
console or pipe, and on GitHub Actions a failed check is also an annotation of the run - with
the end of its output, readable without signing in (the log is not).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
import zlib
from pathlib import Path

#: A folder with a space and letters outside ASCII ("AuraLite test", the second word in
#: Russian), where a zip is unzipped.
ODD_FOLDER = "AuraLite \u0442\u0435\u0441\u0442"
#: A Windows user name in Russian ("User": what many Russian Windows installs call the first
#: account) - the app's data, temporary files and the files it trains on are under it.
USER_FOLDER = "\u041f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c"
SMOKE_ENV = "AURALITE_STUDIO_SMOKE_SECONDS"
SMOKE_SECONDS = 8
STEPS = ("version", "doctor", "selftest", "start", "train", "workers", "bench", "images")
WINDOWS = sys.platform == "win32"
#: How much of a failed check's output its annotation keeps (the end of it).
ANNOTATION_LINES = 60
ANNOTATION_CHARS = 6000


def utf8_output() -> None:
    """Print UTF-8 whatever the console or pipe. Python on Windows writes a pipe - CI's log, a
    `> file` - in the ANSI code page, which has no Cyrillic: in 2.21.0 the first path with Russian
    letters in it (the folder the zip is unzipped into) ended the check in its first second."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):   # a stream that cannot change: leave it
                pass


def on_github() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def annotation(kind: str, title: str, message: str) -> str:
    """A GitHub Actions workflow command (``::error title=...::...``): the run's page shows it,
    and the check run's annotations - readable without signing in, unlike the log - keep it."""
    def escape(text: str) -> str:
        return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    title = escape(title).replace(":", "%3A").replace(",", "%2C")
    return f"::{kind} title={title}::{escape(message)}"


def tail(text: str, lines: int = ANNOTATION_LINES, chars: int = ANNOTATION_CHARS) -> str:
    """The end of `text`: its last `lines` lines, at most `chars` characters."""
    kept = "\n".join(text.rstrip().splitlines()[-lines:])
    return kept[-chars:]


def png_bytes(width: int, height: int, seed: int = 0) -> bytes:
    """A small RGB picture as a PNG file, with the standard library only (no Pillow outside
    the app): stripes whose colors follow `seed`."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    rows = b"".join(b"\x00" + bytes((x * 16 + seed * 31) % 256 if c == 0 else
                                     (y * 16 + seed * 17) % 256 if c == 1 else
                                     (seed * 40) % 256
                                     for x in range(width) for c in range(3))
                    for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows))
            + chunk(b"IEND", b""))


def unzip(archive: Path, target: Path) -> Path:
    """Unzip `archive` into `target` (Unix permissions kept) and return the app's folder."""
    with zipfile.ZipFile(archive) as zf:
        tops = {name.split("/", 1)[0] for name in zf.namelist()}
        if len(tops) != 1:
            raise SystemExit(f"{archive.name}: expected one folder at the top, found {sorted(tops)}")
        for info in zf.infolist():
            path = Path(zf.extract(info, target))
            mode = (info.external_attr >> 16) & 0o777
            if mode and not info.is_dir():
                path.chmod(mode)
    return target / tops.pop()


def _close_std() -> None:                 # Linux: what a double-click gives a Windows GUI program
    os.close(1)
    os.close(2)


class Smoke:
    def __init__(self, app: Path, work: Path, platform: str):
        self.app, self.work = app, work
        self.user = work / USER_FOLDER                  # as C:\Users\<a Russian name>
        self.home = self.user / ".auralite"
        self.files = self.user / "Documents"
        self.temp = self.user / "AppData" / "Local" / "Temp"
        for folder in (self.files, self.temp):
            folder.mkdir(parents=True, exist_ok=True)
        self.cli = app / ("auralite.exe" if WINDOWS else "auralite")
        self.studio = app / ("AuraLite Studio.exe" if WINDOWS else "auralite-studio")
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("PYTHON", "QT_", "AURALITE_", "VIRTUAL_ENV"))}
        env["AURALITE_HOME"] = str(self.home)
        env["TEMP"] = env["TMP"] = env["TMPDIR"] = str(self.temp)
        if platform:
            env["QT_QPA_PLATFORM"] = platform
        self.env = env
        self.results: list[tuple[str, bool, float, str]] = []
        info = json.loads((app / "auralite-app.json").read_text(encoding="utf-8"))
        self.version = str(info.get("version") or "")

    # ---- running the app's programs --------------------------------------------------
    def run(self, command: list[str], *, timeout: float = 900, env: dict | None = None,
            no_console: bool = False) -> tuple[int | None, str]:
        """Run `command`; (exit code or None on a timeout, what it printed). `no_console`:
        no standard streams at all, like a program started by a double-click."""
        kwargs: dict = {"env": dict(self.env, **(env or {})), "cwd": str(self.work)}
        if no_console:
            if not WINDOWS:
                kwargs.update(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, preexec_fn=_close_std)
            # Windows: no stream at all - one redirected one would hand the GUI program this
            # process' others too (STARTF_USESTDHANDLES), and it would start with a console's
            # streams instead of none, as from a double-click
        else:
            kwargs.update(stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
        print("  $ " + " ".join(f'"{c}"' if " " in c else c for c in command), flush=True)
        try:
            proc = subprocess.run(command, timeout=timeout, check=False, **kwargs)
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout if isinstance(exc.stdout, bytes) else b""
            return None, out.decode("utf-8", "replace") + f"\n(no end after {timeout:.0f} s)"
        out = proc.stdout.decode("utf-8", "replace") if isinstance(proc.stdout, bytes) else ""
        return proc.returncode, out

    def log(self, name: str) -> str:
        try:
            return (self.home / "logs" / name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    # ---- the checks -------------------------------------------------------------------
    def step_version(self) -> tuple[bool, str]:
        code, out = self.run([str(self.cli), "--version"], timeout=600)
        return code == 0 and f"auralite {self.version}" in out, out

    def step_doctor(self) -> tuple[bool, str]:
        code, out = self.run([str(self.cli), "doctor", "--no-speed", "--studio"])
        ok = (code == 0 and "=> the window starts" in out
              and "App: the AuraLite Studio app" in out and "RESULT: OK" in out)
        return ok, out

    def step_selftest(self) -> tuple[bool, str]:
        report = self.home / "logs" / "studio-self-test.txt"
        if report.exists():
            report.unlink()
        code, out = self.run([str(self.studio), "--self-test"], no_console=WINDOWS)
        text = self.log("studio-self-test.txt")
        ok = code == 0 and "RESULT: OK" in text and "Installed: the AuraLite Studio app" in text
        if WINDOWS:
            # the Visual C++ runtime is the app's own: msvcp140.dll loaded first from it, and
            # PyTorch's vcruntime140_threads.dll from its folder (rthook_vcruntime.py)
            folder = os.path.normcase(str(self.app))
            lines = text.splitlines()
            runtime = next((x for x in lines if "Visual C++ runtime (" in x), "")
            threads = next((x for x in lines if "vcruntime140_threads.dll:" in x), "")
            ok = ok and ("loaded first: app" in runtime or folder in os.path.normcase(runtime)) \
                and folder in os.path.normcase(threads)
        return ok, f"exit code {code}\n{out}\n--- {report.name}:\n{text}"

    def step_start(self) -> tuple[bool, str]:
        before = len(self.log("studio.log"))
        started = time.time()
        code, _out = self.run([str(self.studio)], timeout=600, no_console=True,
                              env={SMOKE_ENV: str(SMOKE_SECONDS)})
        log = self.log("studio.log")[before:]
        output = self.home / "logs" / "studio-output.log"
        watched = output.is_file() and output.stat().st_mtime >= started - 1
        ok = (code == 0 and "the window is open" in log and "closing the window" in log
              and watched)
        return ok, (f"exit code {code}; the start ran the window in a watched child process: "
                    f"{watched}\n--- studio.log (this start):\n{log}\n--- studio-output.log:\n"
                    f"{self.log('studio-output.log')}\n--- studio-faults.log:\n"
                    f"{self.log('studio-faults.log')}")

    def _text(self) -> Path:
        path = self.files / "text.txt"
        if not path.exists():
            line = "The quick brown fox jumps over the lazy dog. 0123456789\n"
            lines = [f"{i}: {line}" for i in range(400)]
            path.write_text("".join(lines), encoding="utf-8")
        return path

    def step_train(self) -> tuple[bool, str]:
        params = self.files / "params.json"
        params.write_text(json.dumps({"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64,
                                      "seq_length": 16, "batch_size": 8, "epochs": 1,
                                      "bpe_vocab_size": 120}), encoding="utf-8")
        model = self.files / "model.pt"
        code, out = self.run([str(self.cli), "train", "--data", str(self._text()), "--params",
                              str(params), "--model", str(model), "--quiet"])
        runs_code, runs = self.run([str(self.cli), "runs"], timeout=600)
        ok = (code == 0 and model.is_file() and "saved" in out and runs_code == 0
              and "train" in runs)
        return ok, f"{out}\n--- auralite runs:\n{runs}"

    def step_workers(self) -> tuple[bool, str]:
        out_dir = self.files / "shards"
        code, out = self.run([str(self.cli), "data", "prepare", str(self._text()), "--out",
                              str(out_dir), "--workers", "2", "--no-fast", "--vocab-size",
                              "300", "--overwrite"])
        files = sorted(p.name for p in out_dir.rglob("*") if p.is_file()) if out_dir.is_dir() \
            else []
        ok = code == 0 and len(files) >= 2 and "tokenizing in 2 worker processes" in out
        return ok, f"{out}\n--- files: {', '.join(files) or 'none'}"

    def step_bench(self) -> tuple[bool, str]:
        """2.22: `auralite bench --quick` - a model trains and writes, and the report (Markdown
        and JSON) is written under the user's folder."""
        out_dir = self.files / "bench"
        code, out = self.run([str(self.cli), "bench", "--quick", "--sizes", "tiny", "--seconds",
                              "3", "--out", str(out_dir)])
        reports = sorted(p.name for p in out_dir.glob("bench-*")) if out_dir.is_dir() else []
        ok = (code == 0 and len(reports) == 2 and "| tiny: 4 layers x 128 | 0.9M |" in out
              and "Saved: " in out)
        return ok, f"{out}\n--- reports: {', '.join(reports) or 'none'}"

    def step_images(self) -> tuple[bool, str]:
        """2.26: the picture generator trains on a folder of PNG files and writes samples -
        Pillow inside the app reads and writes them."""
        folder = self.files / "pictures"
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(8):
            (folder / f"p{i}.png").write_bytes(png_bytes(16, 16, i))
        model, grid = self.files / "pictures.pt", self.files / "samples.png"
        code, out = self.run([str(self.cli), "image-train", "--images", str(folder),
                              "--preset", "tiny", "--epochs", "1", "--img-size", "16",
                              "--batch-size", "4", "--val-split", "0", "--model", str(model),
                              "--samples", str(grid), "--quiet"])
        ok = (code == 0 and "saved: " in out and model.is_file() and grid.is_file()
              and grid.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n")
        return ok, out

    def check(self, steps: list[str]) -> bool:
        for name in steps:
            print(f"\n== {name}", flush=True)
            started = time.monotonic()
            try:
                ok, text = getattr(self, f"step_{name}")()
            except Exception as exc:          # the check itself broke: still report the rest
                ok, text = False, f"{type(exc).__name__}: {exc}"
            seconds = time.monotonic() - started
            for line in text.rstrip().splitlines():
                print("    " + line)
            print(f"  {'PASS' if ok else 'FAIL'} {name} ({seconds:.0f} s)", flush=True)
            self.results.append((name, ok, seconds, text))
        print("\nThe app: " + str(self.app))
        for name, ok, seconds, _text in self.results:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:9} {seconds:6.0f} s")
        if on_github():
            for name, ok, _seconds, text in self.results:
                if not ok:
                    print(annotation("error", f"smoke: {name}", tail(text)), flush=True)
        return all(ok for _name, ok, _s, _t in self.results)


def main(argv: list[str] | None = None) -> int:
    utf8_output()
    try:
        return _main(argv)
    except Exception:                     # the check itself broke: say where, on the run's page too
        if on_github():
            print(annotation("error", "smoke_app.py", tail(traceback.format_exc())), flush=True)
        raise


def _main(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(description="Check a built AuraLite app.")
    parser.add_argument("target", type=Path, help="the app's folder, or its zip")
    parser.add_argument("--only", nargs="+", choices=STEPS, help="run only these checks")
    parser.add_argument("--platform", default=None,
                        help="Qt's platform (default: Windows' own; offscreen elsewhere "
                             "without a display)")
    parser.add_argument("--keep", action="store_true", help="keep the work folder")
    args = parser.parse_args(argv)
    platform = args.platform
    if platform is None and not WINDOWS and not (os.environ.get("DISPLAY")
                                                 or os.environ.get("WAYLAND_DISPLAY")):
        platform = "offscreen"
    work = Path(tempfile.mkdtemp(prefix="auralite-smoke-")).resolve()
    try:
        target = args.target.resolve()
        if target.suffix.lower() == ".zip":
            print(f"Unzipping {target.name} into {work / ODD_FOLDER}", flush=True)
            app = unzip(target, work / ODD_FOLDER)
        else:
            app = target
        smoke = Smoke(app, work, platform or "")
        ok = smoke.check(list(args.only or STEPS))
    finally:
        if args.keep:
            print(f"(kept {work})")
        else:
            shutil.rmtree(work, ignore_errors=True)
    print("\nRESULT: " + ("the app works" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
