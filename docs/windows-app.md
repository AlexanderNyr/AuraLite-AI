# AuraLite Studio for Windows — no Python needed

Since 2.21 every release has a Windows app: one zip with AuraLite Studio, the
`auralite` command and everything they run on — Python, PyTorch and Qt. Unzip it,
double-click `AuraLite Studio.exe`, and the Studio of [docs/studio.md](studio.md)
opens: the same tabs, the same library of runs, the same models. Nothing to
install, no Python to know about.

## Download and start

1. Open the [latest release](https://github.com/AlexanderNyr/AuraLite-AI/releases/latest)
   and download `AuraLite-Studio-<version>-windows-x64.zip` (about 250 MB).
2. Unzip it — right-click → *Extract All…* — into a folder of your own: your
   Desktop, Documents, `C:\AuraLite`. Not into `C:\Program Files` (the app writes
   nothing next to itself, but Windows makes that folder awkward), and keep the
   whole folder together.
3. Double-click `AuraLite Studio.exe` in the unzipped folder.

The first start takes longer — up to a minute or two while Windows Defender checks
the new files; later starts take seconds. The app is not signed, so Windows may say
**"Windows protected your PC"**: choose *More info*, then *Run anyway*.

The setup wizard opens on the first start: it looks at the computer, asks what you
want to train and on what, and recommends a model and settings that fit — from here
on everything is as in [docs/studio.md](studio.md).

**Requirements.** Windows 10 (version 1809 or newer) or Windows 11, 64-bit. Any
64-bit Intel or AMD processor — the app's Qt is 6.9, which also runs on processors
without SSE4.2; AVX makes training several times faster. 4 GB of memory to start,
8 GB or more to fine-tune half-billion-parameter models. About 1.2 GB of disk for
the app, plus the models you download (0.3–3 GB each).

**It trains on the processor.** PyTorch's CUDA build is larger than a release may
attach (2 GB per file), so the app has the CPU build. For an NVIDIA GPU install
AuraLite with Python and PyTorch's CUDA build ([quickstart](quickstart.md)); the
Studio is the same there.

### Check the download (optional)

Each zip has a `.sha256` file next to it on the release page, and the release has a
signed record of the CI run that built it:

```powershell
Get-FileHash .\AuraLite-Studio-2.21.1-windows-x64.zip -Algorithm SHA256   # compare with the .sha256
gh attestation verify .\AuraLite-Studio-2.21.1-windows-x64.zip -R AlexanderNyr/AuraLite-AI
```

## Where your things are

The app's folder holds only the app. Your models, runs, logs, settings and the
models you download are in `%USERPROFILE%\.auralite` (`C:\Users\<you>\.auralite`) —
the same folder AuraLite installed with Python uses, so the two see the same
library. Set the `AURALITE_HOME` variable to keep them elsewhere (another disk).
*Tools → Open the AuraLite folder* opens it.

## The command line: `auralite.exe`

The folder also has `auralite.exe`, the `auralite` command of
[the quickstart](quickstart.md) — for a console. Open one in the folder (in
Explorer: type `cmd` in the address bar) and run, for example:

```bat
auralite.exe doctor --studio
auralite.exe runs
auralite.exe train --data notes.txt --preset 25m --model my-model.pt
auralite.exe chat --model my-model.pt
```

Everything the command does with Python it does here — training on several
computers too (`--cluster-workers`, [docs/training.md](training.md)) — except the
parts the app leaves out (below).

## Updating and removing

*Help → Check for updates…* says when a new version is out and gives the address
of its zip. Download it, unzip it into a **new** folder, start it from there, and
delete the old folder once the new one works — your models and runs stay in
`%USERPROFILE%\.auralite`, untouched. (Unzipping over the old folder works too, but
leaves the old version's files that the new one no longer has.)

To remove the app delete its folder; delete `%USERPROFILE%\.auralite` as well to
remove your models, runs and settings.

## What the app leaves out

The app carries what the Studio needs — the engine, Hugging Face downloads and
export, GGUF export, the fast `tokenizers` package, `.zst` data — and leaves out
the large optional parts. Each comes with AuraLite installed with Python
(`pip install "auralite-ai[EXTRA]"`, [quickstart](quickstart.md)); the app says so
when you reach one.

| Left out | What it is for | With Python |
|---|---|---|
| the classic interface | `auralite-gui` (tkinter): images, quantization, evaluation, the transformers backend | `pip install auralite-ai`, then `auralite-gui` |
| transformers, PEFT | models AuraLite's own engine does not import, through transformers | `[hf]` |
| Hugging Face datasets, Parquet | `hf:NAME` datasets and `.parquet` files as data | `[data]` |
| the server | `auralite serve`, the OpenAI-compatible API | `[serve]` |
| ONNX | `auralite export --format onnx` | `[export]` |
| llama.cpp | a `.gguf` file as the teacher of `distill` and its perplexity (`auralite info`); a llama.cpp *server* (`http://…`) works without it, and exporting to GGUF is included | `[gguf]` |
| CUDA | training on an NVIDIA GPU | PyTorch's CUDA build |

## If it does not start

Whatever stops the Studio is shown in a message box and written to
`%USERPROFILE%\.auralite\logs` — `studio.log`, `studio-output.log` (what it
printed) and `studio-faults.log` (where it crashed). In a console in the app's
folder:

```bat
auralite.exe doctor --studio
```

checks the computer and starts the Studio's self-test in a fresh process;
`RESULT: OK` means the window starts. `"AuraLite Studio.exe" --self-test` runs the
same check through the Studio's own program and leaves its report in
`%USERPROFILE%\.auralite\logs\studio-self-test.txt`. That output is what an issue
needs.

| What happens | What to do |
|---|---|
| Windows says it cannot extract a file, or *the path is too long* | unzip into a short folder such as `C:\AuraLite` |
| *Windows protected your PC* | *More info* → *Run anyway* (the app is not signed) |
| an antivirus removes or blocks a file | programs built with PyInstaller are sometimes mistaken for malware: restore the file, allow the app's folder, unzip it again — and report the false alarm to the antivirus' maker |
| *A library would not load*, *DLL load failed*, `WinError 1114` | install the latest [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) and unzip the app again into a new, empty folder |
| *Could not find the Qt platform plugin "windows"* | the folder is incomplete — unzip the whole zip again; remove a `QT_PLUGIN_PATH` variable another program set |
| it does not start on Windows 7 or 8.1 | Qt 6 needs Windows 10 or newer |

## Building it yourself

The zip is built by CI from the release's commit — never on somebody's computer —
with the packages pinned in
[`packaging/app/requirements.txt`](../packaging/app/requirements.txt): Python
3.13, PyTorch's CPU build, PySide6-Essentials 6.9.3, PyInstaller. To build the same
app yourself on Windows, run `build_exe.bat` in a clone of the repository, or:

```bat
py -3.13 -m venv .venv-app
.venv-app\Scripts\activate
pip install -r packaging\app\requirements.txt
pip install --no-deps .
python packaging\app\build_app.py --zip
python packaging\app\smoke_app.py dist\AuraLite-Studio-2.21.1-windows-x64.zip
```

`build_app.py` refuses to build when the installed packages differ from the pins
or AuraLite is not installed from that very folder; it writes
`dist\app\AuraLite-Studio-<version>\` and, with `--zip`, the zip and its SHA-256.
The build computer needs a recent
[Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe):
PyTorch needs `vcruntime140_threads.dll`, which neither its wheel nor PyInstaller
brings, so the app takes that computer's copy (and loads it first). `build_app.py`
refuses a build that lacks any part of the runtime its DLLs import.
The same commands build a Linux app on Linux (`auralite-studio`, `auralite`); the
releases attach the Windows one.

Inside the folder:

| | |
|---|---|
| `AuraLite Studio.exe` | the Studio — a program without a console; started with no console at all (a double-click), it runs the window in a child process it watches, so a crash inside Qt or PyTorch still ends in a message box |
| `auralite.exe` | the `auralite` command, a console program |
| `auralite-app.json` | which build this is: version, Python, PyTorch, Qt, date, commit — `auralite.exe doctor` and *Help → About* show it |
| `README.txt` | the short version of this page |
| `licenses\` | the licenses of everything bundled and `THIRD-PARTY-NOTICES.txt`; Qt (PySide6) is used under the LGPL-3.0 — its text and the GNU GPL's are in `licenses\Qt`, its libraries are the `Qt6*.dll` files in `_internal\PySide6`, and the notices say where its source is |
| `_internal\` | Python, PyTorch, Qt and AuraLite |

`smoke_app.py` checks a build the way a user meets it — unzipped into a folder
whose path has a space and Cyrillic letters, with the app's data, temporary files and
the files it trains on under a user name in Cyrillic letters, as `C:\Users\<name>` is
on many Russian Windows computers (2.21.1): `auralite.exe --version`,
`auralite.exe doctor --studio`, the Studio's own program with `--self-test`, a
real start of `AuraLite Studio.exe` with no console that closes itself after a few
seconds (`AURALITE_STUDIO_SMOKE_SECONDS`), a training run with `auralite.exe
train`, and `auralite.exe data prepare --workers 2` (worker processes of a frozen
program). CI's `windows-app` job builds and checks the app on every push and keeps
the zip for a week as the run's artifact; the Release workflow builds it from the
tag, checks it the same way and attaches it to the release
([docs/releasing.md](releasing.md)). A failed check also leaves an annotation with
the end of its output on the run's page.
