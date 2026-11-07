@echo off
REM Build the AuraLite app that needs no Python (2.21) - the same build as the Windows zip
REM the release attaches: dist\app\AuraLite-Studio-<version>\ (AuraLite Studio.exe and
REM auralite.exe) and dist\AuraLite-Studio-<version>-windows-x64.zip.
REM Steps and details: docs\windows-app.md ("Building it yourself").
setlocal
cd /d "%~dp0"
title AuraLite app builder
echo ====================================================
echo    Building the AuraLite app (Windows, no Python)
echo ====================================================

REM --- Python: 3.13 like the release, else the one in PATH ------------------
set "PYTHON_EXE="
py -3.13 --version >nul 2>&1
if not errorlevel 1 set "PYTHON_EXE=py -3.13"
if not defined PYTHON_EXE (
    python --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=python"
)
if not defined PYTHON_EXE (
    echo [!] Python is not installed. Install Python 3.13 from python.org
    echo     and tick "Add python.exe to PATH".
    if not defined CI pause
    exit /b 1
)
echo [+] Python: %PYTHON_EXE%

REM --- A virtual environment of its own, with the build's pinned packages ---
set "VENV_PY=.venv-app\Scripts\python.exe"
set "OK=1"
if not exist "%VENV_PY%" (
    echo [+] Creating .venv-app ...
    %PYTHON_EXE% -m venv .venv-app || set "OK=0"
)
if "%OK%"=="1" (
    echo [+] Installing the packages pinned in packaging\app\requirements.txt ...
    "%VENV_PY%" -m pip install --upgrade pip --disable-pip-version-check || set "OK=0"
)
if "%OK%"=="1" (
    "%VENV_PY%" -m pip install -r packaging\app\requirements.txt --disable-pip-version-check || set "OK=0"
)
if "%OK%"=="1" (
    echo [+] Installing AuraLite from this folder ...
    "%VENV_PY%" -m pip install --no-deps --force-reinstall . --disable-pip-version-check || set "OK=0"
)
if "%OK%"=="1" (
    echo [+] Building - a few minutes ...
    "%VENV_PY%" packaging\app\build_app.py --zip || set "OK=0"
)

echo.
if "%OK%"=="1" (
    echo [OK] The app: dist\app\  - start "AuraLite Studio.exe" in its folder.
    echo      The zip:  dist\AuraLite-Studio-*-windows-x64.zip
    echo      Check it: .venv-app\Scripts\python.exe packaging\app\smoke_app.py dist\app\AuraLite-Studio-...
) else (
    echo [!] The build failed - see the messages above.
)
if not defined CI pause
if "%OK%"=="1" (exit /b 0) else (exit /b 1)
