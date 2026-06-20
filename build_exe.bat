@echo off
cd /d "%~dp0"
chcp 65001 >nul
title AuraLite AI v2.4 Builder
echo ====================================================
echo    Building AuraLite AI v2.4 — Modern Edition
echo ====================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Error: Python is not installed.
    pause
    exit /b
)

echo [+] Installing dependencies (PyTorch, NumPy, PyInstaller)...
echo This might take some time as PyTorch is large...
python -m pip install torch numpy pyinstaller

echo.
echo [+] Starting compilation (--onedir)...
REM v2.4+: collect package submodules explicitly so frozen builds include
REM model_engine._legacy and the gradual-refactor package shims.
python -m PyInstaller --onedir --noconsole --name "AuraLite_AI_v2" ^
    --hidden-import model_engine._legacy ^
    --collect-submodules model_engine ^
    --collect-submodules gui ^
    --collect-submodules kernels ^
    gui_app.py

if %errorlevel% equ 0 (
    echo.
    echo ====================================================
    echo [OK] Build successful!
    echo Your app folder is in: \dist\AuraLite_AI_v2\
    echo Run: \dist\AuraLite_AI_v2\AuraLite_AI_v2.exe
    echo ====================================================
) else (
    echo.
    echo [!] Build failed.
)

pause >nul
