@echo off
cd /d "%~dp0"
chcp 65001 >nul
title AuraLite AI v2.0 Builder
echo ====================================================
echo    Building AuraLite AI v2.0 — Modern Edition
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
python -m PyInstaller --onedir --noconsole --name "AuraLite_AI_v2" gui_app.py

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
