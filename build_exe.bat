@echo off
cd /d "%~dp0"
chcp 65001 >nul
title AuraLite AI Builder
echo ====================================================
echo    Building AuraLite AI CUDA Edition
echo ====================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Error: Python is not installed.
    pause
    exit /b
)

echo [+] Installing Heavy Dependencies (PyTorch, NumPy, PyInstaller)...
echo This might take some time as PyTorch is large...
python -m pip install torch numpy pyinstaller

echo.
echo [+] Starting compilation...
python -m PyInstaller --onefile --noconsole --name "AuraLite_AI_CUDA" gui_app.py

if %errorlevel% equ 0 (
    echo.
    echo ====================================================
    echo [OK] Build successful!
    echo Your file is in: \dist\AuraLite_AI_CUDA.exe
    echo ====================================================
) else (
    echo.
    echo [!] Build failed.
)

pause >nul
