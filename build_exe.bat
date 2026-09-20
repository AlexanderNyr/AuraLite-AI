@echo off
cd /d "%~dp0"
chcp 65001 >nul
title AuraLite AI v2.6 Builder
echo ====================================================
echo    Building AuraLite AI v2.6
echo ====================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Error: Python is not installed.
    pause
    exit /b
)

REM ---- Use a CLEAN virtual environment so PyInstaller does NOT drag
REM      in every package in the global site-packages (scipy/pandas/PyQt/gradio/etc).
if not exist ".venv\Scripts\python.exe" (
    echo [+] Creating clean virtual environment in .venv ...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [!] Failed to create virtual environment.
        pause
        exit /b
    )
)

echo [+] Activating .venv ...
call ".venv\Scripts\activate.bat"

echo [+] Installing minimal runtime dependencies into .venv ...
python -m pip install --upgrade pip
python -m pip install torch numpy matplotlib pyinstaller
REM Optional (uncomment if you want HF/GGUF/serving in the build):
REM python -m pip install transformers peft accelerate sentencepiece protobuf tiktoken fastapi uvicorn pydantic

echo.
echo [+] Cleaning previous build artifacts ...
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist
if exist "AuraLite_AI_v2.spec" del /q AuraLite_AI_v2.spec

echo [+] Starting compilation (--onedir) ...
REM Key fixes:
REM   --exclude-module : keep heavy/unrelated packages OUT of the build so
REM     PyInstaller never imports them (no access-violation from DLL collisions
REM     in the isolated subprocess). Add/remove as you wish.
REM   --collect-submodules : ensure shim packages (model_engine, gui, kernels,
REM     server, agent) are included.
REM   --noconfirm : non-interactive overwrite.
python -m PyInstaller ^
    --onedir ^
    --windowed ^
    --name "AuraLite_AI_v2" ^
    --noconfirm ^
    --hidden-import model_engine._legacy ^
    --collect-submodules model_engine ^
    --collect-submodules gui ^
    --collect-submodules kernels ^
    --collect-submodules server ^
    --collect-submodules agent ^
    --collect-data matplotlib ^
    --exclude-module scipy ^
    --exclude-module pandas ^
    --exclude-module PyQt5 ^
    --exclude-module PyQt6 ^
    --exclude-module PySide2 ^
    --exclude-module PySide6 ^
    --exclude-module gradio ^
    --exclude-module pyglet ^
    --exclude-module pydub ^
    --exclude-module numba ^
    --exclude-module llvmlite ^
    --exclude-module soundfile ^
    --exclude-module sounddevice ^
    --exclude-module lxml ^
    --exclude-module pyarrow ^
    --exclude-module sklearn ^
    --exclude-module skimage ^
    --exclude-module narwhals ^
    --exclude-module babel ^
    --exclude-module rdflib ^
    --exclude-module pytz ^
    --exclude-module imageio ^
    --exclude-module imageio_ffmpeg ^
    --exclude-module IPython ^
    --exclude-module jupyter ^
    --exclude-module notebook ^
    --exclude-module sympy.testing ^
    --exclude-module cv2 ^
    --exclude-module PIL ^
    --exclude-module wx ^
    --exclude-module traits ^
    --exclude-module envisage ^
    --exclude-module mayavi ^
    --exclude-module vtk ^
    --exclude-module torch.utils.tensorboard ^
    --exclude-module torch.distributed.tensor.parallel ^
    gui_app.py

if %errorlevel% equ 0 (
    echo.
    echo ====================================================
    echo [OK] Build successful!
    echo Your app folder is in: dist\AuraLite_AI_v2\
    echo Run: dist\AuraLite_AI_v2\AuraLite_AI_v2.exe
    echo ====================================================
) else (
    echo.
    echo [!] Build failed. See errors above.
)

echo.
echo To rebuild quickly next time just run this script again —
echo the .venv is kept and reused.
pause >nul
