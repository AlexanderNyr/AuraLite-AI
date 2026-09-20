@echo off
cd /d "%~dp0"
title AuraLite AI v2.6 Builder
echo ====================================================
echo    Building AuraLite AI v2.6
echo ====================================================
echo.

REM --- Locate Python ---------------------------------------------------------
set "PYTHON_EXE="
where python >nul 2>&1
if %errorlevel% equ 0 set "PYTHON_EXE=python"
if not defined PYTHON_EXE (
    where py >nul 2>&1
    if %errorlevel% equ 0 set "PYTHON_EXE=py -3"
)
if not defined PYTHON_EXE (
    echo [!] Error: Python is not in PATH. Install Python 3.10-3.13 from python.org
    echo     and check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
echo [+] Using Python: %PYTHON_EXE%
%PYTHON_EXE% --version

REM --- Create / reuse clean virtual environment ------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [+] Creating clean virtual environment in .venv ...
    %PYTHON_EXE% -m venv .venv
    if errorlevel 1 (
        echo [!] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo [+] Activating .venv ...
call ".venv\Scripts\activate.bat"
set "PYTHON_EXE=.venv\Scripts\python.exe"

echo [+] Upgrading pip ...
"%PYTHON_EXE%" -m pip install --upgrade pip --disable-pip-version-check

echo [+] Installing minimal runtime dependencies into .venv ...
"%PYTHON_EXE%" -m pip install torch numpy matplotlib pyinstaller --disable-pip-version-check
REM Uncomment the next line if you need HuggingFace / serving / RAG in build:
REM "%PYTHON_EXE%" -m pip install transformers peft accelerate sentencepiece protobuf tiktoken fastapi uvicorn pydantic --disable-pip-version-check

echo.
echo [+] Cleaning previous build artifacts ...
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist
if exist "AuraLite_AI_v2.spec" del /q AuraLite_AI_v2.spec

echo [+] Starting compilation (--onedir) ...
"%PYTHON_EXE%" -m PyInstaller ^
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

if errorlevel 1 (
    echo.
    echo ====================================================
    echo [!] Build FAILED. See errors above.
    echo ====================================================
) else (
    echo.
    echo ====================================================
    echo [OK] Build SUCCESS!
    echo Output folder: dist\AuraLite_AI_v2\
    echo Run: dist\AuraLite_AI_v2\AuraLite_AI_v2.exe
    echo ====================================================
)

echo.
echo Re-run this script anytime to rebuild (.venv is reused).
pause >nul
