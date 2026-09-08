@echo off
   setlocal
   chcp 65001 >nul
   set PYTHONUTF8=1

set "ROOT=%~dp0"
set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [uvr5lite] Creating virtual environment...
    py -3 -m venv "%ROOT%.venv"
    if errorlevel 1 (
        echo [uvr5lite] Failed to create virtual environment with py -3.
        exit /b 1
    )
)

echo [uvr5lite] Installing dependencies...
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

"%PYTHON_EXE%" -m pip install -r "%ROOT%requirements.txt"
if errorlevel 1 exit /b 1

echo [uvr5lite] Setup complete.
exit /b 0
