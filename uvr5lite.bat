@echo off
   setlocal
   chcp 65001 >nul
   set PYTHONUTF8=1

set "ROOT=%~dp0"
set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    call "%ROOT%setup_uvr5lite.bat"
    if errorlevel 1 exit /b %errorlevel%
)

"%PYTHON_EXE%" "%ROOT%uvr5lite.py" %*
exit /b %errorlevel%
