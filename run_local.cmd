@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if exist "%VENV_PY%" goto verify_venv

echo Preparing local Python environment...
set "PYTHON_CMD=py -3.12"
%PYTHON_CMD% -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
if "%ERRORLEVEL%"=="0" goto verify_python

set "PYTHON_CMD=python"
%PYTHON_CMD% -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
if "%ERRORLEVEL%"=="0" goto verify_python

python --version >nul 2>nul
if "%ERRORLEVEL%"=="0" goto wrong_python
goto missing_python

:verify_python
%PYTHON_CMD% -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
if errorlevel 1 goto wrong_python
%PYTHON_CMD% -m venv .venv
if errorlevel 1 goto setup_failed
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto setup_failed
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto setup_failed
goto run

:verify_venv
"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
if errorlevel 1 goto wrong_python
goto run

:run
"%VENV_PY%" "%~dp0local_app.py" %*
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" (
    echo Local app exited with code %APP_EXIT%.
    pause
)
exit /b %APP_EXIT%

:missing_python
echo Python 3.12 was not found. Install Python 3.12, then run this file again.
pause
exit /b 1

:wrong_python
echo Python was found, but this app requires Python 3.12.
pause
exit /b 1

:setup_failed
echo Failed to prepare the local Python environment.
pause
exit /b 1
