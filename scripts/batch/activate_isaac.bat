@echo off
REM Get the directory of this script
set SCRIPT_DIR=%~dp0
REM Assume .env is one level up
set ENV_FILE=%SCRIPT_DIR%..\.env

REM Check if .env exists
if not exist "%ENV_FILE%" (
    echo .env file not found at "%ENV_FILE%"
    exit /b 1
)

REM Load ISAACLAB_HOME from .env
for /f "tokens=1,2 delims==" %%A in ('type "%ENV_FILE%" ^| findstr /i "^ISAACLAB_HOME="') do (
    set ISAACLAB_HOME=%%B
)

REM Remove surrounding quotes if any
set ISAACLAB_HOME=%ISAACLAB_HOME:"=%

REM Check if ISAACLAB_HOME is set
if "%ISAACLAB_HOME%"=="" (
    echo ISAACLAB_HOME not found in .env
    exit /b 1
)

REM Activate the virtual environment
call "%ISAACLAB_HOME%\.venv\Scripts\activate.bat"

echo Activated virtual environment at %ISAACLAB_HOME%\.venv
