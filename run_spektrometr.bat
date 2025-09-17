@echo off
echo ================================================
echo Python Spektrometr - Starting Application
echo ================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python or run install.bat first
    pause
    exit /b 1
)

REM Check if main file exists
if not exist "index.py" (
    echo ERROR: Application file not found!
    echo Make sure you're in the correct directory
    pause
    exit /b 1
)

REM Run the application
echo Starting Python Spektrometr...
python index.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Application exited with error
    pause
)