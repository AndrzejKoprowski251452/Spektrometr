@echo off
title Python Spektrometr
echo Starting Python Spektrometr...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python or run install.bat first
    pause
    exit /b 1
)

REM Check if main file exists
if not exist "index copy.py" (
    echo ERROR: Application file not found!
    echo Make sure you're in the correct directory
    pause
    exit /b 1
)

REM Run the application
echo ================================================
echo Python Spektrometr - Starting Application
echo ================================================
python "index copy.py"

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Application exited with error
    pause
)