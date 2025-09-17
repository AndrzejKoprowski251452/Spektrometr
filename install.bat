@echo off
echo ================================================
echo Python Spektrometr - Quick Setup for Windows
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

echo Python found: 
python --version

REM Check if we're in the right directory
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found!
    echo Please run this script from the Spektrometr directory
    pause
    exit /b 1
)

echo.
echo Installing Python packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Package installation failed!
    pause
    exit /b 1
)

echo.
echo Creating directories...
if not exist "measurement_data" mkdir measurement_data
if not exist "backup" mkdir backup
if not exist "logs" mkdir logs

echo.
echo ================================================
echo Setup completed successfully!
echo ================================================
echo.
echo To run the application:
echo   python "index copy.py"
echo.
echo Or double-click on: run_spektrometr.bat
echo.
pause