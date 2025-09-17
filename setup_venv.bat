@echo off
echo ================================================
echo Python Spektrometr - Virtual Environment Setup
echo ================================================

REM Create virtual environment
echo Creating virtual environment...
python -m venv spektrometr_env

if errorlevel 1 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)

echo Activating virtual environment...
call spektrometr_env\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing requirements...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install requirements!
    pause
    exit /b 1
)

echo.
echo ================================================
echo Virtual environment setup complete!
echo ================================================
echo.
echo To activate the environment in the future:
echo   spektrometr_env\Scripts\activate.bat
echo.
echo To run the application:
echo   python "index copy.py"
echo.
echo To deactivate:
echo   deactivate
echo.
pause