@echo off
echo ========================================
echo DANTE Alloy Design - Quick Start
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

echo Python is available!
echo.

echo Starting DANTE Alloy Design notebook...
python start_notebook.py

echo.
echo Press any key to exit...
pause >nul
