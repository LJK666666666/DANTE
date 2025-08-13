#!/bin/bash

echo "========================================"
echo "DANTE Alloy Design - Quick Start"
echo "========================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python is not installed or not in PATH"
        echo "Please install Python and try again."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "Python is available: $($PYTHON_CMD --version)"
echo

echo "Starting DANTE Alloy Design notebook..."
$PYTHON_CMD start_notebook.py

echo
echo "Script completed."
