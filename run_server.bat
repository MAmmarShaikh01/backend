@echo off
REM Script to activate virtual environment and run the backend server
REM Usage: run_server.bat

echo Checking for virtual environment...

if not exist "venv" (
    echo Virtual environment not found. Creating one now...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment
        exit /b 1
    )
    echo Virtual environment created successfully
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo Starting the backend server...
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port %PORT%