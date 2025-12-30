#!/bin/bash
# Script to activate virtual environment and run the backend server
# Usage: ./run_server.sh

echo "Checking for virtual environment..."

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one now..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment"
        exit 1
    fi
    echo "Virtual environment created successfully"
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Starting the backend server..."
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port ${PORT:-8000}