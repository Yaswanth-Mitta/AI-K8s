#!/bin/bash
set -e

VENV_DIR="venv"

echo "Starting the AI K8s Chat Manager locally..."

# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists, if not, create it.
if [ ! -d "$VENV_DIR" ]; then
    echo "Python virtual environment ('$VENV_DIR') not found. Creating one..."
    python3 -m venv "$VENV_DIR" || { echo "Failed to create virtual environment"; exit 1; }
    echo "Virtual environment created."
fi

# Activate the virtual environment
echo "Activating Python virtual environment..."
source "$VENV_DIR/bin/activate" || { echo "Failed to activate virtual environment"; exit 1; }

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Please create one from .env.example template."
fi

# Check for requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt file not found"
    exit 1
fi

echo "Installing/updating dependencies from requirements.txt..."
pip install -r requirements.txt || { echo "Failed to install dependencies"; exit 1; }

echo ""
echo "Launching application on http://0.0.0.0:8080"
echo "You can access it from your browser via your EC2 instance's public IP."

# Using 0.0.0.0 makes the app accessible from outside the EC2 instance
uvicorn main:app --host 0.0.0.0 --port 8080 --reload || { echo "Failed to start application"; exit 1; }