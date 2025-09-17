#!/bin/bash

VENV_DIR="venv"

echo "Starting the AI K8s Chat Manager locally..."

# Check if virtual environment exists, if not, create it.
if [ ! -d "$VENV_DIR" ]; then
    echo "Python virtual environment ('$VENV_DIR') not found. Creating one..."
    python3 -m venv $VENV_DIR
    echo "Virtual environment created."
fi

# Activate the virtual environment
echo "Activating Python virtual environment..."
source $VENV_DIR/bin/activate

echo "Make sure you have created a .env file from the .env.example template."

echo "Installing/updating dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Launching application on http://0.0.0.0:8080"
echo "You can access it from your browser via your EC2 instance's public IP."

# Using 0.0.0.0 makes the app accessible from outside the EC2 instance
uvicorn main:app --host 0.0.0.0 --port 8080 --reload