#!/bin/bash

# Install Python dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Start the application
echo "Starting application..."
gunicorn --bind=0.0.0.0 --timeout 600 Backend_flask:app
