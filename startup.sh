#!/bin/bash
echo "Current directory: $(pwd)"
echo "Listing files:"
ls -la

echo "Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Starting application with Python..."
python -m gunicorn --bind=0.0.0.0 --timeout 600 Backend_flask:app
