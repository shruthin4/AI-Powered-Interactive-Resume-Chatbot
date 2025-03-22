import os
import subprocess
import sys

# Log function
def log(message):
    print(message, flush=True)

log("Starting custom startup script...")

# Install dependencies
log("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"])

# Install all requirements
log("Installing requirements from requirements.txt...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Start the application using gunicorn
log("Starting application with gunicorn...")
os.system("gunicorn --bind=0.0.0.0 --timeout 600 Backend_flask:app")
