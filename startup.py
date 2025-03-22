import os
import subprocess
import sys
import time

# Log function
def log(message):
    print(message, flush=True)

log("Starting custom startup script...")

# Make sure pip is available
log("Ensuring pip is installed...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "--version"])
except:
    log("Installing pip...")
    subprocess.check_call([sys.executable, "-m", "ensurepip"])

# Install dependencies explicitly
log("Installing Google Generative AI...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai==0.3.1"])

log("Installing other dependencies...")
dependencies = [
    "flask",
    "flask-cors",
    "chromadb",
    "nltk",
    "spacy",
    "numpy",
    "pdfplumber",
    "langchain",
    "langchain-huggingface",
    "langchain-chroma",
    "scikit-learn",
    "transformers",
    "datasets",
    "textblob",
    "gunicorn"
]

for dep in dependencies:
    log(f"Installing {dep}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
    except Exception as e:
        log(f"Error installing {dep}: {str(e)}")

# List installed packages
log("Listing installed packages:")
subprocess.check_call([sys.executable, "-m", "pip", "list"])

# Start the application using gunicorn
log("Starting application with gunicorn...")
os.system("gunicorn --bind=0.0.0.0 --timeout 600 Backend_flask:app")
