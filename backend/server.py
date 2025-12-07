"""
Backend redirect for Battery RUL Prediction System
The actual application runs as a Streamlit app on port 8501
"""
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import threading
import time
import os

app = FastAPI(title="Battery RUL Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to track Streamlit process
streamlit_process = None

def start_streamlit():
    """Start Streamlit in background"""
    global streamlit_process
    import sys
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", "/app/app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    streamlit_process = subprocess.Popen(streamlit_cmd)
    return streamlit_process

# Start Streamlit on import
streamlit_thread = threading.Thread(target=start_streamlit, daemon=True)
streamlit_thread.start()
time.sleep(2)

@app.get("/")
async def root():
    return {"message": "Battery RUL Prediction System", "streamlit_url": "http://localhost:8501"}

@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "battery-rul-prediction"}

@app.get("/api/status")
async def status():
    return {
        "status": "running",
        "streamlit_port": 8501,
        "models": ["XGBoost", "Random Forest", "Linear Regression", "LSTM"]
    }
