import subprocess
import sys
import os
import signal
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def run_fastapi():
    """
    Start FastAPI using uvicorn
    """
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host", "0.0.0.0",
            "--port", "8080",
            "--reload"
        ],
        cwd=PROJECT_ROOT
    )


def run_streamlit():
    """
    Start Streamlit app
    """
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app_st.py"
        ],
        cwd=PROJECT_ROOT
    )


if __name__ == "__main__":
    print("🚀 Starting FastAPI + Streamlit...")
    print("🔹 FastAPI  → http://localhost:8080")
    print("🔹 Streamlit → http://localhost:8501\n")

    fastapi_proc = None
    streamlit_proc = None

    try:
        # Start FastAPI first
        fastapi_proc = run_fastapi()
        time.sleep(3)  # allow server to bind port

        # Start Streamlit second
        streamlit_proc = run_streamlit()

        # Wait until processes exit
        fastapi_proc.wait()
        streamlit_proc.wait()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")

    finally:
        for proc in [fastapi_proc, streamlit_proc]:
            if proc and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)

        print("✅ Shutdown complete.")
