"""
Multi-Agent Paddy Advisory System - Entry Point Script

Usage:
  python run.py        -> Launches Streamlit Web Interface (ui/app.py)
  python run.py cli    -> Runs CLI evaluation queries
"""

import sys
import os

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "web"
    if mode == "cli":
        from core.agent_orchestrator import run_sample_evaluations
        run_sample_evaluations()
    else:
        print("[INFO] Launching Streamlit Web App from ui/app.py...")
        os.system("python -m streamlit run ui/app.py")
