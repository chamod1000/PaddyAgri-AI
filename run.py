"""
Multi-Agent Paddy Advisory System - Entry Point Script

Usage:
  python run.py        -> Launches Streamlit Web Interface (ui/app.py)
  python run.py cli    -> Runs CLI evaluation queries
"""

import sys
import os

# --- HOTFIX FOR PYTHON 3.14 ANYIO NoEventLoopError ---
# We patch AnyIO BEFORE importing Streamlit so that Uvicorn/Starlette
# inherits this fix globally in the same process.
try:
    import anyio.to_thread
    import asyncio
    async def patched_run_sync(func, *args, abandon_on_cancel=False, **kwargs):
        return await asyncio.to_thread(func, *args)
    anyio.to_thread.run_sync = patched_run_sync
except ImportError:
    pass
# -----------------------------------------------------

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "web"
    if mode == "cli":
        from core.agent_orchestrator import PaddyAgentOrchestrator
        orchestrator = PaddyAgentOrchestrator()
        res = orchestrator.process_user_request(user_query="What are the common symptoms of Paddy Blast disease?")
        print("\n--- CLI Result ---")
        print(getattr(res, "final_synthesis", str(res)))
    else:
        print("[INFO] Launching Streamlit Web App from ui/app.py...")
        os.environ["PYTHONUNBUFFERED"] = "1"
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", "ui/app.py"]
        sys.exit(stcli.main())
