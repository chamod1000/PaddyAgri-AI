"""
Streamlit Community Cloud Entry Point (streamlit_app.py)
Redirects execution to ui/app.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

app_path = BASE_DIR / "ui" / "app.py"
with open(app_path, encoding="utf-8") as f:
    code = compile(f.read(), str(app_path), "exec")
    exec(code, globals())
