"""
Root Wrapper for Streamlit UI
Provides backward compatibility so both 'streamlit run app.py' and 'streamlit run ui/app.py' work seamlessly.
"""

import runpy
from pathlib import Path

ui_app_path = Path(__file__).resolve().parent / "ui" / "app.py"
runpy.run_path(str(ui_app_path), run_name="__main__")
