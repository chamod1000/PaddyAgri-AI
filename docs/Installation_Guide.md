# Environment Setup & Installation Guide

## 1. System Requirements

- **Operating System**: Windows 10/11, macOS 12+, or Linux (Ubuntu 20.04+)
- **Python Version**: Python 3.10, 3.11, or 3.12 (Python 3.14 compatible)
- **RAM**: Minimum 8 GB (16 GB recommended)
- **Disk Space**: 2 GB free disk space

---

## 2. Step-by-Step Installation Procedure

### Step 1: Clone Repository & Open Directory
```bash
git clone https://github.com/chamod1000/PaddyAgri-AI.git
cd "Multi-Agent Paddy Disease Diagnostic and Fertilizer Recommendation System for Sri Lankan Farmers"
```

### Step 2: Create & Activate Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure API Credentials (`.env`)
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_huggingface_token_optional
```

---

## 3. Launching the Application

Run the Streamlit presentation server:
```bash
streamlit run ui/app.py
```
The application will automatically launch in your browser at `http://localhost:8501`.

---

## 4. Troubleshooting Guide

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'fpdf'` | Missing PDF generator package | Run `pip install fpdf2` |
| `FAISS index not found` | Missing vector database | Run `python rag/ingest.py` to rebuild index |
| `Groq API Error / Rate Limit` | Invalid or expired API key | Update `GROQ_API_KEY` in `.env` file |
