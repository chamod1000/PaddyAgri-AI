# Multi-Agent Paddy Disease Diagnostic and Fertilizer Recommendation System for Sri Lankan Farmers

**Module:** IT41043 - Agentic AI (Horizon Campus)  
**Branches:** `master` | `feature/streamlit-ui` | `feature/agent-orchestration` | `feature/rag-pipeline`

---

## 📌 Project Overview
An intelligent multi-agent system designed to assist Sri Lankan farmers and agricultural extension officers in diagnosing paddy plant diseases, recommending optimal fertilizer regimens (suited for Yala and Maha seasons), and retrieving authoritative agricultural knowledge from local policy guidelines, research papers, and technical bulletins.

---

## 📂 Dataset & Knowledge Base Access

The domain knowledge base comprises **20+ PDF documents** (disease manuals, seed acts, fertilizer recommendations, pest control bulletins) organized into 6 logical subdirectories.

* 🔗 **Google Drive Full Dataset Link:** [Sri Lankan Paddy Farming PDF Corpus](https://drive.google.com/drive/folders/1O6Teo6_gPBZOd27rtzAI84RSTKKU5er9?usp=sharing)
* 📁 **GitHub Repository Structure (`Data/PDF/`):**
  - `Fertilizer_and_Chemicals`: Soil nutrient recommendations, NPK ratios, pesticide/weedicide regulations.
  - `General_Cultivation_Guidelines`: Seasonal planting guidelines (Yala/Maha), water management, land preparation.
  - `Imports_Quarantine`: Import restrictions, quarantine acts, biosecurity protocols.
  - `Paddy_Diseases_and_Pests`: Detailed diagnostic manuals for Paddy Blast, Sheath Blight, Brown Planthopper, etc.
  - `Paddy_Seed_Production`: Certification criteria, seed purity rates, germination standards.
  - `Policy_and_Acts`: National agricultural acts, seed acts, environmental regulations.

---

## 🤖 Multi-Agent System Architecture

### 1. Agentic Design Patterns Implemented (Mandatory Requirement 4a)

1. **Router Pattern (`agents.py` -> `RouterAgent`)**:
   - Classifies user queries into distinct intent categories (`DISEASE_DIAGNOSIS`, `FERTILIZER_RECOMMENDATION`, `BOTH`, `GENERAL`).
   - Uses a high-speed, low-latency model (`llama-3.1-8b-instant` via Groq) to route requests without latency overhead.

2. **Tool-Use (ReAct) Pattern (`tools.py` & `agents.py`)**:
   - `DiagnosticAgent` calls the `rag_search_tool` to dynamically query the FAISS vector database for relevant research bulletins.
   - `FertilizerAgent` calls `fertilizer_calculator_tool` to compute NPK dosage per acre based on season and region.

3. **Planning & Task Decomposition Pattern (`agent_orchestrator.py`)**:
   - Decomposes complex compound user queries (e.g. combined disease outbreak + fertilizer request) into parallel sub-tasks dispatched to specialized agents.

4. **Reflection & Self-Critique Pattern (`agents.py` -> `ReflectionAgent`)**:
   - Acts as a Safety & Quality Verifier that double-checks recommended pesticide/fertilizer recommendations against Sri Lankan Department of Agriculture (DoA) environmental safety guidelines and biosecurity regulations before final output synthesis.

---

### 2. Agent-to-Agent Communication Protocol (Mandatory Requirement 4b)

Agents communicate using typed `AgentMessage` objects structured as follows:

```json
{
  "message_id": "msg_8a12b3_diag",
  "sender": "RouterAgent",
  "receiver": "DiagnosticAgent",
  "intent": "DISEASE_DIAGNOSIS",
  "user_query": "ගොයම් පත්‍ර වල දුඹුරු පැහැ ලප ඇති වී ඇත",
  "payload": {"task": "disease_diagnosis"},
  "timestamp": "2026-07-23T01:50:00"
}
```

#### 🔄 Message Flow Diagram
```
    Farmer Query
         │
         ▼
 ┌───────────────┐
 │  RouterAgent  │ (Fast Classification)
 └───────┬───────┘
         │
  ───────┴─────────────────────────
 │                                 │
 ▼                                 ▼
┌──────────────────┐     ┌───────────────────┐
│ DiagnosticAgent  │     │  FertilizerAgent  │
│  (ReAct + RAG)   │     │  (Calculator Tool)│
└────────┬─────────┘     └─────────┬─────────┘
         │                         │
  ───────┴─────────────────────────
         │
         ▼
 ┌──────────────────┐
 │ ReflectionAgent  │ (Safety & DoA Regulatory Verification)
 └───────┬──────────┘
         │
         ▼
 ┌───────────────┐
 │ Orchestrator  │ (Final Synthesis & Report)
 └───────────────┘
```

---

### 3. Model Selection Strategy (Mandatory Requirement 4c)

| Sub-task | Model (Provider) | Latency | Cost / 1M Tokens | Context Window | Reasoning Quality |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intent Routing & Classification** | `llama-3.1-8b-instant` (Groq) | ~200 ms | $0.05 | 128k | Near-free, sufficient for rapid intent classification. |
| **Paddy Disease Synthesis** | `llama-3.3-70b-versatile` (Groq) | ~1.2 s | $0.59 | 128k | Exceptional reasoning quality for complex pathology & Sinhala translation. |
| **Fertilizer Dosage Calculations** | `llama-3.3-70b-versatile` (Groq) | ~600 ms | $0.59 | 128k | High precision structured output generation and tabular calculation processing. |
| **Sinhala/Singlish Language Queries** | `gemini-2.0-flash` (Google AI) | ~800 ms | $0.10 | 1M | Native multilingual fluency for Sinhala script and transliterated Singlish advisory reports. |
| **General Agricultural Q&A (RAG)** | `command-r-plus-08-2024` (Cohere) | ~1.0 s | $2.50 | 128k | Purpose-built for retrieval-augmented generation with excellent citation accuracy. |

---

## 🛠 RAG Architecture (Step 2 Implementation)

- **Text Splitter:** `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`)
- **Multilingual Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Vector Database:** Local FAISS index (`./faiss_db/`)

### 📊 RAG Retrieval Evaluation (5 Sample Queries - Requirement 4d)

| Query # | Test Query | Top Retrieved Source | Page # | Similarity Score | Context Relevance Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1** | *"What are the symptoms and fungicide treatments for Paddy Blast?"* | `Paddy Blast Disease Management.pdf` | Page 4 | 0.88 | **Highly Relevant** — Contains exact symptoms (diamond lesions) & Tricyclazole recommendations. |
| **Q2** | *"What is the recommended Urea, TSP, and MOP fertilizer rate for Yala season?"* | `Fertilizer Recommendation Guide.pdf` | Page 12 | 0.84 | **Highly Relevant** — Specifies exact NPK dosages per acre for Yala dry zone cultivation. |
| **Q3** | *"How to identify Brown Planthopper (BPH) hopper burn in paddy fields?"* | `Insect Pests of Rice in Sri Lanka.pdf` | Page 19 | 0.86 | **Highly Relevant** — Correctly retrieved BPH infestation signs, draining techniques & insecticide limits. |
| **Q4** | *"ශ්‍රී ලංකාවේ සහතික කළ බිත්තර වී සඳහා පැළවීමේ ප්‍රතිශතය සහ ප්‍රමිතීන් මොනවාද?"* | `Seed Act No 22 of 2003 Guidelines.pdf` | Page 8 | 0.89 | **Highly Relevant** — Retrieved exact 85% min germination rate & 98% seed purity standard under SCS. |
| **Q5** | *"What are the soil conservation regulations for sloping paddy lands?"* | `Soil Conservation Act Regulations.pdf` | Page 15 | 0.81 | **Highly Relevant** — Extracted bund terracing rules and erosion prevention directives. |

---

## 💻 Streamlit Web Application & Deployment (Step 4 Implementation)

- **Interactive UI (`app.py`)**: Includes Quick Sample Queries (Paddy Blast, Yala Fertilizer, Sinhala Query, Seed Paddy Purity).
- **Live Agent Message Trace Panel**: Displays real-time JSON message exchange between agents.
- **RAG Citation & Source Quote Highlighting**: Shows exact PDF source, page number, and quote for diagnostic evidence.
- **Downloadable Advisory Report**: Exports generated farmer advisory reports as text/PDF files.

---

## 🚀 Setup & Execution Guide

### 1. Prerequisite Installation
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env` and add your API keys:
```bash
cp .env.example .env
```

### 3. Running the Streamlit Web App
```bash
streamlit run ui/app.py
# OR using the main entry point:
python run.py
```

### 4. Running the Multi-Agent System in CLI
```bash
python run.py cli
# OR directly:
python -c "from core.agent_orchestrator import run_sample_evaluations; run_sample_evaluations()"
```
