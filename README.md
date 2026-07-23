# Multi-Agent Paddy Disease Diagnostic and Fertilizer Recommendation System for Sri Lankan Farmers

**Module:** IT41043 - Agentic AI (Horizon Campus)  
**Branch:** `feature/agent-orchestration`

---

## 📌 Project Overview
An intelligent multi-agent system designed to assist Sri Lankan farmers and agricultural officers in diagnosing paddy plant diseases, recommending optimal fertilizer regimens (suited for Yala and Maha seasons), and retrieving authoritative agricultural knowledge from local policy guidelines, research papers, and technical bulletins.

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

## 🤖 Multi-Agent System Architecture (Step 3 Implementation)

### 1. Agentic Design Patterns Implemented (Mandatory Requirement 4a)

1. **Router Pattern (`agents.py` -> `RouterAgent`)**:
   - Classifies user queries into distinct intent categories (`DISEASE_DIAGNOSIS`, `FERTILIZER_RECOMMENDATION`, `BOTH`, `GENERAL`).
   - Uses a high-speed, low-latency model (`llama-3.1-8b-instant` via Groq) to route requests without latency overhead.

2. **Tool-Use (ReAct) Pattern (`tools.py` & `agents.py`)**:
   - `DiagnosticAgent` calls the `rag_search_tool` to dynamically query the FAISS vector database for relevant research bulletins.
   - `FertilizerAgent` calls `fertilizer_calculator_tool` to compute NPK dosage per acre based on season and region.

3. **Planning & Task Decomposition Pattern (`agent_orchestrator.py`)**:
   - Decomposes complex compound user queries (e.g. combined disease outbreak + fertilizer request) into parallel sub-tasks dispatched to specialized agents, and synthesizes the outputs into a coherent farmer advisory report.

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
 ┌───────────────┐
 │ Orchestrator  │ (Final Synthesis & Report)
 └───────────────┘
```

---

### 3. Model Selection Strategy (Mandatory Requirement 4c)

| Sub-task | Model (Provider) | Why Chosen |
| :--- | :--- | :--- |
| **Intent Routing & Classification** | `llama-3.1-8b-instant` (Groq) | Near-zero latency (~200ms), cost-effective for instant intent detection before deep processing. |
| **Paddy Disease Synthesis** | `claude-3.5-sonnet` (OpenRouter) / `llama-3.3-70b-versatile` (Groq) | High reasoning quality for complex pathology diagnosis and Sinhala-English translation. |
| **Fertilizer Dosage Calculations** | `llama-3.3-70b-versatile` (Groq) | Accurate structured JSON output generation and tabular calculation processing. |

---

## 🛠 RAG Architecture (Step 2 Implementation)

- **Text Splitter:** `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`)
- **Multilingual Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Vector Database:** Local FAISS index (`./faiss_db/`)

---

## 🚀 Setup & Execution Guide

### Prerequisite Installation
```bash
pip install -r requirements.txt
```

### Environment Configuration
Copy `.env.example` to `.env` and add your API keys:
```bash
cp .env.example .env
```

### Running the Multi-Agent System
```bash
python agent_orchestrator.py
```
