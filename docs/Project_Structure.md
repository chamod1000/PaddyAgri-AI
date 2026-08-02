# Project Folder Structure & File Responsibilities

## 1. Directory Tree Overview

```
Multi-Agent Paddy Disease Diagnostic and Fertilizer Recommendation System/
├── config/                  # Provider & Model Settings
│   └── model_provider.py    # LLM Initialization (Groq Llama 3.3 / Gemini)
├── core/                    # Core System Architecture & Engines
│   ├── agents/              # Swarm Agents Directory
│   │   ├── __init__.py
│   │   ├── router_agent.py  # Query Intent Classification
│   │   ├── diagnostic_agent.py # Tool-Use Pathology Search
│   │   ├── fertilizer_agent.py # NPK Dosage Calculation
│   │   ├── reflection_agent.py # Safety & Regulatory Critique
│   │   └── synthesis_agent.py  # Response Streaming Synthesizer
│   ├── agent_messages.py    # Typed A2A Pydantic Payload Models
│   ├── agent_orchestrator.py# Main Swarm Orchestrator
│   ├── analytics.py         # Deterministic Analytics Engine
│   ├── case_manager.py      # Case Management & Storage Repository
│   ├── evaluation.py       # Quality Benchmarking Evaluator
│   ├── explainability.py    # Explainable AI (XAI) Evidence Engine
│   ├── knowledge_center.py  # DOA Reference Library Repository
│   ├── observability.py     # Telemetry & Request Tracing
│   ├── report_generator.py  # 10-Section PDF Generator Engine
│   ├── vision_processor.py  # Multimodal Visual Observation Layer
│   └── weather_service.py   # Microclimate & SeasonAdvisor Rules
├── Data/                    # Persistent Storage Repositories
│   └── Cases/               # JSON Case Records (case_records.json)
├── docs/                    # Final Year Project Documentation
│   ├── Architecture.md      # Layered System Architecture Manual
│   ├── Developer_Guide.md   # Extension & Developer Workflow Manual
│   ├── User_Manual.md       # Farmer Operating Guide
│   ├── Installation_Guide.md# Installation & Setup Guide
│   ├── Testing_Report.md    # Quality Assurance & Testing Report
│   └── Project_Structure.md # File & Folder Responsibility Manual
├── faiss_db/                # FAISS Vector Index & Storage
├── rag/                     # Vector Ingestion & Embeddings Pipeline
├── tests/                   # Automated Unittest Suite (20 Tests)
└── ui/                     # Presentation Layer
    └── app.py               # Streamlit Frontend Application
```

---

## 2. File Responsibilities Summary

- **`core/agent_orchestrator.py`**: Coordinates multi-agent workflow execution, memory, observability, evaluation, case management, weather context, and XAI explanations.
- **`core/agent_messages.py`**: Defines strongly-typed Pydantic DTO models (`AgentResponse`, `ProcessingContext`, `AgentMessage`).
- **`core/report_generator.py`**: Implements 10-section PDF generator consuming existing structured objects directly.
- **`core/case_manager.py`**: Implements repository abstraction (`ICaseRepository` / `JSONCaseRepository`) for diagnosis tracking.
- **`core/analytics.py`**: Aggregates platform usage and quality metrics deterministically.
- **`core/weather_service.py`**: Ingests weather context and evaluates 0-LLM agricultural rules (`SeasonAdvisor`).
- **`core/explainability.py`**: Maps diagnostic findings to traceable visual and microclimate evidence items.
- **`core/knowledge_center.py`**: Manages the Department of Agriculture (DOA) reference library.
- **`ui/app.py`**: Provides modern Streamlit presentation user interface.
