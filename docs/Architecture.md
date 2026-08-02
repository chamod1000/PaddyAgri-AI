# System Architecture & Technical Specifications

## 1. Executive Summary & System Overview

The **Multi-Agent Paddy Disease Diagnostic and Fertilizer Recommendation System** is an enterprise-grade artificial intelligence platform designed for Sri Lankan paddy farmers. The platform integrates autonomous multi-agent swarm orchestration, computer vision leaf pathology analysis, vector-based Retrieval-Augmented Generation (RAG), microclimate weather intelligence, deterministic case management, explainable AI (XAI) evidence mapping, and dynamic PDF report generation.

---

## 2. Layered System Architecture

```
+-----------------------------------------------------------------------+
|                         PRESENTATION LAYER                            |
|                  Streamlit UI Frontend (ui/app.py)                   |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                         ORCHESTRATION LAYER                           |
|              PaddyAgentOrchestrator (core/agent_orchestrator.py)       |
+-----------------------------------------------------------------------+
          |                        |                        |
          v                        v                        v
+-------------------+    +-------------------+    +-------------------+
|  MULTI-AGENT SWARM|    |  CONTEXT & MEMORY |    |  ANCILLARY ENGINE |
|  - RouterAgent    |    |  - ProcessingCtx  |    |  - WeatherService |
|  - DiagnosticAgent|    |  - ConvMemory     |    |  - SeasonAdvisor  |
|  - FertilizerAgent|    |  - CaseManager    |    |  - Explainability |
|  - ReflectionAgent|    |  - RequestTrace   |    |  - KnowledgeCenter|
|  - SynthesisAgent |    |  - Evaluator      |    |  - ReportGenerator|
+-------------------+    +-------------------+    +-------------------+
          |                                                 |
          v                                                 v
+------------------------------------+    +-----------------------------+
|          KNOWLEDGE & RAG           |    |     STORAGE & PERSISTENCE   |
|  - FAISS Vector Index (faiss_db)   |    |  - JSONCaseRepository       |
|  - Multilingual MiniLM Embeddings  |    |  - Data/Cases/records.json  |
|  - DOA Regulatory Guidelines       |    |  - Knowledge Repository     |
+------------------------------------+    +-----------------------------+
```

---

## 3. Core Engine Workflows

### 3.1 Multi-Agent Reasoning Swarm
1. **RouterAgent**: Classifies incoming queries into `DISEASE_DIAGNOSIS`, `FERTILIZER_CALCULATION`, `PEST_IDENTIFICATION`, or `GENERAL`.
2. **DiagnosticAgent**: Executes tool-use RAG search over FAISS database to identify paddy diseases and treatments.
3. **FertilizerAgent**: Computes per-acre NPK dosage (Urea, TSP, MOP) for Yala/Maha cultivation seasons.
4. **ReflectionAgent**: Evaluates recommendations against Sri Lanka Pesticide Act No. 33 safety guidelines.
5. **SynthesisAgent**: Streams real-time response payload to the presentation layer.

### 3.2 Multimodal Vision Layer (`core/vision_processor.py`)
Provides non-invasive visual observation extraction (symptoms, leaf color, spot morphology) without bypassing agent orchestration.

### 3.3 Weather & Seasonal Intelligence (`core/weather_service.py`)
Ingests ambient environmental conditions (temperature, humidity, rainfall forecast) and applies deterministic agricultural rules (`SeasonAdvisor`).

### 3.4 Explainable AI Engine (`core/explainability.py`)
Maps diagnostic findings to visual lesion features and microclimate indicators without making additional LLM calls.

### 3.5 Case Management & Persistence (`core/case_manager.py`)
Uses the Repository Pattern (`ICaseRepository` / `JSONCaseRepository`) to persist diagnostic snapshots and follow-up progression records.

---

## 4. Class Relationships & Design Patterns

- **Repository Pattern**: `ICaseRepository` and `IKnowledgeRepository` provide abstract storage interfaces.
- **Router Pattern**: `RouterAgent` dynamically dispatches queries to specialist agents.
- **Tool-Use Pattern**: `DiagnosticAgent` and `FertilizerAgent` invoke vector search tools.
- **Reflection Pattern**: `ReflectionAgent` validates safety and regulatory compliance.
