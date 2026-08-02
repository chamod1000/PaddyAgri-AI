# Developer Guide & Extension Manual

## 1. Project Organization & Subsystems

```
+-------------------------------------------------------------------+
|                        PROJECT WORKSPACE                          |
+-------------------------------------------------------------------+
├── config/                # LLM Provider configuration (Groq / Gemini)
├── core/                  # Core Business & Multi-Agent Engines
│   ├── agents/            # Specialist Agents (Router, Diag, Fert, Refl, Synth)
│   ├── agent_messages.py  # Strongly-typed A2A Pydantic DTO models
│   ├── agent_orchestrator.py # Multi-Agent Swarm Orchestrator
│   ├── analytics.py       # Deterministic Analytics Engine
│   ├── case_manager.py    # Case Management & Storage Repository
│   ├── evaluation.py     # Post-Execution Quality Scoring Framework
│   ├── explainability.py  # Explainable AI (XAI) Evidence Engine
│   ├── knowledge_center.py# Agricultural Reference Library Repository
│   ├── observability.py   # System Telemetry & Tracing Engine
│   ├── report_generator.py# 10-Section PDF Generation Engine
│   ├── vision_processor.py# Multimodal Visual Observation Layer
│   └── weather_service.py # Weather Context & SeasonAdvisor Rules
├── rag/                   # FAISS Vector Search & PDF Ingestion Pipeline
├── tools/                 # Vector Search Tools & Global Caching
├── ui/                    # Streamlit Presentation Frontend
├── tests/                 # Automated Unittest Suite (20 Test Cases)
└── docs/                  # Comprehensive Final Year Project Manuals
```

---

## 2. Developer Workflow & Extension Rules

### 2.1 Adding a New Swarm Agent
1. Create `core/agents/your_agent.py` inheriting standard agent structure.
2. Define Pydantic request/response payload in `core/agent_messages.py`.
3. Register your agent in `core/agent_orchestrator.py` inside `__init__()`.
4. Update `RouterAgent` intent classifications if necessary.

### 2.2 Swapping Storage Backends (Repository Pattern)
To replace JSON file storage with SQLite or a Cloud DB:
1. Create `SQLiteCaseRepository(ICaseRepository)` in `core/case_manager.py`.
2. Implement `save_case()`, `get_case()`, `list_cases()`, and `search_cases()`.
3. Pass your new repository instance into `CaseManager(repository=SQLiteCaseRepository())`.

### 2.3 Adding New Weather Providers
1. Inherit from `IWeatherProvider` in `core/weather_service.py`.
2. Implement `get_weather(location)` returning `WeatherContext`.
3. Pass provider into `WeatherService(provider=YourWeatherProvider())`.

### 2.4 Adding New Agricultural Knowledge Articles
Add articles to `InMemoryKnowledgeRepository` or connect an external CMS in `core/knowledge_center.py`.

---

## 3. Coding Standards & Guidelines

- **Type Annotations**: All public class attributes and function arguments must use explicit type hints (`Optional[...]`, `List[...]`, `Dict[...]`).
- **Zero Silent Exception Policy**: Always handle exceptions explicitly or log detailed diagnostic notes.
- **Layer Isolation**: The UI layer (`ui/app.py`) must consume structured DTOs returned by `PaddyAgentOrchestrator` without invoking backend models directly.
