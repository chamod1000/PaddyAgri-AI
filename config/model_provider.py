"""
Model Provider Factory & Configuration-Driven Registry Module
Implements Requirement 4c & Tiered Provider Lifecycle Architecture
Connects Google Gemini API, Groq Cloud, Cohere Command R+, and OpenRouter with:
- Tier 0: Primary Model (Constructed at Startup)
- Tier 1: Fast Fallback (Lazy On-Demand / Progressive Construction)
- Tier 2: Emergency Fallback (Lazy On-Demand)
- Tier 3: Rare Fallback (Lazy On-Demand)
- Dynamic Health Tracking & Cooldown Management
- Native LangChain with_fallbacks Integration via LazyProviderRunnable
"""

import os
import time
import threading
import functools
from typing import Optional, Dict, List, Callable, Any
from dotenv import load_dotenv

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig

load_dotenv()

# Streamlit Community Cloud secrets mapping bridge
# Robust version: catches all exceptions, handles flat + nested secrets
def _load_streamlit_secrets() -> None:
    """Maps Streamlit Cloud secrets to os.environ for all providers."""
    try:
        import streamlit as st
        secrets_obj = getattr(st, "secrets", None)
        if secrets_obj is None:
            return
        try:
            items = dict(secrets_obj)
        except Exception:
            return
        loaded = []
        for k, v in items.items():
            if isinstance(v, str):
                if not os.environ.get(k):
                    os.environ[k] = v
                    loaded.append(k)
            elif hasattr(v, "items"):
                # Handle nested TOML sections like [section]\nkey = "val"
                for sub_k, sub_v in v.items():
                    env_key = f"{k}__{sub_k}".upper()
                    if isinstance(sub_v, str) and not os.environ.get(env_key):
                        os.environ[env_key] = sub_v
                        loaded.append(env_key)
        if loaded:
            print(f"[SECRETS] Loaded {len(loaded)} secret(s) from Streamlit Cloud: {loaded}")
    except Exception as _e:
        print(f"[SECRETS] Warning: Could not load Streamlit secrets: {_e}")

_load_streamlit_secrets()


# ══════════════════════════════════════════════
# PROVIDER FACTORY, HEALTH TRACKER & LAZY PROXY
# ══════════════════════════════════════════════

class ProviderHealthState:
    HEALTHY = "HEALTHY"
    COOLDOWN = "COOLDOWN"


class ProviderSpec:
    """
    Metadata specification for an LLM Provider.
    Tracks tiering, priority, lazy factory function, cached instance, and health/cooldown status.
    """
    def __init__(
        self,
        provider_id: str,
        tier: int,  # 0: Primary, 1: Fast Fallback, 2: Emergency, 3: Rare
        priority: int,
        factory: Callable[[], Optional[BaseChatModel]],
        cooldown_seconds: float = 60.0,
        max_failures: int = 2
    ):
        self.provider_id = provider_id
        self.tier = tier
        self.priority = priority
        self.factory = factory
        self.cached_instance: Optional[BaseChatModel] = None

        self.health_status = ProviderHealthState.HEALTHY
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.cooldown_seconds = cooldown_seconds
        self.max_failures = max_failures
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        if self.health_status == ProviderHealthState.COOLDOWN:
            if time.time() - self.last_failure_time > self.cooldown_seconds:
                self.health_status = ProviderHealthState.HEALTHY
                self.failure_count = 0
                return True
            return False
        return True

    def record_success(self):
        with self._lock:
            self.failure_count = 0
            self.health_status = ProviderHealthState.HEALTHY

    def record_failure(self, error: Exception):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.max_failures:
                self.health_status = ProviderHealthState.COOLDOWN
                print(f"[PROVIDER FACTORY] '{self.provider_id}' entered COOLDOWN for {self.cooldown_seconds}s after error: {error}")


class ProviderFactory:
    """
    Central registry responsible for managing ProviderSpec configurations,
    caching constructed LLM instances, and enforcing health tracking.
    """
    _registry: Dict[str, ProviderSpec] = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, spec: ProviderSpec):
        with cls._lock:
            cls._registry[spec.provider_id] = spec

    @classmethod
    def get_or_create(cls, provider_id: str) -> Optional[BaseChatModel]:
        spec = cls._registry.get(provider_id)
        if not spec:
            return None

        if not spec.is_available():
            print(f"[PROVIDER FACTORY] Skipping '{provider_id}' (State: COOLDOWN)")
            return None

        if spec.cached_instance is not None:
            return spec.cached_instance

        with spec._lock:
            if spec.cached_instance is None:
                print(f"[PROVIDER FACTORY] Lazily constructing provider: '{provider_id}' (Tier {spec.tier})", flush=True)
                try:
                    instance = spec.factory()
                    if instance:
                        spec.cached_instance = instance
                    else:
                        spec.record_failure(ValueError("Factory returned None"))
                except Exception as e:
                    spec.record_failure(e)
                    print(f"[PROVIDER FACTORY ERROR] Failed to construct '{provider_id}': {e}", flush=True)
                    return None

        return spec.cached_instance

    @classmethod
    def get_construction_count(cls) -> int:
        return sum(1 for spec in cls._registry.values() if spec.cached_instance is not None)

    @classmethod
    def get_registry_status(cls) -> List[Dict[str, Any]]:
        status = []
        for pid, spec in cls._registry.items():
            status.append({
                "provider_id": pid,
                "tier": spec.tier,
                "priority": spec.priority,
                "is_constructed": spec.cached_instance is not None,
                "health_status": spec.health_status,
                "failure_count": spec.failure_count
            })
        return status


class LazyProviderRunnable(Runnable):
    """
    Native LangChain Runnable proxy for seamless with_fallbacks integration.
    Defers provider client construction until invoke() or stream() is actually called during a fallback event.
    """
    def __init__(self, provider_id: str):
        self.provider_id = provider_id

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs) -> Any:
        model = ProviderFactory.get_or_create(self.provider_id)
        if not model:
            raise RuntimeError(f"Provider '{self.provider_id}' unavailable or in COOLDOWN.")
        spec = ProviderFactory._registry.get(self.provider_id)
        try:
            res = model.invoke(input, config=config, **kwargs)
            if spec: spec.record_success()
            return res
        except Exception as e:
            if spec: spec.record_failure(e)
            raise e

    def stream(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs):
        model = ProviderFactory.get_or_create(self.provider_id)
        if not model:
            raise RuntimeError(f"Provider '{self.provider_id}' unavailable or in COOLDOWN.")
        spec = ProviderFactory._registry.get(self.provider_id)
        try:
            for chunk in model.stream(input, config=config, **kwargs):
                yield chunk
            if spec: spec.record_success()
        except Exception as e:
            if spec: spec.record_failure(e)
            raise e

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs) -> Any:
        model = ProviderFactory.get_or_create(self.provider_id)
        if not model:
            raise RuntimeError(f"Provider '{self.provider_id}' unavailable or in COOLDOWN.")
        spec = ProviderFactory._registry.get(self.provider_id)
        try:
            res = await model.ainvoke(input, config=config, **kwargs)
            if spec: spec.record_success()
            return res
        except Exception as e:
            if spec: spec.record_failure(e)
            raise e


# ══════════════════════════════════════════════
# RAW PROVIDER FACTORY FUNCTIONS
# ══════════════════════════════════════════════

def _build_gemini_model(model_name: str = "gemini-2.0-flash", temperature: float = 0.0) -> Optional[BaseChatModel]:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key and gemini_api_key.strip():
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            models_to_try = [model_name, "gemini-2.0-flash", "gemini-1.5-pro"]
            for m in models_to_try:
                try:
                    return ChatGoogleGenerativeAI(
                        model=m,
                        google_api_key=gemini_api_key.strip(),
                        temperature=temperature,
                        max_output_tokens=1024,
                        timeout=5.0
                    )
                except (ImportError, ConnectionError, ValueError) as e:
                    print(f"[MODEL PROVIDER] Gemini model {m} unavailable: {e}")
                    continue
        except Exception as e:
            print(f"[MODEL PROVIDER WARNING] Gemini initialization error: {e}")
    return None


def _build_groq_model(model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.0, max_tokens: int = 250) -> Optional[BaseChatModel]:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key and groq_api_key.strip():
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model_name=model_name,
                groq_api_key=groq_api_key.strip(),
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=1
            )
        except Exception as e:
            print(f"[MODEL PROVIDER WARNING] Groq initialization error: {e}")
    return None


def _build_openrouter_model(model_name: str = "anthropic/claude-3.5-sonnet", temperature: float = 0.0, max_tokens: int = 400) -> Optional[BaseChatModel]:
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key and openrouter_api_key.strip():
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model_name=model_name,
                openai_api_key=openrouter_api_key.strip(),
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=1
            )
        except Exception as e:
            print(f"[MODEL PROVIDER WARNING] OpenRouter initialization error: {e}")
    return None


def _build_cohere_model(model_name: str = "command-r-plus-08-2024", temperature: float = 0.0) -> Optional[BaseChatModel]:
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if cohere_api_key and cohere_api_key.strip():
        try:
            from langchain_cohere import ChatCohere
            return ChatCohere(
                cohere_api_key=cohere_api_key.strip(),
                model=model_name,
                temperature=temperature
            )
        except Exception as e:
            print(f"[MODEL PROVIDER WARNING] Cohere initialization error: {e}")
    return None


def get_gemini_model(model_name: str = "gemini-2.0-flash", temperature: float = 0.0) -> Optional[BaseChatModel]:
    return _build_gemini_model(model_name=model_name, temperature=temperature)


def get_cohere_model(model_name: str = "command-r-plus-08-2024", temperature: float = 0.0) -> Optional[BaseChatModel]:
    return _build_cohere_model(model_name=model_name, temperature=temperature)


_GEMINI_QUOTA_EXHAUSTED = False

def set_gemini_quota_exhausted():
    global _GEMINI_QUOTA_EXHAUSTED
    _GEMINI_QUOTA_EXHAUSTED = True


# ══════════════════════════════════════════════
# PUBLIC MODEL RETRIEVAL APIS WITH LAZY TIERING
# ══════════════════════════════════════════════

@functools.lru_cache(maxsize=2)
def get_router_model() -> BaseChatModel:
    """
    Returns a fast model for Intent Routing & Query Classification.
    Constructs Primary model immediately; fallbacks are registered for lazy execution.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    specs: List[ProviderSpec] = []
    priority = 0

    if openrouter_api_key:
        specs.append(ProviderSpec(
            provider_id="router-openrouter-free",
            tier=0,
            priority=priority,
            factory=lambda: _build_openrouter_model("meta-llama/llama-3.1-8b-instruct:free", temperature=0.0)
        ))
        priority += 1

    if gemini_api_key and not _GEMINI_QUOTA_EXHAUSTED:
        tier = 0 if not specs else 1
        specs.append(ProviderSpec(
            provider_id="router-gemini-flash",
            tier=tier,
            priority=priority,
            factory=lambda: _build_gemini_model("gemini-2.0-flash", temperature=0.0)
        ))
        priority += 1

    if groq_api_key:
        tier = 0 if not specs else 1
        specs.append(ProviderSpec(
            provider_id="router-groq-70b",
            tier=tier,
            priority=priority,
            factory=lambda: _build_groq_model("llama-3.3-70b-versatile", temperature=0.0)
        ))
        priority += 1

    if cohere_api_key:
        specs.append(ProviderSpec(
            provider_id="router-cohere",
            tier=3,
            priority=priority,
            factory=lambda: _build_cohere_model(temperature=0.0)
        ))

    if not specs:
        raise ValueError("API Key missing! Please set OPENROUTER_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, or COHERE_API_KEY.")

    for spec in specs:
        ProviderFactory.register(spec)

    # Construct ONLY Tier 0 (Primary) provider during startup
    primary_spec = specs[0]
    primary_model = ProviderFactory.get_or_create(primary_spec.provider_id)
    if not primary_model:
        # Fallback to next available tier if primary instantiation fails
        for s in specs[1:]:
            primary_model = ProviderFactory.get_or_create(s.provider_id)
            if primary_model: break

    if not primary_model:
        raise RuntimeError("Failed to construct any valid LLM router model.")

    fallback_runnables = [LazyProviderRunnable(s.provider_id) for s in specs[1:]]
    if fallback_runnables:
        return primary_model.with_fallbacks(fallback_runnables)
    return primary_model


@functools.lru_cache(maxsize=4)
def get_reasoning_model(model_override: Optional[str] = None) -> BaseChatModel:
    """
    Returns high reasoning quality model for Paddy Diagnosis & Advisory.
    Constructs ONLY Tier 0 (Primary Gemini 2.0 Flash / Groq 70B) at startup (~150ms).
    Tiers 1, 2, 3 registered as lazy configuration specs for zero-cost startup.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    specs: List[ProviderSpec] = []
    priority = 0

    # Tier 0: Primary (Gemini 2.0 Flash)
    if gemini_api_key and not _GEMINI_QUOTA_EXHAUSTED:
        specs.append(ProviderSpec(
            provider_id="reasoning-gemini-2.0-flash",
            tier=0,
            priority=priority,
            factory=lambda: _build_gemini_model("gemini-2.0-flash", temperature=0.0)
        ))
        priority += 1

    # Tier 1: Fast Fallback (Groq Llama 3.3 70B)
    if groq_api_key:
        tier = 0 if not specs else 1
        specs.append(ProviderSpec(
            provider_id="reasoning-groq-llama-70b",
            tier=tier,
            priority=priority,
            factory=lambda: _build_groq_model("llama-3.3-70b-versatile", temperature=0.0, max_tokens=250)
        ))
        priority += 1

    # Tier 2: Emergency Fallback (Gemini 1.5 Pro)
    if gemini_api_key and not _GEMINI_QUOTA_EXHAUSTED:
        specs.append(ProviderSpec(
            provider_id="reasoning-gemini-1.5-pro",
            tier=2,
            priority=priority,
            factory=lambda: _build_gemini_model("gemini-1.5-pro", temperature=0.0)
        ))
        priority += 1

    # Tier 3: Rare Fallbacks (OpenRouter Claude & GPT-4o)
    if openrouter_api_key:
        target_model = model_override or "anthropic/claude-3.5-sonnet"
        specs.append(ProviderSpec(
            provider_id="reasoning-openrouter-claude",
            tier=3,
            priority=priority,
            factory=lambda: _build_openrouter_model(target_model, temperature=0.0, max_tokens=400)
        ))
        priority += 1
        specs.append(ProviderSpec(
            provider_id="reasoning-openrouter-gpt4o",
            tier=3,
            priority=priority,
            factory=lambda: _build_openrouter_model("openai/gpt-4o", temperature=0.0, max_tokens=400)
        ))
        priority += 1

    # Tier 2: Emergency Fallback (Groq 8B Instant)
    if groq_api_key:
        specs.append(ProviderSpec(
            provider_id="reasoning-groq-llama-8b",
            tier=2,
            priority=priority,
            factory=lambda: _build_groq_model("llama-3.1-8b-instant", temperature=0.0)
        ))
        priority += 1

    # Tier 3: Rare Fallback (Cohere)
    if cohere_api_key:
        specs.append(ProviderSpec(
            provider_id="reasoning-cohere",
            tier=3,
            priority=priority,
            factory=lambda: _build_cohere_model(temperature=0.0)
        ))

    if not specs:
        raise ValueError("API Key missing! Please set GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, or COHERE_API_KEY.")

    for spec in specs:
        ProviderFactory.register(spec)

    # Construct ONLY Tier 0 (Primary) provider during startup
    primary_spec = specs[0]
    primary_model = ProviderFactory.get_or_create(primary_spec.provider_id)
    if not primary_model:
        for s in specs[1:]:
            primary_model = ProviderFactory.get_or_create(s.provider_id)
            if primary_model: break

    if not primary_model:
        raise RuntimeError("Failed to construct any valid LLM reasoning model.")

    # Wrap fallback tiers in LazyProviderRunnable instances (Zero startup cost)
    fallback_runnables = [LazyProviderRunnable(s.provider_id) for s in specs[1:]]
    if fallback_runnables:
        return primary_model.with_fallbacks(fallback_runnables)
    return primary_model


@functools.lru_cache(maxsize=2)
def get_vision_model() -> Optional[BaseChatModel]:
    """
    Returns Vision-capable Multimodal Chat LLM.
    Constructs primary Gemini vision model at startup; falls back to reasoning model tier.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key and not _GEMINI_QUOTA_EXHAUSTED:
        spec = ProviderSpec(
            provider_id="vision-gemini-2.0-flash",
            tier=0,
            priority=0,
            factory=lambda: _build_gemini_model("gemini-2.0-flash", temperature=0.0)
        )
        ProviderFactory.register(spec)
        vision_model = ProviderFactory.get_or_create(spec.provider_id)
        if vision_model:
            return vision_model
    return get_reasoning_model()


def get_provider_status() -> Dict[str, bool]:
    """
    Returns active connection status for all supported AI providers.
    """
    return {
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "cohere": bool(os.getenv("COHERE_API_KEY")),
        "openrouter": bool(os.getenv("OPENROUTER_API_KEY"))
    }


if __name__ == "__main__":
    print("[INFO] Multi-Provider Configuration-Driven Factory initialized.")
    status = get_provider_status()
    print("  Provider Connection Status:", status)
    print("  Active Router Model:", get_router_model())
    print("  Constructed Provider Count at Startup:", ProviderFactory.get_construction_count())
