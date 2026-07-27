"""
Model Provider Factory Module
Implements Requirement 4c: Deliberate Multi-Provider Model Selection Strategy
Connects Google Gemini API, Groq Cloud, Cohere Command R+, and OpenRouter for high-reasoning, low-latency agent execution.
"""

import os
import re
from typing import Optional, Dict
import functools
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()


def get_gemini_model(model_name: str = "gemini-2.0-flash", temperature: float = 0.0) -> Optional[BaseChatModel]:
    """
    Instantiates Google Gemini LLM via langchain-google-genai.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key and gemini_api_key.strip():
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            models_to_try = [model_name, "gemini-2.0-flash", "gemini-1.5-pro"]
            for m in models_to_try:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=m,
                        google_api_key=gemini_api_key.strip(),
                        temperature=temperature,
                        max_output_tokens=1024
                    )
                    return llm
                except Exception:
                    continue
        except Exception as e:
            print(f"[MODEL PROVIDER WARNING] Gemini initialization error: {e}")
            return None
    return None


def get_cohere_model(model_name: str = "command-r-plus-08-2024", temperature: float = 0.0) -> Optional[BaseChatModel]:
    """
    Instantiates Cohere Command R+ enterprise LLM via langchain-cohere.
    """
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
    return None


_GEMINI_QUOTA_EXHAUSTED = False

def set_gemini_quota_exhausted():
    global _GEMINI_QUOTA_EXHAUSTED
    _GEMINI_QUOTA_EXHAUSTED = True


@functools.lru_cache(maxsize=2)
def get_router_model() -> BaseChatModel:
    """
    Returns a fast model for Intent Routing & Query Classification.
    Priority: 1. OpenRouter -> 2. Gemini -> 3. Groq -> 4. Cohere
    Automatically cascades to fallbacks if an API fails.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    models = []

    # Priority 1: OpenRouter (Fast Model)
    if openrouter_api_key:
        from langchain_openai import ChatOpenAI
        models.append(ChatOpenAI(
            model_name="meta-llama/llama-3.1-8b-instruct:free",
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_retries=1
        ))

    # Priority 2: Gemini (Flash)
    if gemini_api_key and not _GEMINI_QUOTA_EXHAUSTED:
        gemini_llm = get_gemini_model(model_name="gemini-2.0-flash", temperature=0.0)
        if gemini_llm:
            models.append(gemini_llm)

    # Priority 3: Groq (Ultra-Fast 70B/8B)
    if groq_api_key:
        from langchain_groq import ChatGroq
        models.append(ChatGroq(
            model_name="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key,
            temperature=0.0,
            max_retries=1
        ))

    # Priority 4: Cohere
    cohere_llm = get_cohere_model(temperature=0.0)
    if cohere_llm:
        models.append(cohere_llm)

    if not models:
        raise ValueError("API Key missing! Please set OPENROUTER_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, or COHERE_API_KEY.")

    primary_model = models[0]
    if len(models) > 1:
        return primary_model.with_fallbacks(models[1:])
    return primary_model


@functools.lru_cache(maxsize=4)
def get_reasoning_model(model_override: Optional[str] = None) -> BaseChatModel:
    """
    Returns a high reasoning quality model for Paddy Disease Diagnosis & Fertilizer Synthesis.
    Speed-optimized cascade (fastest first):
    Tier 1 (Ultra-Fast): Gemini 2.0 Flash (~1-2s) -> Groq Llama 70B (~1-3s)
    Tier 2 (High Quality): Gemini 1.5 Pro -> OpenRouter Claude/GPT-4o
    Tier 3 (Fallbacks): Groq 8B Instant -> Cohere
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    T = 0.0
    models = []

    # 1. PRIMARY: Gemini 2.0 Flash (Ultra-Fast, Free, High Quality)
    if gemini_api_key and not _GEMINI_QUOTA_EXHAUSTED:
        gemini_flash = get_gemini_model(model_name="gemini-2.0-flash", temperature=T)
        if gemini_flash:
            models.append(gemini_flash)

    # 2. Groq Llama 3.3 70B (Ultra-Fast via Groq hardware)
    if groq_api_key:
        from langchain_groq import ChatGroq
        models.append(ChatGroq(
            model_name="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key,
            temperature=T,
            max_tokens=250,
            max_retries=1
        ))

    # 3. Gemini 1.5 Pro (High Quality Fallback)
    if gemini_api_key and not _GEMINI_QUOTA_EXHAUSTED:
        gemini_pro = get_gemini_model(model_name="gemini-1.5-pro", temperature=T)
        if gemini_pro:
            models.append(gemini_pro)

    # 4. OpenRouter Pro Models (Last Resort - slowest)
    if openrouter_api_key:
        from langchain_openai import ChatOpenAI
        target_model = model_override or "anthropic/claude-3.5-sonnet"
        models.append(ChatOpenAI(
            model_name=target_model,
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=T,
            max_tokens=400,
            max_retries=1
        ))
        models.append(ChatOpenAI(
            model_name="openai/gpt-4o",
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=T,
            max_tokens=400,
            max_retries=1
        ))

    # 5. Groq 8B Instant (Ultra-Fast Lightweight Fallback)
    if groq_api_key:
        from langchain_groq import ChatGroq
        models.append(ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=groq_api_key,
            temperature=T,
            max_retries=1
        ))

    # 6. Cohere Enterprise Tier
    cohere_llm = get_cohere_model(temperature=T)
    if cohere_llm:
        models.append(cohere_llm)

    if not models:
        raise ValueError("API Key missing! Please set GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, or COHERE_API_KEY.")

    primary_model = models[0]
    if len(models) > 1:
        return primary_model.with_fallbacks(models[1:])
    return primary_model


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
    print("[INFO] Multi-Provider Model Factory initialized.")
    status = get_provider_status()
    print("  Provider Connection Status:", status)
    print("  Active Router Model:", get_router_model())
