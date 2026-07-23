"""
Model Provider Factory Module
Implements Requirement 4c: Deliberate Multi-Provider Model Selection Strategy
Connects Google Gemini API, Groq Cloud, and OpenRouter for high-reasoning, low-latency agent execution.
"""

import os
from typing import Optional, Dict
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()


def get_gemini_model(model_name: str = "gemini-1.5-flash", temperature: float = 0.2) -> Optional[BaseChatModel]:
    """
    Instantiates Google Gemini LLM via langchain-google-genai.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key and gemini_api_key.strip():
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=gemini_api_key.strip(),
                temperature=temperature
            )
        except Exception as e:
            print(f"[MODEL PROVIDER WARNING] Gemini initialization error: {e}")
            return None
    return None


def get_router_model() -> BaseChatModel:
    """
    Returns a fast, low-latency model for Intent Routing & Query Classification.
    Primary: Groq Llama 3.1 8B Instant (200ms latency)
    Fallback: Google Gemini 1.5 Flash or OpenRouter
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    if groq_api_key:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=groq_api_key,
            temperature=0.0
        )
    elif gemini_api_key:
        gemini_llm = get_gemini_model(model_name="gemini-1.5-flash", temperature=0.0)
        if gemini_llm:
            return gemini_llm

    if openrouter_api_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model_name="meta-llama/llama-3.1-8b-instruct:free",
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0
        )
    
    raise ValueError(
        "API Key missing! Please set GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY in your .env file."
    )


def get_reasoning_model(model_override: Optional[str] = None) -> BaseChatModel:
    """
    Returns a high reasoning quality model for Paddy Disease Diagnosis & Fertilizer Synthesis.
    Priority Order:
      1. Google Gemini (Gemini 1.5 Flash / Pro)
      2. Groq (Llama 3.3 70B Versatile)
      3. OpenRouter (Claude 3.5 / Llama 70B)
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    # Priority 1: Google Gemini API
    if gemini_api_key:
        target_model = model_override or "gemini-1.5-flash"
        gemini_llm = get_gemini_model(model_name=target_model, temperature=0.2)
        if gemini_llm:
            return gemini_llm

    # Priority 2: Groq Llama 3.3 70B
    if groq_api_key:
        from langchain_groq import ChatGroq
        target_model = model_override or "llama-3.3-70b-versatile"
        return ChatGroq(
            model_name=target_model,
            groq_api_key=groq_api_key,
            temperature=0.2
        )

    # Priority 3: OpenRouter
    if openrouter_api_key:
        from langchain_openai import ChatOpenAI
        target_model = model_override or "anthropic/claude-3.5-sonnet"
        return ChatOpenAI(
            model_name=target_model,
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.2
        )

    raise ValueError(
        "API Key missing! Please set GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY in your .env file."
    )


def get_provider_status() -> Dict[str, bool]:
    """
    Returns active connection status for all 3 supported AI providers.
    """
    return {
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "openrouter": bool(os.getenv("OPENROUTER_API_KEY"))
    }


if __name__ == "__main__":
    print("[INFO] Multi-Provider Model Factory initialized.")
    status = get_provider_status()
    print("  Provider Connection Status:", status)
    print("  Active Router Model:", get_router_model())
    print("  Active Reasoning Model:", get_reasoning_model())
