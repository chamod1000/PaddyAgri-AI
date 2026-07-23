"""
Model Provider Factory Module
Implements Requirement 4c: Deliberate Multi-Provider Model Selection Strategy
Connects Google Gemini API, Groq Cloud, and OpenRouter for high-reasoning, low-latency agent execution.
"""

import os
import re
from typing import Optional, Dict
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()


def detect_language_and_script(query_text: str) -> bool:
    """
    Intelligent Auto-Detection for Sinhala & Singlish context.
    Returns True if Sinhala unicode characters or Singlish phonetic patterns are detected.
    """
    text_lower = query_text.lower()

    # A) Check for Sinhala Unicode characters range (඀ to ෿)
    if re.search(r'[඀-෿]', query_text):
        return True

    # B) Check for Singlish phonetic patterns / agricultural keywords
    singlish_keywords = [
        'goyam', 'pohora', 'kannaya', 'pala', 'peththara',
        'wagawe', 'lapa', 'kaha', 'rogo', 'beheth', 'gedi', 'kola'
    ]

    # Use word boundary regex to avoid partial matches inside English words
    for kw in singlish_keywords:
        if re.search(rf'\b{kw}\b', text_lower):
            return True

    return False


def get_gemini_model(model_name: str = "gemini-2.0-flash", temperature: float = 0.1) -> Optional[BaseChatModel]:
    """
    Instantiates Google Gemini LLM via langchain-google-genai.
    Tries gemini-2.0-flash or gemini-1.5-pro.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key and gemini_api_key.strip():
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Ensure valid model name format for Google GenAI v1beta
            models_to_try = [model_name, "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash-latest"]
            for m in models_to_try:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=m,
                        google_api_key=gemini_api_key.strip(),
                        temperature=temperature
                    )
                    return llm
                except Exception:
                    continue
        except Exception as e:
            print(f"[MODEL PROVIDER WARNING] Gemini initialization error: {e}")
            return None
    return None


def get_router_model() -> BaseChatModel:
    """
    Returns a fast, low-latency model for Intent Routing & Query Classification.
    Primary: Groq Llama 3.1 8B Instant (200ms latency)
    Fallback: Google Gemini or OpenRouter
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    if groq_api_key:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=groq_api_key,
            temperature=0.0  # Zero for routing strictly
        )
    elif gemini_api_key:
        gemini_llm = get_gemini_model(model_name="gemini-2.0-flash", temperature=0.0)
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


def get_reasoning_model(model_override: Optional[str] = None, is_sinhala_or_singlish: bool = False) -> BaseChatModel:
    """
    Returns a high reasoning quality model for Paddy Disease Diagnosis & Fertilizer Synthesis.
    Priority Order when Sinhala/Singlish is DETECTED:
      1. Google Gemini API (gemini-2.0-flash / gemini-1.5-pro) - Best multilingual comprehension
      2. Groq Llama 3.3 70B Versatile
      3. OpenRouter

    Priority Order for Standard English:
      1. Groq Llama 3.3 70B Versatile (SOTA Fast 70B Model)
      2. Google Gemini API
      3. OpenRouter
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    # Ultra-Fast Generation: Temperature 0.1 for precise tokens
    T = 0.1

    if is_sinhala_or_singlish and gemini_api_key:
        # Prio 1 for Sinhala/Singlish: Gemini
        target_model = model_override or "gemini-2.0-flash"
        gemini_llm = get_gemini_model(model_name=target_model, temperature=T)
        if gemini_llm:
            print("[MODEL PROVIDER] Detected Sinhala/Singlish! Routing to Google Gemini API.")
            return gemini_llm

    # Priority 1 for English or Fallback for Sinhala: Groq Llama 3.3 70B
    if groq_api_key:
        from langchain_groq import ChatGroq
        target_model = model_override or "llama-3.3-70b-versatile"
        return ChatGroq(
            model_name=target_model,
            groq_api_key=groq_api_key,
            temperature=T
        )

    # Priority 2 for English: Google Gemini API
    if gemini_api_key:
        target_model = model_override or "gemini-2.0-flash"
        gemini_llm = get_gemini_model(model_name=target_model, temperature=T)
        if gemini_llm:
            return gemini_llm

    # Priority 3: OpenRouter
    if openrouter_api_key:
        from langchain_openai import ChatOpenAI
        target_model = model_override or "anthropic/claude-3.5-sonnet"
        return ChatOpenAI(
            model_name=target_model,
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=T
        )

    raise ValueError(
        "API Key missing! Please set GROQ_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY in your .env file."
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
