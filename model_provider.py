"""
Model Provider Factory Module
Implements Mandatory Requirement 4c: Deliberate Model Selection Strategy
Selects appropriate models across Groq and OpenRouter based on task latency, cost, and reasoning needs.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()


def get_router_model() -> BaseChatModel:
    """
    Returns a fast, low-latency, low-cost model for Intent Routing / Query Classification.
    Provider: Groq (Llama 3.1 8B Instant) or OpenRouter fallback.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    if groq_api_key:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=groq_api_key,
            temperature=0.0
        )
    elif openrouter_api_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model_name="meta-llama/llama-3.1-8b-instruct:free",
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0
        )
    else:
        # Fallback to local HuggingFace or mock notification if no API key set
        raise ValueError(
            "API Key missing! Please set GROQ_API_KEY or OPENROUTER_API_KEY in your .env file."
        )


def get_reasoning_model(model_override: Optional[str] = None) -> BaseChatModel:
    """
    Returns a high reasoning quality model for Paddy Disease Synthesis and Fertilizer Recommendations.
    Provider: Groq (Llama 3.3 70B Versatile) or OpenRouter (Claude 3.5 Sonnet / GPT-4o-mini).
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    if openrouter_api_key:
        from langchain_openai import ChatOpenAI
        target_model = model_override or "anthropic/claude-3.5-sonnet"
        return ChatOpenAI(
            model_name=target_model,
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.2
        )
    elif groq_api_key:
        from langchain_groq import ChatGroq
        target_model = model_override or "llama-3.3-70b-versatile"
        return ChatGroq(
            model_name=target_model,
            groq_api_key=groq_api_key,
            temperature=0.2
        )
    else:
        raise ValueError(
            "API Key missing! Please set GROQ_API_KEY or OPENROUTER_API_KEY in your .env file."
        )


if __name__ == "__main__":
    print("[INFO] Model Provider Factory initialized.")
    print("  Router Model    : Groq llama-3.1-8b-instant (Fast / Low Latency)")
    print("  Reasoning Model : OpenRouter / Groq Llama 3.3 70B (High Synthesis Quality)")
