"""
BaseAgent Abstract Parent Class
Provides common LLM invocation, error handling, resilient fallback, and message logging infrastructure.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import time
from abc import ABC, abstractmethod
from typing import Any

from core.agent_messages import AgentMessage


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents.
    Handles common LLM invocation patterns, error logging, and message timing.
    """

    def __init__(self, name: str, model: Any):
        self.name = name
        self.model = model

    def _log_start(self, message: AgentMessage):
        print(f"[{self.name}] Processing message ID: {message.message_id}...")

    def _log_error(self, error: Exception, context: str = ""):
        print(f"[{self.name} ERROR] {context}: {error}")

    def _log_success(self, message_id: str):
        print(f"[{self.name}] Message {message_id} processed successfully.")

    def invoke_llm(self, messages: list) -> Any:
        """Common LLM invocation relying on LangChain's native with_fallbacks cascading."""
        try:
            t0 = time.perf_counter()
            response = self.model.invoke(messages)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Extract provider / model name
            model_info = "Unknown Model"
            if hasattr(response, "response_metadata") and isinstance(response.response_metadata, dict):
                model_info = (
                    response.response_metadata.get("model_name") or
                    response.response_metadata.get("model") or
                    str(response.response_metadata)
                )
            elif hasattr(self.model, "model_name"):
                model_info = getattr(self.model, "model_name")

            # Extract output token count
            output_tokens = 0
            if hasattr(response, "usage_metadata") and isinstance(response.usage_metadata, dict):
                output_tokens = response.usage_metadata.get("output_tokens", 0)
            elif hasattr(response, "response_metadata") and isinstance(response.response_metadata, dict):
                token_usage = response.response_metadata.get("token_usage", {}) or response.response_metadata.get("usage", {})
                if isinstance(token_usage, dict):
                    output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0

            if not output_tokens and hasattr(response, "content"):
                output_tokens = len(str(response.content).split())

            print(f"[{self.name}] LLM Call: {elapsed_ms:.2f} ms | Model/Provider: {model_info} | Output Tokens: {output_tokens}")
            return response
        except Exception as e:
            self._log_error(e, "All fallback models failed during LLM invocation")
            raise e

    @abstractmethod
    def process(self, message: AgentMessage) -> Any:
        """Process an agent message. Must be implemented by subclasses."""
        pass
