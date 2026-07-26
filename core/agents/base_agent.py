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
            start_time = time.time()
            response = self.model.invoke(messages)
            duration = time.time() - start_time
            print(f"[{self.name}] LLM response received in {duration:.2f}s")
            return response
        except Exception as e:
            self._log_error(e, "All fallback models failed during LLM invocation")
            raise e

    @abstractmethod
    def process(self, message: AgentMessage) -> Any:
        """Process an agent message. Must be implemented by subclasses."""
        pass
