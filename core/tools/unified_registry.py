"""
UnifiedRegistry Module - Version 3.0 Architecture
Thread-safe singleton registry combining Capability and Tool metadata.
"""

import threading
from typing import Dict, List, Optional
from core.capabilities.capability_spec import CapabilitySpec
from core.tools.tool_spec import ToolSpec


class UnifiedRegistry:
    """Central repository storing active Capability and Tool specifications."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(UnifiedRegistry, cls).__new__(cls)
                cls._instance._capabilities: Dict[str, CapabilitySpec] = {}
                cls._instance._tools: Dict[str, ToolSpec] = {}
        return cls._instance

    def register_capability(self, spec: CapabilitySpec) -> None:
        with self._lock:
            self._capabilities[spec.capability_id] = spec

    def register_tool(self, spec: ToolSpec) -> None:
        with self._lock:
            self._tools[spec.tool_id] = spec

    def get_capability(self, capability_id: str) -> Optional[CapabilitySpec]:
        with self._lock:
            return self._capabilities.get(capability_id)

    def get_tool(self, tool_id: str) -> Optional[ToolSpec]:
        with self._lock:
            return self._tools.get(tool_id)

    def list_capabilities(self) -> List[CapabilitySpec]:
        with self._lock:
            return list(self._capabilities.values())

    def list_tools(self) -> List[ToolSpec]:
        with self._lock:
            return list(self._tools.values())

    def clear(self) -> None:
        with self._lock:
            self._capabilities.clear()
            self._tools.clear()


unified_registry = UnifiedRegistry()


def register_tool_decorator(capability_id: str, tool_id: str, **kwargs):
    """Decorator helper for registering concrete tool functions."""
    def decorator(fn):
        spec = ToolSpec(
            tool_id=tool_id,
            capability_id=capability_id,
            execute_fn=fn,
            **kwargs
        )
        unified_registry.register_tool(spec)
        return fn
    return decorator
