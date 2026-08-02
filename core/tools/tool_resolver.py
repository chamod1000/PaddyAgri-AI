"""
ToolResolver Module - Version 3.0 Architecture
Maps abstract capability IDs to active, healthy tool specs with fallback support.
"""

from typing import List, Optional
from core.capabilities.capability_spec import CapabilitySpec
from core.tools.tool_spec import ToolSpec
from core.tools.unified_registry import unified_registry
from core.tools.circuit_breaker import circuit_breaker, CircuitState


class AdaptiveToolResolver:
    """Resolves abstract capability requests into healthy, executable ToolSpec instances."""

    @staticmethod
    def resolve_capability(capability_id: str) -> Optional[ToolSpec]:
        cap_spec: Optional[CapabilitySpec] = unified_registry.get_capability(capability_id)
        if not cap_spec:
            # Fallback direct tool ID lookup
            return unified_registry.get_tool(capability_id)

        # 1. Try Primary Default Tool
        primary_tool_id = cap_spec.default_tool_id
        if primary_tool_id:
            state = circuit_breaker.get_state(primary_tool_id)
            if state != CircuitState.OPEN:
                tool = unified_registry.get_tool(primary_tool_id)
                if tool:
                    return tool

        # 2. Try Fallback Tool
        fallback_tool_id = cap_spec.fallback_tool_id
        if fallback_tool_id:
            state = circuit_breaker.get_state(fallback_tool_id)
            if state != CircuitState.OPEN:
                tool = unified_registry.get_tool(fallback_tool_id)
                if tool:
                    return tool

        # 3. Return primary tool as last resort if no alternative exists
        return unified_registry.get_tool(primary_tool_id)

    @classmethod
    def resolve_capabilities(cls, capability_ids: List[str]) -> List[ToolSpec]:
        resolved: List[ToolSpec] = []
        for cid in capability_ids:
            tool = cls.resolve_capability(cid)
            if tool and tool not in resolved:
                resolved.append(tool)
        return resolved
