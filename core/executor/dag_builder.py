"""
DAGBuilder Module - Version 3.0 Architecture
Performs Topological Sorting (Kahn's Algorithm) to resolve parallel tool execution stages.
"""

from typing import List, Dict
from core.tools.tool_spec import ToolSpec


class DAGBuilder:
    """Constructs staged execution plans from tool dependencies."""

    @staticmethod
    def build_execution_stages(tools: List[ToolSpec]) -> List[List[ToolSpec]]:
        if not tools:
            return []

        # Maps tool_id to ToolSpec
        tool_map: Dict[str, ToolSpec] = {t.tool_id: t for t in tools}
        in_degree: Dict[str, int] = {t.tool_id: 0 for t in tools}
        graph: Dict[str, List[str]] = {t.tool_id: [] for t in tools}

        # Calculate dependencies within provided tool list
        for t in tools:
            for dep in t.dependencies:
                if dep in tool_map:
                    graph[dep].append(t.tool_id)
                    in_degree[t.tool_id] += 1

        stages: List[List[ToolSpec]] = []
        current_stage = [t for t in tools if in_degree[t.tool_id] == 0]

        while current_stage:
            stages.append(current_stage)
            next_stage: List[ToolSpec] = []
            for t in current_stage:
                for neighbor in graph[t.tool_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_stage.append(tool_map[neighbor])
            current_stage = next_stage

        return stages
