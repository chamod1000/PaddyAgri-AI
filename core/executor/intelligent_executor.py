"""
IntelligentExecutor Module - Version 3.0 Architecture
Executes tool DAG stages in parallel via ThreadPoolExecutor with circuit breaker failovers.
"""

import time
import concurrent.futures
from typing import List, Dict, Any, Optional
from core.tools.tool_spec import ToolSpec
from core.tools.circuit_breaker import circuit_breaker
from core.executor.dag_builder import DAGBuilder
from core.evidence.evidence_graph import EvidenceGraph
from core.observability.telemetry_bus import telemetry_bus


class IntelligentExecutor:
    """Dynamic DAG execution engine with parallel dispatch and error handling."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def execute_plan(self, tools: List[ToolSpec], processing_context: Any, evidence_graph: EvidenceGraph) -> EvidenceGraph:
        if not tools:
            return evidence_graph

        stages = DAGBuilder.build_execution_stages(tools)
        t_start = time.perf_counter()

        for stage_idx, stage_tools in enumerate(stages):
            print(f"[EXECUTOR] Executing Stage {stage_idx + 1}/{len(stages)} ({len(stage_tools)} parallel tasks)...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(stage_tools))) as executor:
                future_map = {}
                for tool in stage_tools:
                    future = executor.submit(self._run_tool, tool, processing_context)
                    future_map[future] = tool

                for future in concurrent.futures.as_completed(future_map):
                    tool = future_map[future]
                    try:
                        result, dur_ms = future.result()
                        evidence_graph.deposit_artifact(tool.capability_id, result, dur_ms)
                        circuit_breaker.record_success(tool.tool_id)
                        telemetry_bus.emit("TOOL_EXECUTION_SUCCESS", {
                            "tool_id": tool.tool_id,
                            "capability_id": tool.capability_id,
                            "duration_ms": dur_ms
                        })
                    except Exception as e:
                        print(f"[EXECUTOR ERROR] Tool '{tool.tool_id}' failed: {e}")
                        circuit_breaker.record_failure(tool.tool_id)
                        evidence_graph.record_error(tool.tool_id, str(e))
                        telemetry_bus.emit("TOOL_EXECUTION_FAILURE", {
                            "tool_id": tool.tool_id,
                            "error": str(e)
                        })

        tot_dur_ms = (time.perf_counter() - t_start) * 1000.0
        evidence_graph.execution_metrics["total_executor_ms"] = tot_dur_ms
        return evidence_graph

    def _run_tool(self, tool: ToolSpec, context: Any) -> tuple[Dict[str, Any], float]:
        t0 = time.perf_counter()
        if not tool.execute_fn:
            return ({}, 0.0)

        # Execute tool callable
        res = tool.execute_fn(context)
        dur_ms = (time.perf_counter() - t0) * 1000.0
        return (res, dur_ms)
