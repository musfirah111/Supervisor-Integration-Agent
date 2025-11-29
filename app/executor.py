"""
Plan execution: resolve inputs for each step, call the right agent, and collect
outputs for later synthesis or UI debugging.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .agent_caller import call_agent
from .models import AgentMetadata, AgentResponse, Plan, UsedAgentEntry
from .registry import find_agent_by_name


def resolve_input(input_source: str, user_query: str, step_outputs: Dict[int, AgentResponse]) -> str:
    """Resolve an input_source directive into text for the worker."""
    if input_source == "user_query":
        return user_query
    if input_source.startswith("step:"):
        parts = input_source.split(":")
        if len(parts) >= 2:
            step_id_str = parts[1].split(".")[0]
            try:
                step_id = int(step_id_str)
                prior = step_outputs.get(step_id)
                if prior and prior.output:
                    return str(prior.output.result)
            except ValueError:
                pass
    return user_query


async def execute_plan(
    query: str,
    plan: Plan,
    registry: List[AgentMetadata],
    context: Dict[str, Any],
) -> Tuple[Dict[int, AgentResponse], List[UsedAgentEntry]]:
    """Execute each planned step in order and capture responses."""
    step_outputs: Dict[int, AgentResponse] = {}
    used_agents: List[UsedAgentEntry] = []

    for step in plan.steps:
        agent_meta = find_agent_by_name(step.agent, registry)
        text = resolve_input(step.input_source, query, step_outputs)
        # Pass file uploads from context to agent caller
        response = await call_agent(agent_meta, step.intent, text, context)
        step_outputs[step.step_id] = response
        used_agents.append(
            UsedAgentEntry(name=agent_meta.name, intent=step.intent, status=response.status)
        )

    return step_outputs, used_agents
