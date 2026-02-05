"""
Msty Admin MCP - Autonomous Agent Swarm (Phase 27)

Spawn and orchestrate multiple specialized AI agents working in parallel.
Each agent has a specialty and can work independently or collaboratively.

Features:
- Specialized agents (Code, Research, Writing, Analysis)
- Parallel execution
- Orchestrator coordination
- Result synthesis
- Agent lifecycle management
"""

import json
import logging
import uuid
import time
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from enum import Enum

from .network import make_api_request, get_available_service_ports
from .smart_router import route_request, classify_task

logger = logging.getLogger("msty-admin-mcp")


class AgentRole(Enum):
    """Specialized agent roles."""
    CODE = "code"
    RESEARCH = "research"
    WRITING = "writing"
    ANALYSIS = "analysis"
    REVIEW = "review"
    ORCHESTRATOR = "orchestrator"
    GENERAL = "general"


class AgentStatus(Enum):
    """Agent lifecycle status."""
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Agent:
    """Represents a single AI agent."""
    id: str
    role: AgentRole
    model_id: str
    port: int
    status: AgentStatus
    created_at: str
    system_prompt: str
    current_task: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


# Agent pool
_agents: Dict[str, Agent] = {}
_agent_results: Dict[str, List[Dict[str, Any]]] = {}


# Default system prompts for each role
ROLE_SYSTEM_PROMPTS = {
    AgentRole.CODE: """You are a specialized coding agent. Your expertise includes:
- Writing clean, efficient, well-documented code
- Debugging and fixing errors
- Code review and optimization
- Implementing algorithms and data structures
Focus on producing working, tested code. Be precise and technical.""",

    AgentRole.RESEARCH: """You are a specialized research agent. Your expertise includes:
- Gathering information from multiple sources
- Fact-checking and verification
- Synthesizing findings into clear summaries
- Identifying key insights and patterns
Focus on accuracy and comprehensiveness. Cite your reasoning.""",

    AgentRole.WRITING: """You are a specialized writing agent. Your expertise includes:
- Crafting clear, engaging prose
- Adapting tone and style for different audiences
- Structuring content logically
- Editing and refining text
Focus on clarity, flow, and impact.""",

    AgentRole.ANALYSIS: """You are a specialized analysis agent. Your expertise includes:
- Breaking down complex problems
- Identifying patterns and trends
- Evaluating options and tradeoffs
- Making data-driven recommendations
Focus on logical reasoning and actionable insights.""",

    AgentRole.REVIEW: """You are a specialized review agent. Your expertise includes:
- Quality assurance and validation
- Identifying errors, gaps, and improvements
- Providing constructive feedback
- Ensuring consistency and completeness
Focus on thoroughness and constructive criticism.""",

    AgentRole.ORCHESTRATOR: """You are an orchestrator agent. Your role is to:
- Coordinate work between multiple agents
- Synthesize results from different specialists
- Resolve conflicts and inconsistencies
- Produce coherent final outputs
Focus on integration and quality control.""",

    AgentRole.GENERAL: """You are a general-purpose AI assistant.
Be helpful, accurate, and thorough in your responses."""
}


def create_agent(
    role: AgentRole,
    model_id: Optional[str] = None,
    custom_system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a new specialized agent.

    Args:
        role: The agent's specialized role
        model_id: Specific model to use (auto-selects if None)
        custom_system_prompt: Override the default system prompt
    """
    agent_id = f"agent_{role.value}_{uuid.uuid4().hex[:8]}"

    # Auto-select model based on role if not specified
    if not model_id:
        task_hint = {
            AgentRole.CODE: "write code and debug",
            AgentRole.RESEARCH: "research and gather information",
            AgentRole.WRITING: "write creative content",
            AgentRole.ANALYSIS: "analyze data and reason",
            AgentRole.REVIEW: "review and critique",
            AgentRole.ORCHESTRATOR: "coordinate and synthesize",
            AgentRole.GENERAL: "general assistant task"
        }.get(role, "general task")

        routing = route_request(task_hint, prefer_quality=(role == AgentRole.ORCHESTRATOR))
        if routing.get("routed"):
            model_id = routing["selected_model"]
            port = routing["port"]
        else:
            return {
                "error": "No suitable model available for agent",
                "role": role.value
            }
    else:
        # Find port for specified model
        ports = get_available_service_ports()
        port = ports.get("local_ai", 11964)

    system_prompt = custom_system_prompt or ROLE_SYSTEM_PROMPTS.get(role, "")

    agent = Agent(
        id=agent_id,
        role=role,
        model_id=model_id,
        port=port,
        status=AgentStatus.IDLE,
        created_at=datetime.now().isoformat(),
        system_prompt=system_prompt
    )

    _agents[agent_id] = agent

    return {
        "timestamp": datetime.now().isoformat(),
        "agent_id": agent_id,
        "role": role.value,
        "model_id": model_id,
        "status": agent.status.value,
        "created": True
    }


def execute_agent_task(
    agent_id: str,
    task: str,
    context: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> Dict[str, Any]:
    """
    Execute a task with a specific agent.
    """
    if agent_id not in _agents:
        return {"error": f"Agent {agent_id} not found"}

    agent = _agents[agent_id]
    agent.status = AgentStatus.WORKING
    agent.current_task = task

    start_time = time.time()

    try:
        # Build messages
        messages = [
            {"role": "system", "content": agent.system_prompt}
        ]

        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})

        messages.append({"role": "user", "content": task})

        # Make API request
        response = make_api_request(
            f"http://127.0.0.1:{agent.port}/v1/chat/completions",
            method="POST",
            data={
                "model": agent.model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )

        execution_time = time.time() - start_time

        if response and "choices" in response:
            result = response["choices"][0]["message"]["content"]
            agent.result = result
            agent.status = AgentStatus.COMPLETED
            agent.execution_time = execution_time

            return {
                "timestamp": datetime.now().isoformat(),
                "agent_id": agent_id,
                "role": agent.role.value,
                "status": "completed",
                "result": result,
                "execution_time": execution_time,
                "model_id": agent.model_id
            }
        else:
            agent.status = AgentStatus.FAILED
            agent.error = "No response from model"
            return {
                "timestamp": datetime.now().isoformat(),
                "agent_id": agent_id,
                "status": "failed",
                "error": "No response from model"
            }

    except Exception as e:
        agent.status = AgentStatus.FAILED
        agent.error = str(e)
        return {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "status": "failed",
            "error": str(e)
        }


def spawn_swarm(
    task: str,
    roles: Optional[List[str]] = None,
    parallel: bool = True,
    synthesize: bool = True
) -> Dict[str, Any]:
    """
    Spawn a swarm of agents to tackle a complex task.

    Args:
        task: The main task to accomplish
        roles: List of roles to include (default: auto-detect)
        parallel: Run agents in parallel
        synthesize: Synthesize results with orchestrator
    """
    swarm_id = f"swarm_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now().isoformat()

    # Auto-detect roles if not specified
    if not roles:
        task_scores = classify_task(task)
        roles = []

        if task_scores.get("coding", 0) > 0.3:
            roles.append("code")
        if task_scores.get("reasoning", 0) > 0.3 or task_scores.get("complex", 0) > 0.3:
            roles.append("analysis")
        if task_scores.get("creative", 0) > 0.3:
            roles.append("writing")

        # Always include research for complex tasks
        if not roles or task_scores.get("complex", 0) > 0.5:
            roles.append("research")

        # Default to general if nothing detected
        if not roles:
            roles = ["general"]

    # Create agents for each role
    agents = []
    for role_name in roles:
        try:
            role = AgentRole(role_name)
        except ValueError:
            role = AgentRole.GENERAL

        result = create_agent(role)
        if "agent_id" in result:
            agents.append(result)

    if not agents:
        return {
            "timestamp": timestamp,
            "swarm_id": swarm_id,
            "error": "Failed to create any agents",
            "spawned": False
        }

    # Execute tasks
    results = []

    if parallel:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            futures = {
                executor.submit(
                    execute_agent_task,
                    agent["agent_id"],
                    task
                ): agent
                for agent in agents
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
    else:
        # Sequential execution
        for agent in agents:
            result = execute_agent_task(agent["agent_id"], task)
            results.append(result)

    # Synthesize results if requested
    synthesis = None
    if synthesize and len(results) > 1:
        synthesis = synthesize_results(swarm_id, results, task)

    # Store results
    _agent_results[swarm_id] = results

    return {
        "timestamp": timestamp,
        "swarm_id": swarm_id,
        "spawned": True,
        "agents_count": len(agents),
        "roles": roles,
        "parallel": parallel,
        "agent_results": results,
        "synthesis": synthesis,
        "execution_summary": {
            "completed": sum(1 for r in results if r.get("status") == "completed"),
            "failed": sum(1 for r in results if r.get("status") == "failed"),
            "total_time": sum(r.get("execution_time", 0) for r in results)
        }
    }


def synthesize_results(
    swarm_id: str,
    results: List[Dict[str, Any]],
    original_task: str
) -> Dict[str, Any]:
    """
    Synthesize results from multiple agents using an orchestrator.
    """
    # Create orchestrator
    orchestrator = create_agent(AgentRole.ORCHESTRATOR)

    if "error" in orchestrator:
        return {"error": "Could not create orchestrator", "details": orchestrator}

    # Build synthesis prompt
    agent_outputs = []
    for i, result in enumerate(results):
        if result.get("status") == "completed":
            role = result.get("role", "unknown")
            output = result.get("result", "")
            agent_outputs.append(f"### {role.upper()} Agent Output:\n{output}\n")

    synthesis_prompt = f"""Original Task: {original_task}

You have received outputs from multiple specialized agents. Your job is to:
1. Synthesize these into a coherent, unified response
2. Resolve any conflicts or inconsistencies
3. Ensure completeness and quality
4. Format the final output professionally

Agent Outputs:
{chr(10).join(agent_outputs)}

Please provide a synthesized, final response:"""

    # Execute synthesis
    synthesis_result = execute_agent_task(
        orchestrator["agent_id"],
        synthesis_prompt,
        temperature=0.5  # Lower temp for more focused synthesis
    )

    return {
        "swarm_id": swarm_id,
        "orchestrator_id": orchestrator["agent_id"],
        "synthesized": synthesis_result.get("status") == "completed",
        "synthesis": synthesis_result.get("result"),
        "synthesis_time": synthesis_result.get("execution_time")
    }


def get_agent_status(agent_id: str) -> Dict[str, Any]:
    """Get the current status of an agent."""
    if agent_id not in _agents:
        return {"error": f"Agent {agent_id} not found"}

    agent = _agents[agent_id]
    return {
        "timestamp": datetime.now().isoformat(),
        "agent_id": agent_id,
        "role": agent.role.value,
        "model_id": agent.model_id,
        "status": agent.status.value,
        "current_task": agent.current_task,
        "has_result": agent.result is not None,
        "error": agent.error,
        "execution_time": agent.execution_time
    }


def list_agents() -> Dict[str, Any]:
    """List all agents in the pool."""
    agents_list = []
    for agent_id, agent in _agents.items():
        agents_list.append({
            "agent_id": agent_id,
            "role": agent.role.value,
            "model_id": agent.model_id,
            "status": agent.status.value,
            "created_at": agent.created_at
        })

    return {
        "timestamp": datetime.now().isoformat(),
        "total_agents": len(agents_list),
        "agents": agents_list,
        "by_status": {
            status.value: sum(1 for a in _agents.values() if a.status == status)
            for status in AgentStatus
        }
    }


def terminate_agent(agent_id: str) -> Dict[str, Any]:
    """Terminate and remove an agent."""
    if agent_id not in _agents:
        return {"error": f"Agent {agent_id} not found"}

    agent = _agents.pop(agent_id)

    return {
        "timestamp": datetime.now().isoformat(),
        "agent_id": agent_id,
        "terminated": True,
        "final_status": agent.status.value
    }


def clear_all_agents() -> Dict[str, Any]:
    """Clear all agents from the pool."""
    count = len(_agents)
    _agents.clear()
    _agent_results.clear()

    return {
        "timestamp": datetime.now().isoformat(),
        "cleared": True,
        "agents_removed": count
    }


def get_swarm_results(swarm_id: str) -> Dict[str, Any]:
    """Get results from a previous swarm execution."""
    if swarm_id not in _agent_results:
        return {"error": f"Swarm {swarm_id} not found"}

    return {
        "timestamp": datetime.now().isoformat(),
        "swarm_id": swarm_id,
        "results": _agent_results[swarm_id]
    }


__all__ = [
    "AgentRole",
    "AgentStatus",
    "create_agent",
    "execute_agent_task",
    "spawn_swarm",
    "synthesize_results",
    "get_agent_status",
    "list_agents",
    "terminate_agent",
    "clear_all_agents",
    "get_swarm_results",
    "ROLE_SYSTEM_PROMPTS"
]
