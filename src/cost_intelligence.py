"""
Msty Admin MCP - Cost Intelligence Dashboard (Phase 34)

Track token usage, estimate costs, and optimize spending across
local and cloud models.

Features:
- Token tracking per model/session
- Cost estimation (local vs cloud comparison)
- Budget alerts and limits
- Usage analytics and trends
- Optimization recommendations
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict
from dataclasses import dataclass, field

from .paths import get_msty_paths
from .network import make_api_request, get_available_service_ports

logger = logging.getLogger("msty-admin-mcp")


@dataclass
class UsageRecord:
    """A single usage record."""
    id: str
    timestamp: str
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    is_local: bool
    estimated_cost: float
    session_id: Optional[str] = None
    task_type: Optional[str] = None


# Storage
_usage_records: List[UsageRecord] = []
_session_budgets: Dict[str, float] = {}
_budget_alerts: List[Dict[str, Any]] = []

# Cost estimates per 1K tokens (rough estimates)
CLOUD_COSTS_PER_1K = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "default_cloud": {"input": 0.005, "output": 0.015}
}

# Local model electricity cost estimate (very rough)
LOCAL_COST_PER_1K_TOKENS = 0.0001  # ~$0.0001 per 1K tokens (electricity only)


def record_usage(
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    is_local: bool = True,
    session_id: Optional[str] = None,
    task_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Record a usage event.

    Args:
        model_id: The model that was used
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        latency_ms: Request latency in milliseconds
        is_local: Whether this was a local model
        session_id: Optional session identifier
        task_type: Optional task type for categorization
    """
    timestamp = datetime.now()
    total_tokens = prompt_tokens + completion_tokens

    # Estimate cost
    if is_local:
        estimated_cost = (total_tokens / 1000) * LOCAL_COST_PER_1K_TOKENS
    else:
        # Try to match cloud model
        costs = CLOUD_COSTS_PER_1K.get("default_cloud")
        for cloud_model, model_costs in CLOUD_COSTS_PER_1K.items():
            if cloud_model in model_id.lower():
                costs = model_costs
                break
        estimated_cost = (
            (prompt_tokens / 1000) * costs["input"] +
            (completion_tokens / 1000) * costs["output"]
        )

    record = UsageRecord(
        id=f"usage_{timestamp.strftime('%Y%m%d%H%M%S')}_{len(_usage_records)}",
        timestamp=timestamp.isoformat(),
        model_id=model_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        is_local=is_local,
        estimated_cost=estimated_cost,
        session_id=session_id,
        task_type=task_type
    )

    _usage_records.append(record)

    # Check budget alerts
    if session_id and session_id in _session_budgets:
        session_total = sum(
            r.estimated_cost for r in _usage_records
            if r.session_id == session_id
        )
        budget = _session_budgets[session_id]
        if session_total >= budget * 0.8:
            _create_budget_alert(session_id, session_total, budget)

    # Keep only last 10000 records
    if len(_usage_records) > 10000:
        _usage_records.pop(0)

    return {
        "timestamp": timestamp.isoformat(),
        "recorded": True,
        "record_id": record.id,
        "total_tokens": total_tokens,
        "estimated_cost": f"${estimated_cost:.6f}",
        "is_local": is_local
    }


def get_usage_summary(
    days: int = 30,
    model_filter: Optional[str] = None,
    session_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get usage summary for a time period.
    """
    cutoff = datetime.now() - timedelta(days=days)

    filtered = [
        r for r in _usage_records
        if datetime.fromisoformat(r.timestamp) >= cutoff
    ]

    if model_filter:
        filtered = [r for r in filtered if model_filter.lower() in r.model_id.lower()]

    if session_filter:
        filtered = [r for r in filtered if r.session_id == session_filter]

    if not filtered:
        return {
            "timestamp": datetime.now().isoformat(),
            "days": days,
            "total_requests": 0,
            "message": "No usage data in this period"
        }

    total_tokens = sum(r.total_tokens for r in filtered)
    total_cost = sum(r.estimated_cost for r in filtered)
    local_requests = sum(1 for r in filtered if r.is_local)
    cloud_requests = len(filtered) - local_requests

    # By model breakdown
    by_model = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "requests": 0})
    for r in filtered:
        by_model[r.model_id]["tokens"] += r.total_tokens
        by_model[r.model_id]["cost"] += r.estimated_cost
        by_model[r.model_id]["requests"] += 1

    # By task type
    by_task = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "requests": 0})
    for r in filtered:
        task = r.task_type or "unknown"
        by_task[task]["tokens"] += r.total_tokens
        by_task[task]["cost"] += r.estimated_cost
        by_task[task]["requests"] += 1

    return {
        "timestamp": datetime.now().isoformat(),
        "period_days": days,
        "total_requests": len(filtered),
        "local_requests": local_requests,
        "cloud_requests": cloud_requests,
        "total_tokens": total_tokens,
        "total_cost": f"${total_cost:.4f}",
        "average_tokens_per_request": total_tokens // len(filtered) if filtered else 0,
        "average_cost_per_request": f"${total_cost / len(filtered):.6f}" if filtered else "$0",
        "by_model": {
            model: {
                "tokens": data["tokens"],
                "cost": f"${data['cost']:.4f}",
                "requests": data["requests"]
            }
            for model, data in sorted(by_model.items(), key=lambda x: x[1]["cost"], reverse=True)[:10]
        },
        "by_task": {
            task: {
                "tokens": data["tokens"],
                "cost": f"${data['cost']:.4f}",
                "requests": data["requests"]
            }
            for task, data in sorted(by_task.items(), key=lambda x: x[1]["cost"], reverse=True)[:10]
        }
    }


def compare_local_vs_cloud() -> Dict[str, Any]:
    """
    Compare costs if all local requests had been made to cloud.
    Shows savings from using local models.
    """
    local_records = [r for r in _usage_records if r.is_local]

    if not local_records:
        return {
            "timestamp": datetime.now().isoformat(),
            "message": "No local usage data available"
        }

    # Calculate actual local cost
    actual_cost = sum(r.estimated_cost for r in local_records)

    # Calculate hypothetical cloud cost
    hypothetical_cloud_cost = 0.0
    for r in local_records:
        # Use Claude 3 Haiku as baseline comparison (cheapest capable model)
        costs = CLOUD_COSTS_PER_1K["claude-3-haiku"]
        hypothetical_cloud_cost += (
            (r.prompt_tokens / 1000) * costs["input"] +
            (r.completion_tokens / 1000) * costs["output"]
        )

    # Also calculate for more expensive models
    hypothetical_sonnet = 0.0
    for r in local_records:
        costs = CLOUD_COSTS_PER_1K["claude-3-sonnet"]
        hypothetical_sonnet += (
            (r.prompt_tokens / 1000) * costs["input"] +
            (r.completion_tokens / 1000) * costs["output"]
        )

    hypothetical_gpt4 = 0.0
    for r in local_records:
        costs = CLOUD_COSTS_PER_1K["gpt-4"]
        hypothetical_gpt4 += (
            (r.prompt_tokens / 1000) * costs["input"] +
            (r.completion_tokens / 1000) * costs["output"]
        )

    total_tokens = sum(r.total_tokens for r in local_records)

    return {
        "timestamp": datetime.now().isoformat(),
        "local_requests": len(local_records),
        "total_local_tokens": total_tokens,
        "actual_local_cost": f"${actual_cost:.4f}",
        "hypothetical_costs": {
            "claude_3_haiku": f"${hypothetical_cloud_cost:.4f}",
            "claude_3_sonnet": f"${hypothetical_sonnet:.4f}",
            "gpt_4": f"${hypothetical_gpt4:.4f}"
        },
        "savings": {
            "vs_haiku": f"${hypothetical_cloud_cost - actual_cost:.4f}",
            "vs_sonnet": f"${hypothetical_sonnet - actual_cost:.4f}",
            "vs_gpt4": f"${hypothetical_gpt4 - actual_cost:.4f}"
        },
        "savings_percentage": {
            "vs_haiku": f"{((hypothetical_cloud_cost - actual_cost) / hypothetical_cloud_cost * 100):.1f}%" if hypothetical_cloud_cost > 0 else "N/A",
            "vs_sonnet": f"{((hypothetical_sonnet - actual_cost) / hypothetical_sonnet * 100):.1f}%" if hypothetical_sonnet > 0 else "N/A",
            "vs_gpt4": f"{((hypothetical_gpt4 - actual_cost) / hypothetical_gpt4 * 100):.1f}%" if hypothetical_gpt4 > 0 else "N/A"
        }
    }


def get_daily_breakdown(days: int = 7) -> Dict[str, Any]:
    """
    Get day-by-day usage breakdown.
    """
    cutoff = datetime.now() - timedelta(days=days)

    daily = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "requests": 0})

    for r in _usage_records:
        record_date = datetime.fromisoformat(r.timestamp)
        if record_date >= cutoff:
            day_key = record_date.strftime("%Y-%m-%d")
            daily[day_key]["tokens"] += r.total_tokens
            daily[day_key]["cost"] += r.estimated_cost
            daily[day_key]["requests"] += 1

    return {
        "timestamp": datetime.now().isoformat(),
        "days": days,
        "daily_breakdown": {
            day: {
                "tokens": data["tokens"],
                "cost": f"${data['cost']:.4f}",
                "requests": data["requests"]
            }
            for day, data in sorted(daily.items())
        }
    }


def set_session_budget(session_id: str, budget: float) -> Dict[str, Any]:
    """
    Set a budget limit for a session.
    """
    _session_budgets[session_id] = budget

    # Calculate current usage
    current_usage = sum(
        r.estimated_cost for r in _usage_records
        if r.session_id == session_id
    )

    return {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "budget": f"${budget:.4f}",
        "current_usage": f"${current_usage:.4f}",
        "remaining": f"${max(0, budget - current_usage):.4f}",
        "percentage_used": f"{(current_usage / budget * 100):.1f}%" if budget > 0 else "N/A"
    }


def _create_budget_alert(session_id: str, current: float, budget: float):
    """Create a budget alert."""
    alert = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "current_usage": current,
        "budget": budget,
        "percentage": (current / budget * 100) if budget > 0 else 0,
        "message": f"Session {session_id} has used {current / budget * 100:.1f}% of budget"
    }
    _budget_alerts.append(alert)

    # Keep only last 50 alerts
    if len(_budget_alerts) > 50:
        _budget_alerts.pop(0)


def get_budget_alerts() -> Dict[str, Any]:
    """Get recent budget alerts."""
    return {
        "timestamp": datetime.now().isoformat(),
        "total_alerts": len(_budget_alerts),
        "alerts": _budget_alerts[-20:]  # Last 20
    }


def get_optimization_recommendations() -> Dict[str, Any]:
    """
    Analyze usage patterns and provide cost optimization recommendations.
    """
    if not _usage_records:
        return {
            "timestamp": datetime.now().isoformat(),
            "message": "Not enough usage data for recommendations"
        }

    recommendations = []

    # Analyze local vs cloud usage
    local_count = sum(1 for r in _usage_records if r.is_local)
    cloud_count = len(_usage_records) - local_count

    if cloud_count > local_count * 2:
        recommendations.append({
            "type": "use_local_models",
            "priority": "high",
            "message": "Consider using local models more often for routine tasks",
            "potential_savings": "Up to 95% on token costs"
        })

    # Check for expensive tasks
    by_task = defaultdict(list)
    for r in _usage_records:
        by_task[r.task_type or "unknown"].append(r)

    for task, records in by_task.items():
        avg_tokens = sum(r.total_tokens for r in records) / len(records)
        if avg_tokens > 2000:
            recommendations.append({
                "type": "optimize_prompts",
                "priority": "medium",
                "task_type": task,
                "message": f"Tasks of type '{task}' use high token counts (avg {avg_tokens:.0f})",
                "suggestion": "Consider optimizing prompts or using summarization"
            })

    # Check for model efficiency
    by_model = defaultdict(list)
    for r in _usage_records:
        by_model[r.model_id].append(r)

    for model, records in by_model.items():
        avg_latency = sum(r.latency_ms for r in records) / len(records)
        if avg_latency > 5000:  # 5 seconds
            recommendations.append({
                "type": "slow_model",
                "priority": "low",
                "model": model,
                "message": f"Model '{model}' has high average latency ({avg_latency:.0f}ms)",
                "suggestion": "Consider using a faster model for time-sensitive tasks"
            })

    # Cache usage recommendation
    if len(_usage_records) > 100:
        recommendations.append({
            "type": "enable_caching",
            "priority": "medium",
            "message": "High request volume detected",
            "suggestion": "Consider enabling semantic caching for repeated queries"
        })

    return {
        "timestamp": datetime.now().isoformat(),
        "total_records_analyzed": len(_usage_records),
        "local_percentage": f"{local_count / len(_usage_records) * 100:.1f}%" if _usage_records else "N/A",
        "recommendations": recommendations
    }


def get_hourly_usage(hours: int = 24) -> Dict[str, Any]:
    """
    Get hourly usage breakdown.
    """
    cutoff = datetime.now() - timedelta(hours=hours)

    hourly = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "requests": 0})

    for r in _usage_records:
        record_time = datetime.fromisoformat(r.timestamp)
        if record_time >= cutoff:
            hour_key = record_time.strftime("%Y-%m-%d %H:00")
            hourly[hour_key]["tokens"] += r.total_tokens
            hourly[hour_key]["cost"] += r.estimated_cost
            hourly[hour_key]["requests"] += 1

    return {
        "timestamp": datetime.now().isoformat(),
        "hours": hours,
        "hourly_breakdown": {
            hour: {
                "tokens": data["tokens"],
                "cost": f"${data['cost']:.4f}",
                "requests": data["requests"]
            }
            for hour, data in sorted(hourly.items())
        }
    }


def get_cost_projection(days_ahead: int = 30) -> Dict[str, Any]:
    """
    Project future costs based on recent usage.
    """
    # Use last 7 days as baseline
    week_ago = datetime.now() - timedelta(days=7)
    recent = [
        r for r in _usage_records
        if datetime.fromisoformat(r.timestamp) >= week_ago
    ]

    if not recent:
        return {
            "timestamp": datetime.now().isoformat(),
            "message": "Not enough recent data for projection"
        }

    # Calculate daily average
    daily_cost = sum(r.estimated_cost for r in recent) / 7
    daily_tokens = sum(r.total_tokens for r in recent) / 7
    daily_requests = len(recent) / 7

    projected_cost = daily_cost * days_ahead
    projected_tokens = daily_tokens * days_ahead

    return {
        "timestamp": datetime.now().isoformat(),
        "projection_days": days_ahead,
        "based_on_days": 7,
        "daily_averages": {
            "cost": f"${daily_cost:.4f}",
            "tokens": int(daily_tokens),
            "requests": f"{daily_requests:.1f}"
        },
        "projections": {
            "cost": f"${projected_cost:.4f}",
            "tokens": int(projected_tokens)
        },
        "notes": "Projections based on last 7 days of usage"
    }


def clear_usage_history() -> Dict[str, Any]:
    """Clear all usage history."""
    count = len(_usage_records)
    _usage_records.clear()
    _budget_alerts.clear()

    return {
        "timestamp": datetime.now().isoformat(),
        "cleared": True,
        "records_removed": count
    }


def export_usage_data(format: str = "json") -> Dict[str, Any]:
    """
    Export usage data.
    """
    if format == "json":
        data = [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "model_id": r.model_id,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_tokens": r.total_tokens,
                "latency_ms": r.latency_ms,
                "is_local": r.is_local,
                "estimated_cost": r.estimated_cost,
                "session_id": r.session_id,
                "task_type": r.task_type
            }
            for r in _usage_records
        ]
        return {
            "timestamp": datetime.now().isoformat(),
            "format": "json",
            "record_count": len(data),
            "data": data
        }
    elif format == "csv":
        headers = "id,timestamp,model_id,prompt_tokens,completion_tokens,total_tokens,latency_ms,is_local,estimated_cost,session_id,task_type"
        rows = [headers]
        for r in _usage_records:
            rows.append(
                f"{r.id},{r.timestamp},{r.model_id},{r.prompt_tokens},{r.completion_tokens},"
                f"{r.total_tokens},{r.latency_ms},{r.is_local},{r.estimated_cost},"
                f"{r.session_id or ''},{r.task_type or ''}"
            )
        return {
            "timestamp": datetime.now().isoformat(),
            "format": "csv",
            "record_count": len(_usage_records),
            "csv_data": "\n".join(rows)
        }
    else:
        return {"error": f"Unknown format: {format}"}


__all__ = [
    "record_usage",
    "get_usage_summary",
    "compare_local_vs_cloud",
    "get_daily_breakdown",
    "set_session_budget",
    "get_budget_alerts",
    "get_optimization_recommendations",
    "get_hourly_usage",
    "get_cost_projection",
    "clear_usage_history",
    "export_usage_data",
    "CLOUD_COSTS_PER_1K"
]
