"""
Msty Admin MCP - Predictive Model Pre-Loading (Phase 30)

Analyzes usage patterns and pre-loads appropriate models
to reduce latency for common workflows.

Features:
- Usage pattern analysis
- Time-based predictions (morning = coding, etc.)
- Calendar integration hints
- Model warming/pre-loading
- Latency optimization
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict

from .paths import get_msty_paths
from .network import make_api_request, get_available_service_ports
from .smart_router import classify_task

logger = logging.getLogger("msty-admin-mcp")


# Usage pattern storage
_usage_patterns: Dict[str, Dict[str, Any]] = {
    "by_hour": defaultdict(lambda: defaultdict(int)),  # hour -> task_type -> count
    "by_day": defaultdict(lambda: defaultdict(int)),   # weekday -> task_type -> count
    "by_model": defaultdict(int),                       # model_id -> total_uses
    "recent_tasks": [],                                 # Last N tasks
    "session_start": None
}

_prediction_config = {
    "lookback_days": 30,
    "min_samples_for_prediction": 5,
    "pre_warm_enabled": False,  # Requires Msty integration
    "prediction_confidence_threshold": 0.6
}


def record_usage(
    task_type: str,
    model_id: str,
    prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Record a usage event for pattern learning.

    Args:
        task_type: Type of task (coding, research, writing, etc.)
        model_id: Model that was used
        prompt: Optional prompt for more detailed analysis
    """
    now = datetime.now()
    hour = now.hour
    weekday = now.strftime("%A").lower()

    # Update patterns
    _usage_patterns["by_hour"][hour][task_type] += 1
    _usage_patterns["by_day"][weekday][task_type] += 1
    _usage_patterns["by_model"][model_id] += 1

    # Track recent tasks
    _usage_patterns["recent_tasks"].append({
        "timestamp": now.isoformat(),
        "task_type": task_type,
        "model_id": model_id,
        "hour": hour,
        "weekday": weekday
    })

    # Keep only last 1000 tasks
    if len(_usage_patterns["recent_tasks"]) > 1000:
        _usage_patterns["recent_tasks"] = _usage_patterns["recent_tasks"][-1000:]

    return {
        "timestamp": now.isoformat(),
        "recorded": True,
        "task_type": task_type,
        "model_id": model_id
    }


def predict_next_task() -> Dict[str, Any]:
    """
    Predict what type of task the user is likely to do next.
    Based on current time, day, and recent patterns.
    """
    now = datetime.now()
    hour = now.hour
    weekday = now.strftime("%A").lower()

    predictions = []

    # Get predictions from hour patterns
    hour_data = dict(_usage_patterns["by_hour"].get(hour, {}))
    if hour_data:
        total = sum(hour_data.values())
        for task_type, count in hour_data.items():
            confidence = count / total if total > 0 else 0
            predictions.append({
                "task_type": task_type,
                "confidence": confidence,
                "source": "time_of_day",
                "reasoning": f"At {hour}:00, you typically do {task_type} tasks"
            })

    # Get predictions from day patterns
    day_data = dict(_usage_patterns["by_day"].get(weekday, {}))
    if day_data:
        total = sum(day_data.values())
        for task_type, count in day_data.items():
            confidence = count / total if total > 0 else 0
            # Check if already in predictions
            existing = next((p for p in predictions if p["task_type"] == task_type), None)
            if existing:
                # Average the confidences
                existing["confidence"] = (existing["confidence"] + confidence) / 2
                existing["source"] = "combined"
            else:
                predictions.append({
                    "task_type": task_type,
                    "confidence": confidence,
                    "source": "day_of_week",
                    "reasoning": f"On {weekday.title()}s, you often do {task_type} tasks"
                })

    # Check recent task momentum (what have you been doing lately?)
    recent = _usage_patterns["recent_tasks"][-10:]
    if recent:
        recent_types = [t["task_type"] for t in recent]
        most_recent = recent_types[-1] if recent_types else None
        if most_recent:
            # Boost confidence for recent task types (momentum)
            for p in predictions:
                if p["task_type"] == most_recent:
                    p["confidence"] = min(1.0, p["confidence"] * 1.3)
                    p["reasoning"] += " (recent momentum)"

    # Sort by confidence
    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    # Filter by threshold
    confident_predictions = [
        p for p in predictions
        if p["confidence"] >= _prediction_config["prediction_confidence_threshold"]
    ]

    return {
        "timestamp": now.isoformat(),
        "predictions": confident_predictions[:3],  # Top 3
        "all_predictions": predictions[:10],
        "context": {
            "hour": hour,
            "weekday": weekday,
            "recent_task_count": len(_usage_patterns["recent_tasks"])
        }
    }


def recommend_models_to_load() -> Dict[str, Any]:
    """
    Recommend which models should be pre-loaded based on predictions.
    """
    prediction = predict_next_task()
    predicted_tasks = prediction.get("predictions", [])

    recommendations = []

    # Map task types to preferred model characteristics
    task_model_preferences = {
        "coding": ["code", "coder", "deepseek", "qwen"],
        "reasoning": ["think", "reason", "r1", "qwq"],
        "creative": ["mistral", "llama", "creative"],
        "writing": ["mistral", "llama", "write"],
        "analysis": ["qwen", "deepseek", "analyze"],
        "general": ["qwen", "llama", "gemma"]
    }

    # Get available models
    available_models = []
    ports = get_available_service_ports()
    for service, port in ports.items():
        try:
            response = make_api_request(f"http://127.0.0.1:{port}/v1/models")
            if response and "data" in response:
                for model in response["data"]:
                    available_models.append({
                        "id": model.get("id"),
                        "service": service,
                        "port": port
                    })
        except:
            pass

    # Match predictions to available models
    for task_pred in predicted_tasks:
        task_type = task_pred["task_type"]
        preferences = task_model_preferences.get(task_type, ["general"])

        for model in available_models:
            model_id_lower = model["id"].lower()
            if any(pref in model_id_lower for pref in preferences):
                recommendations.append({
                    "model_id": model["id"],
                    "service": model["service"],
                    "port": model["port"],
                    "for_task": task_type,
                    "confidence": task_pred["confidence"],
                    "reasoning": task_pred.get("reasoning", "")
                })
                break

    # Add most used models as fallback
    most_used = sorted(
        _usage_patterns["by_model"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    for model_id, uses in most_used:
        if not any(r["model_id"] == model_id for r in recommendations):
            recommendations.append({
                "model_id": model_id,
                "for_task": "frequently_used",
                "confidence": 0.5,
                "reasoning": f"Used {uses} times historically"
            })

    return {
        "timestamp": datetime.now().isoformat(),
        "recommendations": recommendations[:5],
        "based_on_predictions": predicted_tasks,
        "pre_warm_enabled": _prediction_config["pre_warm_enabled"]
    }


def get_usage_summary() -> Dict[str, Any]:
    """
    Get a summary of usage patterns.
    """
    # Most active hours
    hour_totals = {
        hour: sum(tasks.values())
        for hour, tasks in _usage_patterns["by_hour"].items()
    }
    peak_hours = sorted(hour_totals.items(), key=lambda x: x[1], reverse=True)[:5]

    # Most common tasks by day
    day_summaries = {}
    for day, tasks in _usage_patterns["by_day"].items():
        if tasks:
            most_common = max(tasks.items(), key=lambda x: x[1])
            day_summaries[day] = {
                "most_common_task": most_common[0],
                "count": most_common[1]
            }

    # Top models
    top_models = sorted(
        _usage_patterns["by_model"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    return {
        "timestamp": datetime.now().isoformat(),
        "total_recorded_tasks": len(_usage_patterns["recent_tasks"]),
        "peak_hours": [{"hour": h, "count": c} for h, c in peak_hours],
        "day_summaries": day_summaries,
        "top_models": [{"model_id": m, "uses": c} for m, c in top_models],
        "unique_task_types": len(set(
            t["task_type"] for t in _usage_patterns["recent_tasks"]
        ))
    }


def get_hourly_breakdown() -> Dict[str, Any]:
    """
    Get detailed hourly usage breakdown.
    """
    breakdown = {}

    for hour in range(24):
        tasks = dict(_usage_patterns["by_hour"].get(hour, {}))
        if tasks:
            breakdown[f"{hour:02d}:00"] = {
                "total": sum(tasks.values()),
                "tasks": tasks
            }

    return {
        "timestamp": datetime.now().isoformat(),
        "hourly_breakdown": breakdown
    }


def configure_prediction(
    lookback_days: Optional[int] = None,
    min_samples: Optional[int] = None,
    confidence_threshold: Optional[float] = None,
    pre_warm_enabled: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Configure prediction settings.
    """
    if lookback_days is not None:
        _prediction_config["lookback_days"] = max(1, lookback_days)

    if min_samples is not None:
        _prediction_config["min_samples_for_prediction"] = max(1, min_samples)

    if confidence_threshold is not None:
        _prediction_config["prediction_confidence_threshold"] = max(0.1, min(1.0, confidence_threshold))

    if pre_warm_enabled is not None:
        _prediction_config["pre_warm_enabled"] = pre_warm_enabled

    return {
        "timestamp": datetime.now().isoformat(),
        "config": _prediction_config.copy()
    }


def clear_usage_history() -> Dict[str, Any]:
    """
    Clear all usage history.
    """
    _usage_patterns["by_hour"].clear()
    _usage_patterns["by_day"].clear()
    _usage_patterns["by_model"].clear()
    _usage_patterns["recent_tasks"].clear()

    return {
        "timestamp": datetime.now().isoformat(),
        "cleared": True
    }


def start_session() -> Dict[str, Any]:
    """
    Start a new usage session with predictions.
    """
    _usage_patterns["session_start"] = datetime.now().isoformat()

    prediction = predict_next_task()
    recommendations = recommend_models_to_load()

    return {
        "timestamp": datetime.now().isoformat(),
        "session_started": True,
        "predictions": prediction.get("predictions", []),
        "recommended_models": recommendations.get("recommendations", []),
        "message": "Session started. Usage will be tracked for better predictions."
    }


__all__ = [
    "record_usage",
    "predict_next_task",
    "recommend_models_to_load",
    "get_usage_summary",
    "get_hourly_breakdown",
    "configure_prediction",
    "clear_usage_history",
    "start_session"
]
