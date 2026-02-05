"""
Msty Admin MCP - Intelligent Auto-Router (Phase 26)

Zero-config smart routing - analyzes requests and routes to optimal model.
Learns from usage patterns and optimizes for cost/quality tradeoffs.

Features:
- Task classification (coding, reasoning, creative, simple, complex)
- Model capability matching
- Cost-aware routing
- Learning from success/failure patterns
- Confidence-based decisions
"""

import json
import logging
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .paths import get_msty_paths
from .network import make_api_request, get_available_service_ports
from .tagging import get_model_tags

logger = logging.getLogger("msty-admin-mcp")

# Task classification patterns
TASK_PATTERNS = {
    "coding": {
        "keywords": ["code", "function", "class", "bug", "fix", "implement", "debug", "refactor",
                     "python", "javascript", "typescript", "java", "rust", "go", "sql", "api",
                     "error", "exception", "compile", "syntax", "algorithm", "data structure"],
        "weight": 1.0
    },
    "reasoning": {
        "keywords": ["explain", "why", "how", "analyze", "compare", "evaluate", "think",
                     "reason", "logic", "deduce", "infer", "conclude", "argument", "proof",
                     "math", "calculate", "solve", "equation"],
        "weight": 1.0
    },
    "creative": {
        "keywords": ["write", "story", "poem", "creative", "imagine", "describe", "narrative",
                     "fiction", "character", "dialogue", "scene", "plot", "metaphor"],
        "weight": 1.0
    },
    "translation": {
        "keywords": ["translate", "spanish", "french", "german", "chinese", "japanese",
                     "korean", "portuguese", "italian", "russian", "arabic", "language"],
        "weight": 1.0
    },
    "summarization": {
        "keywords": ["summarize", "summary", "tldr", "brief", "condense", "key points",
                     "main ideas", "overview", "recap"],
        "weight": 1.0
    },
    "simple": {
        "keywords": ["what is", "define", "list", "name", "when", "where", "who",
                     "yes or no", "true or false", "quick", "simple", "basic"],
        "weight": 0.5
    },
    "complex": {
        "keywords": ["design", "architect", "system", "comprehensive", "detailed",
                     "in-depth", "thorough", "complete", "full", "extensive", "research"],
        "weight": 1.5
    }
}

# Model capability scores (higher = better for that task)
MODEL_CAPABILITIES = {
    "coding": {
        "patterns": ["code", "coder", "deepseek-coder", "codestral", "starcoder", "qwen"],
        "fallback_score": 0.6
    },
    "reasoning": {
        "patterns": ["think", "r1", "qwq", "deepseek-r", "o1", "reason"],
        "fallback_score": 0.5
    },
    "creative": {
        "patterns": ["creative", "story", "mistral", "llama", "claude"],
        "fallback_score": 0.7
    },
    "translation": {
        "patterns": ["mistral", "llama", "multilingual", "translate"],
        "fallback_score": 0.6
    },
    "fast": {
        "patterns": ["mini", "tiny", "small", "flash", "lite", "nano", "phi", "gemma-2b"],
        "fallback_score": 0.3
    }
}

# Routing history storage
_routing_history: List[Dict[str, Any]] = []
_model_success_rates: Dict[str, Dict[str, float]] = {}


def classify_task(prompt: str) -> Dict[str, float]:
    """
    Classify a task based on the prompt content.

    Returns a dictionary of task types with confidence scores.
    """
    prompt_lower = prompt.lower()
    scores: Dict[str, float] = {}

    for task_type, config in TASK_PATTERNS.items():
        score = 0.0
        matches = 0

        for keyword in config["keywords"]:
            if keyword in prompt_lower:
                matches += 1
                score += config["weight"]

        if matches > 0:
            # Normalize by number of keywords (partial match bonus)
            scores[task_type] = min(1.0, score / len(config["keywords"]) * 5)

    # If no matches, default to "general"
    if not scores:
        scores["general"] = 0.5

    return scores


def estimate_complexity(prompt: str) -> str:
    """
    Estimate task complexity based on prompt characteristics.

    Returns: "trivial", "simple", "moderate", "complex", "very_complex"
    """
    # Length-based estimation
    word_count = len(prompt.split())

    # Check for complexity indicators
    has_code = bool(re.search(r'```|def |class |function |import ', prompt))
    has_multiple_questions = prompt.count('?') > 2
    has_requirements = any(word in prompt.lower() for word in
                          ["must", "should", "require", "need to", "make sure"])

    complexity_score = 0

    # Word count contribution
    if word_count < 20:
        complexity_score += 1
    elif word_count < 50:
        complexity_score += 2
    elif word_count < 150:
        complexity_score += 3
    elif word_count < 300:
        complexity_score += 4
    else:
        complexity_score += 5

    # Feature contributions
    if has_code:
        complexity_score += 2
    if has_multiple_questions:
        complexity_score += 1
    if has_requirements:
        complexity_score += 1

    # Map to complexity level
    if complexity_score <= 2:
        return "trivial"
    elif complexity_score <= 4:
        return "simple"
    elif complexity_score <= 6:
        return "moderate"
    elif complexity_score <= 8:
        return "complex"
    else:
        return "very_complex"


def score_model_for_task(model_id: str, task_scores: Dict[str, float],
                         complexity: str) -> float:
    """
    Score how well a model matches the task requirements.
    """
    model_lower = model_id.lower()
    total_score = 0.0

    # Get model tags for better matching
    try:
        tags = get_model_tags(model_id)
    except:
        tags = []

    # Score based on task type matches
    for task_type, task_confidence in task_scores.items():
        if task_type in MODEL_CAPABILITIES:
            capability = MODEL_CAPABILITIES[task_type]

            # Check if model matches capability patterns
            pattern_match = any(p in model_lower for p in capability["patterns"])
            tag_match = any(p in tags for p in capability["patterns"])

            if pattern_match or tag_match:
                total_score += task_confidence * 1.0
            else:
                total_score += task_confidence * capability["fallback_score"]

    # Adjust for complexity
    is_small_model = any(p in model_lower for p in ["mini", "tiny", "small", "3b", "1b", "2b"])
    is_large_model = any(p in model_lower for p in ["70b", "72b", "32b", "34b", "large"])

    if complexity in ["trivial", "simple"]:
        if is_small_model:
            total_score *= 1.3  # Prefer small models for simple tasks
        elif is_large_model:
            total_score *= 0.8  # Penalize large models for simple tasks (overkill)
    elif complexity in ["complex", "very_complex"]:
        if is_large_model:
            total_score *= 1.3  # Prefer large models for complex tasks
        elif is_small_model:
            total_score *= 0.6  # Penalize small models for complex tasks

    # Factor in historical success rate
    if model_id in _model_success_rates:
        for task_type in task_scores:
            if task_type in _model_success_rates[model_id]:
                historical_rate = _model_success_rates[model_id][task_type]
                total_score *= (0.5 + historical_rate * 0.5)  # Blend with history

    return total_score


def get_available_models() -> List[Dict[str, Any]]:
    """
    Get list of available models from all services.
    """
    models = []
    ports = get_available_service_ports()

    for service, port in ports.items():
        try:
            response = make_api_request(f"http://127.0.0.1:{port}/v1/models")
            if response and "data" in response:
                for model in response["data"]:
                    model["service"] = service
                    model["port"] = port
                    models.append(model)
        except Exception as e:
            logger.debug(f"Could not get models from {service}: {e}")

    return models


def route_request(
    prompt: str,
    prefer_speed: bool = False,
    prefer_quality: bool = False,
    max_cost_tier: str = "any",
    excluded_models: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Intelligently route a request to the optimal model.

    Args:
        prompt: The user's prompt/request
        prefer_speed: Prioritize faster models
        prefer_quality: Prioritize higher quality models
        max_cost_tier: "free" (local only), "low", "medium", "high", "any"
        excluded_models: Models to exclude from consideration

    Returns:
        Routing decision with model recommendation and reasoning
    """
    timestamp = datetime.now().isoformat()
    excluded = excluded_models or []

    # Classify the task
    task_scores = classify_task(prompt)
    complexity = estimate_complexity(prompt)
    primary_task = max(task_scores, key=task_scores.get) if task_scores else "general"

    # Get available models
    available_models = get_available_models()

    if not available_models:
        return {
            "timestamp": timestamp,
            "routed": False,
            "error": "No models available",
            "task_classification": task_scores,
            "complexity": complexity,
            "recommendation": "Start Msty Studio or load a model"
        }

    # Score each model
    model_scores = []
    for model in available_models:
        model_id = model.get("id", "unknown")

        if model_id in excluded:
            continue

        score = score_model_for_task(model_id, task_scores, complexity)

        # Apply preference adjustments
        if prefer_speed:
            is_fast = any(p in model_id.lower() for p in ["mini", "tiny", "small", "flash", "lite"])
            score *= 1.5 if is_fast else 0.8

        if prefer_quality:
            is_quality = any(p in model_id.lower() for p in ["70b", "72b", "32b", "large", "pro"])
            score *= 1.5 if is_quality else 0.8

        model_scores.append({
            "model_id": model_id,
            "service": model.get("service", "unknown"),
            "port": model.get("port"),
            "score": score,
            "is_local": True  # All Msty models are local
        })

    if not model_scores:
        return {
            "timestamp": timestamp,
            "routed": False,
            "error": "No suitable models after filtering",
            "task_classification": task_scores,
            "complexity": complexity
        }

    # Sort by score (highest first)
    model_scores.sort(key=lambda x: x["score"], reverse=True)

    # Select best model
    selected = model_scores[0]
    alternatives = model_scores[1:4] if len(model_scores) > 1 else []

    # Generate reasoning
    reasoning = []
    reasoning.append(f"Task classified as: {primary_task} (confidence: {task_scores.get(primary_task, 0):.2f})")
    reasoning.append(f"Complexity estimated as: {complexity}")
    reasoning.append(f"Selected {selected['model_id']} with score {selected['score']:.2f}")

    if prefer_speed:
        reasoning.append("Speed preference applied")
    if prefer_quality:
        reasoning.append("Quality preference applied")

    result = {
        "timestamp": timestamp,
        "routed": True,
        "selected_model": selected["model_id"],
        "service": selected["service"],
        "port": selected["port"],
        "score": selected["score"],
        "task_classification": task_scores,
        "primary_task": primary_task,
        "complexity": complexity,
        "reasoning": reasoning,
        "alternatives": [
            {"model_id": m["model_id"], "score": m["score"]}
            for m in alternatives
        ],
        "is_local": True,
        "estimated_cost": "free"
    }

    # Record routing decision
    _routing_history.append({
        "timestamp": timestamp,
        "prompt_hash": hashlib.md5(prompt[:100].encode()).hexdigest(),
        "selected_model": selected["model_id"],
        "task_type": primary_task,
        "complexity": complexity
    })

    return result


def record_routing_outcome(
    model_id: str,
    task_type: str,
    success: bool,
    user_rating: Optional[int] = None
) -> Dict[str, Any]:
    """
    Record the outcome of a routing decision for learning.

    Args:
        model_id: The model that was used
        task_type: The classified task type
        success: Whether the response was successful
        user_rating: Optional 1-5 rating from user
    """
    global _model_success_rates

    if model_id not in _model_success_rates:
        _model_success_rates[model_id] = {}

    if task_type not in _model_success_rates[model_id]:
        _model_success_rates[model_id][task_type] = 0.5  # Start at neutral

    # Update success rate with exponential moving average
    current_rate = _model_success_rates[model_id][task_type]
    outcome_value = 1.0 if success else 0.0

    if user_rating is not None:
        outcome_value = user_rating / 5.0

    # EMA with alpha = 0.2 (gives more weight to recent outcomes)
    new_rate = 0.8 * current_rate + 0.2 * outcome_value
    _model_success_rates[model_id][task_type] = new_rate

    return {
        "timestamp": datetime.now().isoformat(),
        "model_id": model_id,
        "task_type": task_type,
        "previous_rate": current_rate,
        "new_rate": new_rate,
        "recorded": True
    }


def get_routing_stats() -> Dict[str, Any]:
    """
    Get statistics about routing decisions and model performance.
    """
    total_routings = len(_routing_history)

    # Count by task type
    task_counts: Dict[str, int] = {}
    model_counts: Dict[str, int] = {}

    for entry in _routing_history:
        task = entry.get("task_type", "unknown")
        model = entry.get("selected_model", "unknown")

        task_counts[task] = task_counts.get(task, 0) + 1
        model_counts[model] = model_counts.get(model, 0) + 1

    return {
        "timestamp": datetime.now().isoformat(),
        "total_routings": total_routings,
        "routings_by_task": task_counts,
        "routings_by_model": model_counts,
        "model_success_rates": _model_success_rates,
        "learning_enabled": True
    }


def get_model_recommendation(task_description: str) -> Dict[str, Any]:
    """
    Get a model recommendation for a task description without executing.
    """
    return route_request(task_description)


def clear_routing_history() -> Dict[str, Any]:
    """
    Clear routing history and learned patterns.
    """
    global _routing_history, _model_success_rates

    old_count = len(_routing_history)
    _routing_history = []
    _model_success_rates = {}

    return {
        "timestamp": datetime.now().isoformat(),
        "cleared": True,
        "entries_removed": old_count
    }


__all__ = [
    "classify_task",
    "estimate_complexity",
    "route_request",
    "record_routing_outcome",
    "get_routing_stats",
    "get_model_recommendation",
    "clear_routing_history",
    "TASK_PATTERNS",
    "MODEL_CAPABILITIES"
]
