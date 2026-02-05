"""
Msty Admin MCP - Cascade Execution (Phase 33)

Chain models with automatic escalation based on confidence.
Only uses more expensive/capable models when needed.

Features:
- Tiered model chains
- Confidence-based escalation
- Automatic fallback
- Cost optimization
- Response quality verification
"""

import json
import logging
import time
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .network import make_api_request, get_available_service_ports
from .smart_router import classify_task, estimate_complexity

logger = logging.getLogger("msty-admin-mcp")


@dataclass
class CascadeResult:
    """Result from a cascade execution."""
    tier_reached: int
    model_used: str
    response: str
    confidence: float
    total_latency_ms: float
    escalation_reasons: List[str]
    attempts: List[Dict[str, Any]]


# Default cascade tiers (from fastest/cheapest to most capable)
DEFAULT_CASCADE_TIERS = [
    {
        "tier": 1,
        "name": "fast",
        "model_patterns": ["mini", "tiny", "small", "phi", "gemma-2b", "3b"],
        "confidence_threshold": 0.85,
        "max_tokens": 512,
        "description": "Fast, lightweight models for simple tasks"
    },
    {
        "tier": 2,
        "name": "balanced",
        "model_patterns": ["7b", "8b", "qwen", "mistral", "llama"],
        "confidence_threshold": 0.75,
        "max_tokens": 1024,
        "description": "Balanced models for moderate complexity"
    },
    {
        "tier": 3,
        "name": "capable",
        "model_patterns": ["32b", "34b", "70b", "72b", "large", "pro"],
        "confidence_threshold": 0.6,
        "max_tokens": 2048,
        "description": "Capable models for complex tasks"
    },
    {
        "tier": 4,
        "name": "expert",
        "model_patterns": ["claude", "gpt-4", "opus"],
        "confidence_threshold": 0.0,  # Always accept
        "max_tokens": 4096,
        "description": "Expert models for the most demanding tasks"
    }
]


# Confidence detection patterns
LOW_CONFIDENCE_PATTERNS = [
    r"i'm not sure",
    r"i don't know",
    r"i cannot",
    r"i can't",
    r"unclear",
    r"uncertain",
    r"might be",
    r"possibly",
    r"perhaps",
    r"i think",
    r"maybe",
    r"not certain",
    r"hard to say",
    r"difficult to determine"
]

ERROR_PATTERNS = [
    r"error",
    r"exception",
    r"failed",
    r"unable to",
    r"sorry",
    r"apologize"
]


def estimate_response_confidence(response: str) -> float:
    """
    Estimate confidence level of a response based on language patterns.
    Returns 0.0 to 1.0
    """
    response_lower = response.lower()

    # Check for error patterns
    error_count = sum(1 for pattern in ERROR_PATTERNS if re.search(pattern, response_lower))
    if error_count > 2:
        return 0.2

    # Check for low confidence patterns
    low_conf_count = sum(1 for pattern in LOW_CONFIDENCE_PATTERNS if re.search(pattern, response_lower))

    # Base confidence
    confidence = 0.9

    # Reduce for each low confidence indicator
    confidence -= low_conf_count * 0.1

    # Reduce for very short responses (might be incomplete)
    if len(response) < 50:
        confidence -= 0.2

    # Reduce for error patterns
    confidence -= error_count * 0.15

    # Check for positive confidence indicators
    if any(phrase in response_lower for phrase in ["here is", "the answer is", "to summarize", "in conclusion"]):
        confidence += 0.1

    return max(0.0, min(1.0, confidence))


def find_model_for_tier(tier_config: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    """
    Find an available model matching the tier's patterns.
    Returns (model_id, port) or None.
    """
    ports = get_available_service_ports()

    for service, port in ports.items():
        try:
            response = make_api_request(f"http://127.0.0.1:{port}/v1/models")
            if response and "data" in response:
                for model in response["data"]:
                    model_id = model.get("id", "").lower()
                    if any(pattern in model_id for pattern in tier_config["model_patterns"]):
                        return (model["id"], port)
        except:
            pass

    return None


def execute_with_cascade(
    prompt: str,
    start_tier: int = 1,
    max_tier: int = 3,
    custom_tiers: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute a prompt with cascade escalation.

    Args:
        prompt: The prompt to execute
        start_tier: Which tier to start at (1-4)
        max_tier: Maximum tier to escalate to
        custom_tiers: Custom tier configuration
        temperature: Model temperature
        system_prompt: Optional system prompt
    """
    timestamp = datetime.now()
    tiers = custom_tiers or DEFAULT_CASCADE_TIERS

    # Validate tiers
    start_tier = max(1, min(start_tier, len(tiers)))
    max_tier = max(start_tier, min(max_tier, len(tiers)))

    attempts = []
    escalation_reasons = []
    total_latency = 0

    # Try each tier
    for tier_idx in range(start_tier - 1, max_tier):
        tier_config = tiers[tier_idx]
        tier_num = tier_config["tier"]

        # Find model for this tier
        model_info = find_model_for_tier(tier_config)

        if not model_info:
            escalation_reasons.append(f"Tier {tier_num}: No model available")
            continue

        model_id, port = model_info

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Execute
        start_time = time.time()

        try:
            response = make_api_request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                method="POST",
                data={
                    "model": model_id,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": tier_config["max_tokens"]
                }
            )

            latency_ms = (time.time() - start_time) * 1000
            total_latency += latency_ms

            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"]
                confidence = estimate_response_confidence(content)

                attempt = {
                    "tier": tier_num,
                    "tier_name": tier_config["name"],
                    "model_id": model_id,
                    "response_preview": content[:200],
                    "confidence": confidence,
                    "latency_ms": latency_ms,
                    "threshold": tier_config["confidence_threshold"]
                }
                attempts.append(attempt)

                # Check if confidence meets threshold
                if confidence >= tier_config["confidence_threshold"]:
                    # Success! Return result
                    return {
                        "timestamp": timestamp.isoformat(),
                        "success": True,
                        "tier_reached": tier_num,
                        "tier_name": tier_config["name"],
                        "model_used": model_id,
                        "response": content,
                        "confidence": confidence,
                        "total_latency_ms": total_latency,
                        "escalation_reasons": escalation_reasons,
                        "attempts": attempts,
                        "cost_tier": tier_config["name"]
                    }
                else:
                    # Need to escalate
                    escalation_reasons.append(
                        f"Tier {tier_num} ({model_id}): Confidence {confidence:.2f} < threshold {tier_config['confidence_threshold']}"
                    )
            else:
                escalation_reasons.append(f"Tier {tier_num}: No response from {model_id}")

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            total_latency += latency_ms
            escalation_reasons.append(f"Tier {tier_num}: Error - {str(e)}")

    # If we get here, we've exhausted all tiers
    # Return the best attempt we have
    if attempts:
        best_attempt = max(attempts, key=lambda x: x["confidence"])
        return {
            "timestamp": timestamp.isoformat(),
            "success": False,
            "tier_reached": best_attempt["tier"],
            "tier_name": best_attempt["tier_name"],
            "model_used": best_attempt["model_id"],
            "response": "Best available response (below confidence threshold)",
            "confidence": best_attempt["confidence"],
            "total_latency_ms": total_latency,
            "escalation_reasons": escalation_reasons,
            "attempts": attempts,
            "warning": "All tiers attempted, returning best available response"
        }

    return {
        "timestamp": timestamp.isoformat(),
        "success": False,
        "error": "No models available in any tier",
        "escalation_reasons": escalation_reasons,
        "total_latency_ms": total_latency
    }


def smart_execute(
    prompt: str,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Smart execution that chooses starting tier based on task complexity.
    """
    # Analyze the task
    task_scores = classify_task(prompt)
    complexity = estimate_complexity(prompt)

    # Choose starting tier based on complexity
    tier_map = {
        "trivial": 1,
        "simple": 1,
        "moderate": 2,
        "complex": 2,
        "very_complex": 3
    }

    start_tier = tier_map.get(complexity, 2)

    # Adjust based on task type
    primary_task = max(task_scores, key=task_scores.get) if task_scores else "general"

    # Coding and reasoning tasks might need higher tier
    if primary_task in ["coding", "reasoning"] and complexity in ["moderate", "complex"]:
        start_tier = max(start_tier, 2)

    result = execute_with_cascade(
        prompt,
        start_tier=start_tier,
        temperature=temperature,
        system_prompt=system_prompt
    )

    result["task_analysis"] = {
        "complexity": complexity,
        "primary_task": primary_task,
        "chosen_start_tier": start_tier
    }

    return result


def get_cascade_config() -> Dict[str, Any]:
    """Get current cascade configuration."""
    return {
        "timestamp": datetime.now().isoformat(),
        "tiers": DEFAULT_CASCADE_TIERS,
        "low_confidence_patterns": LOW_CONFIDENCE_PATTERNS,
        "error_patterns": ERROR_PATTERNS
    }


def test_cascade_tiers() -> Dict[str, Any]:
    """Test which models are available at each tier."""
    timestamp = datetime.now().isoformat()

    tier_status = []
    for tier in DEFAULT_CASCADE_TIERS:
        model_info = find_model_for_tier(tier)
        tier_status.append({
            "tier": tier["tier"],
            "name": tier["name"],
            "description": tier["description"],
            "model_available": model_info is not None,
            "model_id": model_info[0] if model_info else None,
            "patterns": tier["model_patterns"]
        })

    available_count = sum(1 for t in tier_status if t["model_available"])

    return {
        "timestamp": timestamp,
        "tiers": tier_status,
        "available_tiers": available_count,
        "total_tiers": len(DEFAULT_CASCADE_TIERS)
    }


__all__ = [
    "execute_with_cascade",
    "smart_execute",
    "estimate_response_confidence",
    "get_cascade_config",
    "test_cascade_tiers",
    "DEFAULT_CASCADE_TIERS"
]
