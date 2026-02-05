"""
Msty Admin MCP - A/B Testing Framework (Phase 32)

Run experiments on models to compare performance.
Track metrics, collect feedback, and determine winners.

Features:
- Experiment definition and management
- Multi-model comparison
- Statistical analysis
- User feedback collection
- Results visualization
"""

import json
import logging
import uuid
import time
import statistics
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from .network import make_api_request, get_available_service_ports

logger = logging.getLogger("msty-admin-mcp")


class ExperimentStatus(Enum):
    """Experiment lifecycle status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentResult:
    """Result from a single model in an experiment."""
    model_id: str
    prompt: str
    response: str
    latency_ms: float
    tokens_generated: int
    timestamp: str
    user_rating: Optional[int] = None
    feedback: Optional[str] = None


@dataclass
class Experiment:
    """An A/B testing experiment."""
    id: str
    name: str
    description: str
    models: List[str]
    prompts: List[str]
    status: ExperimentStatus
    created_at: str
    metrics: List[str]
    results: List[ExperimentResult] = field(default_factory=list)
    winner: Optional[str] = None
    completed_at: Optional[str] = None


# Storage
_experiments: Dict[str, Experiment] = {}


def create_experiment(
    name: str,
    models: List[str],
    prompts: List[str],
    description: str = "",
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a new A/B testing experiment.

    Args:
        name: Experiment name
        models: List of model IDs to compare
        prompts: List of test prompts
        description: Experiment description
        metrics: Metrics to track (default: latency, quality)
    """
    experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now().isoformat()

    if len(models) < 2:
        return {"error": "Need at least 2 models to compare"}

    if not prompts:
        return {"error": "Need at least 1 test prompt"}

    experiment = Experiment(
        id=experiment_id,
        name=name,
        description=description,
        models=models,
        prompts=prompts,
        status=ExperimentStatus.DRAFT,
        created_at=timestamp,
        metrics=metrics or ["latency", "response_length", "user_rating"]
    )

    _experiments[experiment_id] = experiment

    return {
        "timestamp": timestamp,
        "experiment_id": experiment_id,
        "name": name,
        "models": models,
        "prompts_count": len(prompts),
        "status": experiment.status.value,
        "created": True
    }


def run_experiment(
    experiment_id: str,
    parallel: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> Dict[str, Any]:
    """
    Run an experiment, collecting responses from all models.
    """
    if experiment_id not in _experiments:
        return {"error": f"Experiment {experiment_id} not found"}

    experiment = _experiments[experiment_id]
    experiment.status = ExperimentStatus.RUNNING

    # Get available ports
    ports = get_available_service_ports()
    default_port = ports.get("local_ai", 11964)

    results = []

    def run_single_test(model_id: str, prompt: str) -> ExperimentResult:
        """Run a single test."""
        start_time = time.time()

        try:
            response = make_api_request(
                f"http://127.0.0.1:{default_port}/v1/chat/completions",
                method="POST",
                data={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )

            latency_ms = (time.time() - start_time) * 1000

            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"]
                tokens = response.get("usage", {}).get("completion_tokens", len(content.split()))

                return ExperimentResult(
                    model_id=model_id,
                    prompt=prompt,
                    response=content,
                    latency_ms=latency_ms,
                    tokens_generated=tokens,
                    timestamp=datetime.now().isoformat()
                )
            else:
                return ExperimentResult(
                    model_id=model_id,
                    prompt=prompt,
                    response="ERROR: No response",
                    latency_ms=latency_ms,
                    tokens_generated=0,
                    timestamp=datetime.now().isoformat()
                )

        except Exception as e:
            return ExperimentResult(
                model_id=model_id,
                prompt=prompt,
                response=f"ERROR: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                tokens_generated=0,
                timestamp=datetime.now().isoformat()
            )

    # Generate all test combinations
    test_cases = [
        (model, prompt)
        for model in experiment.models
        for prompt in experiment.prompts
    ]

    if parallel:
        with ThreadPoolExecutor(max_workers=len(experiment.models)) as executor:
            futures = {
                executor.submit(run_single_test, model, prompt): (model, prompt)
                for model, prompt in test_cases
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                experiment.results.append(result)
    else:
        for model, prompt in test_cases:
            result = run_single_test(model, prompt)
            results.append(result)
            experiment.results.append(result)

    experiment.status = ExperimentStatus.COMPLETED
    experiment.completed_at = datetime.now().isoformat()

    # Analyze results
    analysis = analyze_experiment(experiment_id)

    return {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": experiment_id,
        "status": "completed",
        "total_tests": len(results),
        "results_summary": {
            model: {
                "tests": len([r for r in results if r.model_id == model]),
                "avg_latency_ms": statistics.mean([
                    r.latency_ms for r in results if r.model_id == model
                ]) if results else 0,
                "errors": len([r for r in results if r.model_id == model and "ERROR" in r.response])
            }
            for model in experiment.models
        },
        "analysis": analysis
    }


def analyze_experiment(experiment_id: str) -> Dict[str, Any]:
    """
    Analyze experiment results and determine winner.
    """
    if experiment_id not in _experiments:
        return {"error": f"Experiment {experiment_id} not found"}

    experiment = _experiments[experiment_id]

    if not experiment.results:
        return {"error": "No results to analyze"}

    # Group results by model
    model_results: Dict[str, List[ExperimentResult]] = {}
    for model in experiment.models:
        model_results[model] = [r for r in experiment.results if r.model_id == model]

    # Calculate metrics per model
    model_metrics = {}
    for model, results in model_results.items():
        if not results:
            continue

        valid_results = [r for r in results if "ERROR" not in r.response]

        if valid_results:
            latencies = [r.latency_ms for r in valid_results]
            response_lengths = [len(r.response) for r in valid_results]
            ratings = [r.user_rating for r in valid_results if r.user_rating is not None]

            model_metrics[model] = {
                "total_tests": len(results),
                "successful_tests": len(valid_results),
                "error_rate": (len(results) - len(valid_results)) / len(results),
                "avg_latency_ms": statistics.mean(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "latency_std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "avg_response_length": statistics.mean(response_lengths),
                "avg_user_rating": statistics.mean(ratings) if ratings else None,
                "rating_count": len(ratings)
            }
        else:
            model_metrics[model] = {
                "total_tests": len(results),
                "successful_tests": 0,
                "error_rate": 1.0
            }

    # Determine winner (composite score)
    scores = {}
    for model, metrics in model_metrics.items():
        if metrics.get("successful_tests", 0) == 0:
            scores[model] = 0
            continue

        # Score components (normalized 0-1, higher is better)
        success_score = 1 - metrics.get("error_rate", 1)

        # Latency score (lower is better, invert)
        all_latencies = [m.get("avg_latency_ms", 0) for m in model_metrics.values() if m.get("avg_latency_ms")]
        if all_latencies:
            max_latency = max(all_latencies)
            latency_score = 1 - (metrics.get("avg_latency_ms", max_latency) / max_latency) if max_latency > 0 else 0
        else:
            latency_score = 0

        # Rating score
        rating = metrics.get("avg_user_rating")
        rating_score = rating / 5.0 if rating else 0.5  # Default to middle if no ratings

        # Composite score
        scores[model] = (success_score * 0.4) + (latency_score * 0.3) + (rating_score * 0.3)

    # Determine winner
    if scores:
        winner = max(scores, key=scores.get)
        experiment.winner = winner
    else:
        winner = None

    return {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": experiment_id,
        "model_metrics": model_metrics,
        "scores": scores,
        "winner": winner,
        "confidence": max(scores.values()) - sorted(scores.values())[-2] if len(scores) > 1 else 1.0
    }


def rate_result(
    experiment_id: str,
    model_id: str,
    prompt: str,
    rating: int,
    feedback: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add a user rating to an experiment result.

    Args:
        experiment_id: The experiment
        model_id: The model being rated
        prompt: The prompt (to identify the specific result)
        rating: 1-5 rating
        feedback: Optional text feedback
    """
    if experiment_id not in _experiments:
        return {"error": f"Experiment {experiment_id} not found"}

    if rating < 1 or rating > 5:
        return {"error": "Rating must be 1-5"}

    experiment = _experiments[experiment_id]

    # Find matching result
    for result in experiment.results:
        if result.model_id == model_id and result.prompt == prompt:
            result.user_rating = rating
            result.feedback = feedback
            return {
                "timestamp": datetime.now().isoformat(),
                "rated": True,
                "experiment_id": experiment_id,
                "model_id": model_id,
                "rating": rating
            }

    return {"error": "Result not found"}


def get_experiment(experiment_id: str) -> Dict[str, Any]:
    """Get details of an experiment."""
    if experiment_id not in _experiments:
        return {"error": f"Experiment {experiment_id} not found"}

    experiment = _experiments[experiment_id]

    return {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": experiment.id,
        "name": experiment.name,
        "description": experiment.description,
        "models": experiment.models,
        "prompts_count": len(experiment.prompts),
        "status": experiment.status.value,
        "created_at": experiment.created_at,
        "completed_at": experiment.completed_at,
        "results_count": len(experiment.results),
        "winner": experiment.winner,
        "metrics": experiment.metrics
    }


def list_experiments() -> Dict[str, Any]:
    """List all experiments."""
    experiments_list = []

    for exp_id, exp in _experiments.items():
        experiments_list.append({
            "experiment_id": exp_id,
            "name": exp.name,
            "status": exp.status.value,
            "models": exp.models,
            "created_at": exp.created_at,
            "winner": exp.winner
        })

    return {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(experiments_list),
        "experiments": experiments_list
    }


def get_experiment_results(experiment_id: str) -> Dict[str, Any]:
    """Get detailed results from an experiment."""
    if experiment_id not in _experiments:
        return {"error": f"Experiment {experiment_id} not found"}

    experiment = _experiments[experiment_id]

    results = [
        {
            "model_id": r.model_id,
            "prompt": r.prompt[:100] + "..." if len(r.prompt) > 100 else r.prompt,
            "response_preview": r.response[:200] + "..." if len(r.response) > 200 else r.response,
            "latency_ms": r.latency_ms,
            "tokens": r.tokens_generated,
            "user_rating": r.user_rating,
            "feedback": r.feedback,
            "timestamp": r.timestamp
        }
        for r in experiment.results
    ]

    return {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": experiment_id,
        "total_results": len(results),
        "results": results
    }


def delete_experiment(experiment_id: str) -> Dict[str, Any]:
    """Delete an experiment."""
    if experiment_id not in _experiments:
        return {"error": f"Experiment {experiment_id} not found"}

    del _experiments[experiment_id]

    return {
        "timestamp": datetime.now().isoformat(),
        "deleted": True,
        "experiment_id": experiment_id
    }


def compare_models_quick(
    models: List[str],
    prompt: str,
    runs: int = 3
) -> Dict[str, Any]:
    """
    Quick comparison of models on a single prompt.
    Runs multiple times for statistical significance.
    """
    # Create temporary experiment
    result = create_experiment(
        name=f"Quick comparison {datetime.now().strftime('%H:%M')}",
        models=models,
        prompts=[prompt] * runs,
        description="Quick model comparison"
    )

    if "error" in result:
        return result

    # Run it
    return run_experiment(result["experiment_id"])


__all__ = [
    "ExperimentStatus",
    "create_experiment",
    "run_experiment",
    "analyze_experiment",
    "rate_result",
    "get_experiment",
    "list_experiments",
    "get_experiment_results",
    "delete_experiment",
    "compare_models_quick"
]
