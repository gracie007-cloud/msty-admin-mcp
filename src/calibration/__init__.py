"""
Phase 5: Tiered AI Workflow / Calibration Tools for Msty Admin MCP

Tools:
- run_calibration_test
- evaluate_response_quality
- identify_handoff_triggers
- get_calibration_history

Created by Pineapple 🍍 AI Administration System
"""

import json
import time
import hashlib
from datetime import datetime
from typing import Optional, List

from ..phase4_5_tools import (
    init_metrics_db,
    record_model_metric,
    save_calibration_result,
    get_calibration_results,
    record_handoff_trigger,
    get_handoff_triggers,
    evaluate_response_heuristic,
    CALIBRATION_PROMPTS,
    QUALITY_RUBRIC
)


def register_phase5_tools(mcp, make_api_request, is_process_running, LOCAL_AI_SERVICE_PORT):
    """Register Phase 5 Tiered AI Workflow / Calibration tools with the MCP server"""
    
    @mcp.tool()
    def run_calibration_test(
        model_id: Optional[str] = None,
        category: str = "general",
        custom_prompt: Optional[str] = None,
        passing_threshold: float = 0.6
    ) -> str:
        """
        Run a calibration test on a local model.
        
        Tests the model's capability in specific categories and records
        results for tracking improvement over time.
        
        Args:
            model_id: Model to test (None = auto-select first available)
            category: Test category:
                - "general": Mixed test (one from each category)
                - "reasoning": Logic and problem-solving
                - "coding": Code generation
                - "writing": Content creation
                - "analysis": Critical thinking
                - "creative": Creative tasks
            custom_prompt: Use a custom prompt instead of built-in tests
            passing_threshold: Minimum score to pass (0.0-1.0, default 0.6)
        
        Returns:
            Test results with quality scores and recommendations
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
            "category": category,
            "passing_threshold": passing_threshold,
            "tests": [],
            "summary": {}
        }
        
        if not is_process_running("MstySidecar"):
            result["error"] = "Sidecar is not running"
            return json.dumps(result, indent=2)
        
        # Auto-select model if not specified
        if not model_id:
            models_response = make_api_request("/v1/models", port=LOCAL_AI_SERVICE_PORT)
            if models_response.get("success"):
                data = models_response.get("data", {})
                if isinstance(data, dict) and "data" in data and data["data"]:
                    model_id = data["data"][0].get("id")
                    result["model_id"] = model_id
                    result["note"] = f"Auto-selected model: {model_id}"
            
            if not model_id:
                result["error"] = "No models available"
                return json.dumps(result, indent=2)
        
        # Determine prompts to test
        prompts_to_test = []
        if custom_prompt:
            prompts_to_test = [(category, custom_prompt)]
        elif category == "general":
            # One from each category
            for cat, prompts in CALIBRATION_PROMPTS.items():
                if prompts:
                    prompts_to_test.append((cat, prompts[0]))
        elif category in CALIBRATION_PROMPTS:
            # All prompts from specified category
            for prompt in CALIBRATION_PROMPTS[category]:
                prompts_to_test.append((category, prompt))
        else:
            result["error"] = f"Unknown category: {category}"
            result["valid_categories"] = ["general"] + list(CALIBRATION_PROMPTS.keys())
            return json.dumps(result, indent=2)
        
        init_metrics_db()
        
        passed_count = 0
        total_score = 0.0
        total_tps = 0.0
        
        for test_category, prompt in prompts_to_test:
            # Generate test ID
            test_id = hashlib.md5(f"{model_id}:{prompt}:{datetime.now().isoformat()}".encode()).hexdigest()[:12]
            
            # Build request
            messages = [{"role": "user", "content": prompt}]
            request_data = {
                "model": model_id,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024,
                "stream": False
            }
            
            start_time = time.time()
            
            response = make_api_request(
                endpoint="/v1/chat/completions",
                port=LOCAL_AI_SERVICE_PORT,
                method="POST",
                data=request_data,
                timeout=120
            )
            
            elapsed_time = time.time() - start_time
            
            test_result = {
                "test_id": test_id,
                "category": test_category,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "success": response.get("success", False),
                "latency_seconds": round(elapsed_time, 2)
            }
            
            if response.get("success"):
                data = response.get("data", {})
                if "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    msg = choice.get("message", {})
                    content = msg.get("content", "") or msg.get("reasoning", "")
                    
                    test_result["response_preview"] = content[:200] + "..." if len(content) > 200 else content
                    test_result["response_length"] = len(content)
                    
                    if "usage" in data:
                        completion_tokens = data["usage"].get("completion_tokens", 0)
                        tps = completion_tokens / max(elapsed_time, 0.1)
                        test_result["tokens_per_second"] = round(tps, 1)
                        total_tps += tps
                    
                    # Evaluate quality
                    evaluation = evaluate_response_heuristic(prompt, content, test_category)
                    test_result["quality_score"] = round(evaluation["score"], 2)
                    test_result["evaluation_notes"] = evaluation["notes"]
                    test_result["passed"] = evaluation["score"] >= passing_threshold
                    
                    total_score += evaluation["score"]
                    if test_result["passed"]:
                        passed_count += 1
                    
                    # Save calibration result
                    save_calibration_result(
                        test_id=test_id,
                        model_id=model_id,
                        prompt_category=test_category,
                        prompt=prompt,
                        local_response=content,
                        quality_score=evaluation["score"],
                        evaluation_notes=json.dumps(evaluation["notes"]),
                        tokens_per_second=test_result.get("tokens_per_second", 0),
                        passed=test_result["passed"]
                    )
                    
                    # Record metric
                    record_model_metric(
                        model_id=model_id,
                        prompt_tokens=int(len(prompt.split()) * 1.3),
                        completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
                        latency_seconds=elapsed_time,
                        success=True,
                        use_case=f"calibration_{test_category}"
                    )
            else:
                test_result["error"] = response.get("error", "Unknown error")
                test_result["passed"] = False
            
            result["tests"].append(test_result)
        
        # Summary
        total_tests = len(prompts_to_test)
        result["summary"] = {
            "total_tests": total_tests,
            "passed": passed_count,
            "failed": total_tests - passed_count,
            "pass_rate": round(passed_count / max(total_tests, 1) * 100, 1),
            "average_score": round(total_score / max(total_tests, 1), 2),
            "average_tokens_per_second": round(total_tps / max(total_tests, 1), 1),
            "overall_passed": passed_count >= total_tests * 0.7  # 70% pass rate
        }
        
        # Recommendations
        recommendations = []
        if result["summary"]["average_score"] < 0.5:
            recommendations.append("Model may not be suitable for this task. Consider a larger model.")
        elif result["summary"]["average_score"] < passing_threshold:
            recommendations.append("Model is borderline. May need persona tuning or prompt engineering.")
        else:
            recommendations.append("Model performs adequately for this task category.")
        
        if result["summary"]["average_tokens_per_second"] < 10:
            recommendations.append("Model is slow. Consider a smaller model for interactive use.")
        
        result["recommendations"] = recommendations
        
        return json.dumps(result, indent=2, default=str)
    
    @mcp.tool()
    def evaluate_response_quality(
        prompt: str,
        response: str,
        category: str = "general"
    ) -> str:
        """
        Evaluate the quality of a model response.
        
        Uses heuristics and scoring rubrics to assess:
        - Accuracy and relevance
        - Completeness
        - Clarity and formatting
        - Category-specific criteria
        
        Args:
            prompt: The original prompt
            response: The model's response
            category: Response category for specific evaluation criteria:
                - "general", "reasoning", "coding", "writing", "analysis", "creative"
        
        Returns:
            Quality score (0.0-1.0) with detailed breakdown
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "response_length": len(response)
        }
        
        # Run evaluation
        evaluation = evaluate_response_heuristic(prompt, response, category)
        
        result["quality_score"] = round(evaluation["score"], 3)
        result["passed"] = evaluation["passed"]
        result["criteria_scores"] = {k: round(v, 2) for k, v in evaluation.get("criteria_scores", {}).items()}
        result["notes"] = evaluation["notes"]
        
        # Add rubric reference
        result["evaluation_rubric"] = QUALITY_RUBRIC
        
        # Interpretation
        score = evaluation["score"]
        if score >= 0.8:
            result["interpretation"] = "Excellent - High quality response"
        elif score >= 0.6:
            result["interpretation"] = "Good - Acceptable for most uses"
        elif score >= 0.4:
            result["interpretation"] = "Fair - May need improvement"
        else:
            result["interpretation"] = "Poor - Significant issues detected"
        
        return json.dumps(result, indent=2)
    
    @mcp.tool()
    def identify_handoff_triggers(
        analyse_recent: bool = True,
        add_pattern: Optional[str] = None,
        pattern_type: Optional[str] = None
    ) -> str:
        """
        Identify and manage patterns that should trigger escalation to Claude.
        
        Tracks prompt patterns where local models underperform and should
        hand off to Claude for better results.
        
        Args:
            analyse_recent: Analyse recent calibration tests for triggers
            add_pattern: Manually add a trigger pattern description
            pattern_type: Type of pattern being added:
                - "complexity": Complex reasoning required
                - "domain": Specific domain knowledge needed
                - "quality": High quality requirement
                - "safety": Safety-sensitive content
                - "creativity": Advanced creative tasks
        
        Returns:
            List of identified handoff triggers with confidence scores
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "triggers": [],
            "analysis": {}
        }
        
        init_metrics_db()
        
        # Add manual pattern if specified
        if add_pattern and pattern_type:
            record_handoff_trigger(
                pattern_type=pattern_type,
                pattern_description=add_pattern,
                confidence=0.7  # Manual patterns start with higher confidence
            )
            result["added_pattern"] = {
                "type": pattern_type,
                "description": add_pattern
            }
        
        # Analyse recent calibration tests
        if analyse_recent:
            calibration_results = get_calibration_results(limit=100)
            
            # Find patterns in failed tests
            failed_tests = [r for r in calibration_results if not r.get("passed")]
            
            category_failures = {}
            for test in failed_tests:
                cat = test.get("prompt_category", "unknown")
                category_failures[cat] = category_failures.get(cat, 0) + 1
            
            result["analysis"]["failed_tests_count"] = len(failed_tests)
            result["analysis"]["total_tests_analysed"] = len(calibration_results)
            result["analysis"]["failure_by_category"] = category_failures
            
            # Identify trigger patterns
            for cat, count in category_failures.items():
                if count >= 3:  # Pattern threshold
                    confidence = min(count / 10, 1.0)
                    record_handoff_trigger(
                        pattern_type="category_failure",
                        pattern_description=f"Local model frequently fails {cat} tasks",
                        confidence=confidence
                    )
        
        # Get all triggers
        triggers = get_handoff_triggers(active_only=True)
        result["triggers"] = triggers
        result["trigger_count"] = len(triggers)
        
        # Recommendations
        recommendations = []
        if len(triggers) == 0:
            recommendations.append("No handoff triggers identified yet. Run more calibration tests.")
        else:
            high_confidence = [t for t in triggers if t.get("confidence", 0) >= 0.7]
            if high_confidence:
                recommendations.append(f"{len(high_confidence)} high-confidence triggers found. Consider always escalating these to Claude.")
            
            top_trigger = triggers[0] if triggers else None
            if top_trigger:
                recommendations.append(f"Most common trigger: {top_trigger.get('pattern_description', 'Unknown')}")
        
        result["recommendations"] = recommendations
        
        return json.dumps(result, indent=2, default=str)
    
    @mcp.tool()
    def get_calibration_history(
        model_id: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 50
    ) -> str:
        """
        Get historical calibration test results.
        
        Tracks improvement over time and identifies trends.
        
        Args:
            model_id: Filter by specific model (None = all models)
            category: Filter by test category (None = all categories)
            limit: Maximum results to return (default: 50)
        
        Returns:
            Historical test results with trends and statistics
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "filters": {
                "model_id": model_id,
                "category": category,
                "limit": limit
            },
            "history": [],
            "statistics": {}
        }
        
        init_metrics_db()
        
        # Get calibration results
        all_results = get_calibration_results(model_id=model_id, limit=limit)
        
        # Filter by category if specified
        if category:
            all_results = [r for r in all_results if r.get("prompt_category") == category]
        
        result["history"] = all_results
        result["total_tests"] = len(all_results)
        
        if all_results:
            # Calculate statistics
            scores = [r.get("quality_score", 0) for r in all_results if r.get("quality_score")]
            passed = sum(1 for r in all_results if r.get("passed"))
            
            result["statistics"] = {
                "average_score": round(sum(scores) / len(scores), 2) if scores else 0,
                "min_score": round(min(scores), 2) if scores else 0,
                "max_score": round(max(scores), 2) if scores else 0,
                "pass_count": passed,
                "fail_count": len(all_results) - passed,
                "pass_rate": round(passed / len(all_results) * 100, 1)
            }
            
            # Group by model
            by_model = {}
            for r in all_results:
                model = r.get("model_id", "unknown")
                if model not in by_model:
                    by_model[model] = {"tests": 0, "passed": 0, "total_score": 0}
                by_model[model]["tests"] += 1
                by_model[model]["passed"] += 1 if r.get("passed") else 0
                by_model[model]["total_score"] += r.get("quality_score", 0)
            
            result["by_model"] = {
                model: {
                    "tests": data["tests"],
                    "pass_rate": round(data["passed"] / data["tests"] * 100, 1),
                    "avg_score": round(data["total_score"] / data["tests"], 2)
                }
                for model, data in by_model.items()
            }
            
            # Trend analysis (first vs last 10)
            if len(all_results) >= 20:
                recent = all_results[:10]
                older = all_results[-10:]
                
                recent_avg = sum(r.get("quality_score", 0) for r in recent) / 10
                older_avg = sum(r.get("quality_score", 0) for r in older) / 10
                
                if recent_avg > older_avg + 0.05:
                    result["trend"] = "improving"
                    result["trend_note"] = f"Quality improving: {older_avg:.2f} → {recent_avg:.2f}"
                elif recent_avg < older_avg - 0.05:
                    result["trend"] = "declining"
                    result["trend_note"] = f"Quality declining: {older_avg:.2f} → {recent_avg:.2f}"
                else:
                    result["trend"] = "stable"
                    result["trend_note"] = f"Quality stable around {recent_avg:.2f}"
        else:
            result["note"] = "No calibration tests found. Run run_calibration_test to generate data."
        
        return json.dumps(result, indent=2, default=str)
    
    return {
        "run_calibration_test": run_calibration_test,
        "evaluate_response_quality": evaluate_response_quality,
        "identify_handoff_triggers": identify_handoff_triggers,
        "get_calibration_history": get_calibration_history
    }
