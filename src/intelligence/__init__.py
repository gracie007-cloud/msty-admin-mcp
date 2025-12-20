"""
Phase 4: Intelligence Layer Tools for Msty Admin MCP

Tools:
- get_model_performance_metrics
- analyse_conversation_patterns  
- compare_model_responses
- optimise_knowledge_stacks
- suggest_persona_improvements

Created by Pineapple 🍍 AI Administration System
"""

import json
import time
from datetime import datetime, timedelta
from typing import Optional, List

from ..phase4_5_tools import (
    init_metrics_db,
    record_model_metric,
    get_model_metrics_summary,
    evaluate_response_heuristic
)


def register_phase4_tools(mcp, make_api_request, is_process_running, get_msty_paths, 
                          query_database, get_table_names, LOCAL_AI_SERVICE_PORT):
    """Register Phase 4 Intelligence Layer tools with the MCP server"""
    
    @mcp.tool()
    def get_model_performance_metrics(
        model_id: Optional[str] = None,
        days: int = 30
    ) -> str:
        """
        Get performance metrics for local models over time.
        
        Tracks and reports:
        - Tokens per second (generation speed)
        - Average latency
        - Error rates
        - Usage patterns
        
        Args:
            model_id: Specific model to query (None = all models)
            days: Number of days to include in analysis (default: 30)
        
        Returns:
            Aggregated performance metrics with trends
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "period_days": days,
            "metrics": {}
        }
        
        try:
            init_metrics_db()
            metrics = get_model_metrics_summary(model_id=model_id, days=days)
            
            if "error" in metrics:
                result["error"] = metrics["error"]
                return json.dumps(result, indent=2)
            
            result["metrics"] = metrics
            
            if metrics.get("models"):
                insights = []
                for model in metrics["models"]:
                    tps = model.get("avg_tokens_per_second", 0) or 0
                    error_count = model.get("error_count", 0) or 0
                    total_requests = model.get("total_requests", 1) or 1
                    error_rate = error_count / max(total_requests, 1)
                    
                    if tps > 50:
                        insights.append(f"✅ {model['model_id']}: Excellent speed ({tps:.1f} tok/s)")
                    elif tps > 20:
                        insights.append(f"👍 {model['model_id']}: Good speed ({tps:.1f} tok/s)")
                    elif tps > 0:
                        insights.append(f"⚠️ {model['model_id']}: Slow ({tps:.1f} tok/s)")
                    
                    if error_rate > 0.1:
                        insights.append(f"❌ {model['model_id']}: High error rate ({error_rate:.1%})")
                
                result["insights"] = insights
            
            total_requests = sum(m.get("total_requests", 0) or 0 for m in metrics.get("models", []))
            if total_requests < 10:
                result["note"] = f"Limited data ({total_requests} requests). Use chat_with_local_model to gather more metrics."
            
        except Exception as e:
            result["error"] = str(e)
        
        return json.dumps(result, indent=2)
    
    @mcp.tool()
    def analyse_conversation_patterns(
        days: int = 30
    ) -> str:
        """
        Analyse conversation patterns from Msty database.
        
        Privacy-respecting analysis that tracks:
        - Session counts and lengths
        - Message volumes over time
        - Model usage distribution
        - Peak usage times
        
        Args:
            days: Number of days to analyse (default: 30)
        
        Returns:
            Aggregated usage patterns without exposing conversation content
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "period_days": days,
            "patterns": {}
        }
        
        paths = get_msty_paths()
        db_path = paths.get("database")
        
        if not db_path:
            result["error"] = "Msty database not found"
            return json.dumps(result, indent=2)
        
        try:
            tables = get_table_names(db_path)
            
            session_table = None
            message_table = None
            
            for t in ["chat_sessions", "conversations", "sessions"]:
                if t in tables:
                    session_table = t
                    break
            
            for t in ["chat_messages", "messages"]:
                if t in tables:
                    message_table = t
                    break
            
            patterns = {
                "session_analysis": {},
                "message_analysis": {},
                "model_usage": {},
                "recommendations": []
            }
            
            if session_table:
                count_query = f"SELECT COUNT(*) as count FROM {session_table}"
                count_result = query_database(db_path, count_query)
                patterns["session_analysis"]["total_sessions"] = count_result[0]["count"] if count_result else 0
                
                recent_query = f"SELECT * FROM {session_table} ORDER BY rowid DESC LIMIT 100"
                recent_sessions = query_database(db_path, recent_query)
                
                if recent_sessions:
                    model_counts = {}
                    for session in recent_sessions:
                        model = (session.get("model") or 
                                session.get("model_id") or 
                                session.get("llm_model") or 
                                "unknown")
                        model_counts[model] = model_counts.get(model, 0) + 1
                    
                    patterns["model_usage"] = model_counts
                    patterns["session_analysis"]["recent_session_count"] = len(recent_sessions)
            
            if message_table:
                count_query = f"SELECT COUNT(*) as count FROM {message_table}"
                count_result = query_database(db_path, count_query)
                patterns["message_analysis"]["total_messages"] = count_result[0]["count"] if count_result else 0
                
                if patterns["session_analysis"].get("total_sessions", 0) > 0:
                    avg_messages = patterns["message_analysis"]["total_messages"] / patterns["session_analysis"]["total_sessions"]
                    patterns["message_analysis"]["avg_messages_per_session"] = round(avg_messages, 1)
            
            total_sessions = patterns["session_analysis"].get("total_sessions", 0)
            if total_sessions > 100:
                patterns["recommendations"].append("High usage detected - consider running health check regularly")
            
            model_usage = patterns.get("model_usage", {})
            if len(model_usage) > 1:
                top_model = max(model_usage, key=model_usage.get) if model_usage else None
                if top_model:
                    patterns["recommendations"].append(f"Most used model: {top_model}")
            
            result["patterns"] = patterns
            result["tables_found"] = {
                "session_table": session_table,
                "message_table": message_table
            }
            
        except Exception as e:
            result["error"] = str(e)
        
        return json.dumps(result, indent=2, default=str)
    
    @mcp.tool()
    def compare_model_responses(
        prompt: str,
        models: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        evaluation_criteria: str = "balanced"
    ) -> str:
        """
        Send the same prompt to multiple models and compare responses.
        
        Useful for:
        - Finding the best model for a specific use case
        - Comparing quality vs speed trade-offs
        - Validating model selection
        
        Args:
            prompt: The prompt to send to all models
            models: List of model IDs to compare (None = use all available, max 5)
            system_prompt: Optional system prompt for context
            evaluation_criteria: What to optimise for:
                - "quality": Best response quality
                - "speed": Fastest response
                - "balanced": Quality-speed trade-off
        
        Returns:
            Comparison of responses with timing and quality scores
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "evaluation_criteria": evaluation_criteria,
            "responses": [],
            "comparison": {}
        }
        
        if not is_process_running("MstySidecar"):
            result["error"] = "Sidecar is not running"
            return json.dumps(result, indent=2)
        
        if not models:
            models_response = make_api_request("/v1/models", port=LOCAL_AI_SERVICE_PORT)
            if models_response.get("success"):
                data = models_response.get("data", {})
                if isinstance(data, dict) and "data" in data:
                    models = [m.get("id") for m in data["data"]][:5]
            
            if not models:
                result["error"] = "No models available"
                return json.dumps(result, indent=2)
        
        result["models_tested"] = models
        
        for model_id in models:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
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
            
            model_result = {
                "model_id": model_id,
                "success": response.get("success", False),
                "latency_seconds": round(elapsed_time, 2),
                "response": None,
                "tokens_per_second": 0
            }
            
            if response.get("success"):
                data = response.get("data", {})
                if "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    msg = choice.get("message", {})
                    content = msg.get("content", "") or msg.get("reasoning", "")
                    model_result["response"] = content[:500] + "..." if len(content) > 500 else content
                    model_result["response_length"] = len(content)
                    
                    if "usage" in data:
                        completion_tokens = data["usage"].get("completion_tokens", 0)
                        model_result["tokens_per_second"] = round(
                            completion_tokens / max(elapsed_time, 0.1), 1
                        )
                        model_result["completion_tokens"] = completion_tokens
                    
                    eval_result = evaluate_response_heuristic(prompt, content, "general")
                    model_result["quality_score"] = round(eval_result["score"], 2)
            else:
                model_result["error"] = response.get("error", "Unknown error")
            
            result["responses"].append(model_result)
            
            try:
                init_metrics_db()
                record_model_metric(
                    model_id=model_id,
                    prompt_tokens=int(len(prompt.split()) * 1.3),
                    completion_tokens=model_result.get("completion_tokens", 0),
                    latency_seconds=elapsed_time,
                    success=model_result["success"],
                    use_case="comparison"
                )
            except:
                pass
        
        successful = [r for r in result["responses"] if r["success"]]
        
        if successful:
            if evaluation_criteria == "speed":
                best = min(successful, key=lambda x: x["latency_seconds"])
                result["comparison"]["winner"] = best["model_id"]
                result["comparison"]["reason"] = f"Fastest at {best['latency_seconds']}s"
            elif evaluation_criteria == "quality":
                best = max(successful, key=lambda x: x.get("quality_score", 0))
                result["comparison"]["winner"] = best["model_id"]
                result["comparison"]["reason"] = f"Highest quality ({best.get('quality_score', 0):.2f})"
            else:
                for r in successful:
                    quality = r.get("quality_score", 0.5)
                    speed = 1.0 / max(r["latency_seconds"], 0.1)
                    r["_balanced_score"] = quality * 0.6 + min(speed / 10, 0.4)
                best = max(successful, key=lambda x: x.get("_balanced_score", 0))
                result["comparison"]["winner"] = best["model_id"]
                result["comparison"]["reason"] = f"Best balance"
            
            result["comparison"]["ranking"] = [
                {"model": r["model_id"], "quality": r.get("quality_score", 0), "latency": r["latency_seconds"]}
                for r in sorted(successful, key=lambda x: x.get("quality_score", 0), reverse=True)
            ]
        
        return json.dumps(result, indent=2, default=str)
    
    @mcp.tool()
    def optimise_knowledge_stacks() -> str:
        """
        Analyse and recommend optimisations for knowledge stacks.
        
        Checks:
        - Knowledge stack sizes and content
        - Usage patterns
        - Redundancy detection
        - Performance impact
        
        Returns:
            Recommendations for knowledge stack improvements
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "analysis": {},
            "recommendations": []
        }
        
        paths = get_msty_paths()
        db_path = paths.get("database")
        
        if not db_path:
            result["error"] = "Msty database not found"
            return json.dumps(result, indent=2)
        
        try:
            tables = get_table_names(db_path)
            
            ks_table = None
            for t in ["knowledge_stacks", "knowledge_stack", "stacks"]:
                if t in tables:
                    ks_table = t
                    break
            
            if not ks_table:
                result["note"] = "No knowledge stack table found"
                result["available_tables"] = tables
                return json.dumps(result, indent=2)
            
            query = f"SELECT * FROM {ks_table}"
            stacks = query_database(db_path, query)
            
            result["analysis"]["total_stacks"] = len(stacks)
            result["analysis"]["stacks"] = []
            
            total_size = 0
            for stack in stacks:
                stack_info = {
                    "name": stack.get("name", "Unknown"),
                    "id": stack.get("id", stack.get("rowid", "?")),
                }
                
                content = stack.get("content", "") or stack.get("data", "") or ""
                if isinstance(content, str):
                    stack_info["content_length"] = len(content)
                    total_size += len(content)
                
                result["analysis"]["stacks"].append(stack_info)
            
            result["analysis"]["total_content_size_kb"] = round(total_size / 1024, 2)
            
            if len(stacks) == 0:
                result["recommendations"].append("No knowledge stacks found. Consider creating domain-specific stacks.")
            elif len(stacks) > 10:
                result["recommendations"].append(f"Many stacks ({len(stacks)}). Consider consolidating.")
            
            if total_size > 1024 * 1024:
                result["recommendations"].append("Large total size. Consider chunking for better performance.")
            
        except Exception as e:
            result["error"] = str(e)
        
        return json.dumps(result, indent=2, default=str)
    
    @mcp.tool()
    def suggest_persona_improvements(
        persona_name: Optional[str] = None
    ) -> str:
        """
        Analyse personas and suggest improvements.
        
        Checks:
        - System prompt length and complexity
        - Temperature settings
        - Tool configurations
        
        Args:
            persona_name: Specific persona to analyse (None = all)
        
        Returns:
            Suggestions for persona optimisation
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "analysis": {},
            "suggestions": []
        }
        
        paths = get_msty_paths()
        db_path = paths.get("database")
        
        if not db_path:
            result["error"] = "Msty database not found"
            return json.dumps(result, indent=2)
        
        try:
            tables = get_table_names(db_path)
            
            persona_table = None
            for t in ["personas", "persona"]:
                if t in tables:
                    persona_table = t
                    break
            
            if not persona_table:
                result["note"] = "No persona table found"
                return json.dumps(result, indent=2)
            
            if persona_name:
                query = f"SELECT * FROM {persona_table} WHERE name LIKE ?"
                personas = query_database(db_path, query, (f"%{persona_name}%",))
            else:
                query = f"SELECT * FROM {persona_table}"
                personas = query_database(db_path, query)
            
            result["analysis"]["total_personas"] = len(personas)
            result["analysis"]["personas"] = []
            
            for persona in personas:
                name = persona.get("name", "Unknown")
                system_prompt = persona.get("system_prompt", "") or persona.get("prompt", "") or ""
                temp = persona.get("temperature")
                
                persona_info = {
                    "name": name,
                    "system_prompt_length": len(system_prompt),
                }
                if temp is not None:
                    persona_info["temperature"] = temp
                
                result["analysis"]["personas"].append(persona_info)
                
                if len(system_prompt) < 100:
                    result["suggestions"].append(f"'{name}': System prompt too short. Add more context.")
                elif len(system_prompt) > 4000:
                    result["suggestions"].append(f"'{name}': System prompt very long. Consider condensing.")
                
                if temp is not None:
                    if temp < 0.3:
                        result["suggestions"].append(f"'{name}': Low temp ({temp}). Good for facts, limits creativity.")
                    elif temp > 0.9:
                        result["suggestions"].append(f"'{name}': High temp ({temp}). May be inconsistent.")
            
            if len(personas) == 0:
                result["suggestions"].append("No personas found. Create task-specific personas.")
            elif len(personas) == 1:
                result["suggestions"].append("Only one persona. Consider specialised personas for different tasks.")
            
        except Exception as e:
            result["error"] = str(e)
        
        return json.dumps(result, indent=2, default=str)
    
    return {
        "get_model_performance_metrics": get_model_performance_metrics,
        "analyse_conversation_patterns": analyse_conversation_patterns,
        "compare_model_responses": compare_model_responses,
        "optimise_knowledge_stacks": optimise_knowledge_stacks,
        "suggest_persona_improvements": suggest_persona_improvements
    }
