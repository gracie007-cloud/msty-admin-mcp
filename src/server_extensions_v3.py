"""
Msty Admin MCP Server Extensions v9.0.0

New tools for Phase 26-35 (Advanced AI Orchestration):
- Intelligent Auto-Router
- Autonomous Agent Swarm
- Continuous Background Agents
- Semantic Response Cache
- Predictive Model Pre-Loading
- Conversation Archaeology
- A/B Testing Framework
- Cascade Execution
- Cost Intelligence Dashboard
- Persona Fusion

These extensions add 42 new tools to the server (Phases 26-35).
"""

import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Try to import ResearchOrchestrator from msty-openclaw-bridge
try:
    BRIDGE_PATH = Path("G:/ai-agentic-dev/msty-openclaw-bridge")
    if str(BRIDGE_PATH) not in sys.path:
        sys.path.append(str(BRIDGE_PATH))
    from apps.research_agent.orchestrator import ResearchOrchestrator
    from apps.music_producer.producer import MusicProducer
    from apps.voice_assistant.orchestrator import VoiceOrchestrator
    RESEARCH_AVAILABLE = True
    MUSIC_AVAILABLE = True
    VOICE_AVAILABLE = True
except ImportError:
    RESEARCH_AVAILABLE = False
    MUSIC_AVAILABLE = False
    VOICE_AVAILABLE = False

# Import new modules (Phase 26-35)
from .smart_router import (
    classify_task,
    estimate_complexity,
    route_request,
    record_routing_outcome,
    get_routing_stats,
    get_model_recommendation,
    clear_routing_history,
)
from .agent_swarm import (
    create_agent,
    execute_agent_task,
    spawn_swarm,
    get_agent_status,
    list_agents,
    terminate_agent,
    clear_all_agents,
    get_swarm_results,
    AgentRole,
)
from .background_agents import (
    create_background_agent,
    start_background_agent,
    stop_background_agent,
    get_alerts,
    acknowledge_alert,
    list_background_agents,
    delete_background_agent,
    trigger_agent_run,
    BackgroundAgentType,
)
from .semantic_cache import (
    cache_response as semantic_cache_response,
    find_similar_response,
    get_cache_stats as semantic_get_cache_stats,
    clear_cache as semantic_clear_cache,
    delete_cache_entry,
    list_cache_entries,
    configure_cache,
)
from .predictive_loader import (
    predict_next_task,
    recommend_models_to_load,
    get_usage_summary as predictor_usage_summary,
    get_hourly_breakdown as predictor_hourly_breakdown,
    configure_prediction,
    clear_usage_history as predictor_clear_history,
    start_session as predictor_start_session,
)
from .conversation_archaeology import (
    search_conversations as archaeology_search,
    find_decisions,
    build_timeline,
    extract_action_items,
    find_related_conversations,
    get_archaeology_stats,
)
from .ab_testing import (
    create_experiment,
    run_experiment,
    analyze_experiment,
    rate_result,
    get_experiment,
    list_experiments,
    delete_experiment,
    compare_models_quick,
)
from .cascade import (
    execute_with_cascade,
    smart_execute,
    get_cascade_config,
    test_cascade_tiers,
)
from .cost_intelligence import (
    record_usage as cost_record_usage,
    get_usage_summary as cost_usage_summary,
    compare_local_vs_cloud,
    get_daily_breakdown,
    set_session_budget,
    get_budget_alerts,
    get_optimization_recommendations,
    get_cost_projection,
    export_usage_data,
)
from .persona_fusion import (
    fuse_personas,
    suggest_fusion_for_task,
    get_fused_persona,
    list_fused_personas,
    list_available_personas,
    analyze_persona_compatibility,
    delete_fused_persona,
    quick_fuse_for_task,
)

logger = logging.getLogger("msty-admin-mcp")


def register_extension_tools_v3(mcp: FastMCP):
    """Register all v3 extension tools with the MCP server (Phases 26-35)."""

    # =========================================================================
    # Phase 26: Intelligent Auto-Router
    # =========================================================================

    @mcp.tool()
    def router_classify(task_description: str) -> str:
        """
        Classify a task to determine optimal model routing.

        Args:
            task_description: Description of the task to classify

        Returns:
            JSON with task classification scores for different categories
        """
        result = classify_task(task_description)
        return json.dumps({
            "timestamp": datetime.now().isoformat(),
            "task": task_description[:100],
            "classification": result
        }, indent=2, default=str)

    @mcp.tool()
    def router_route(
        task_description: str,
        prefer_speed: bool = False,
        prefer_quality: bool = False
    ) -> str:
        """
        Route a request to the optimal model based on task analysis.

        Args:
            task_description: Description of the task
            prefer_speed: Prioritize faster models
            prefer_quality: Prioritize higher quality models

        Returns:
            JSON with routing decision and recommended model
        """
        result = route_request(task_description, prefer_speed, prefer_quality)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def router_stats() -> str:
        """
        Get routing statistics and history.

        Returns:
            JSON with routing statistics
        """
        result = get_routing_stats()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def router_recommend(task_type: str = "general") -> str:
        """
        Get model recommendation for a task type.

        Args:
            task_type: Type of task (coding, research, creative, etc.)

        Returns:
            JSON with recommended model and reasoning
        """
        result = get_model_recommendation(task_type)
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 27: Autonomous Agent Swarm
    # =========================================================================

    @mcp.tool()
    def swarm_spawn(
        task: str,
        roles: Optional[str] = None,
        parallel: bool = True,
        synthesize: bool = True
    ) -> str:
        """
        Spawn a swarm of AI agents to tackle a complex task.

        Args:
            task: The main task to accomplish
            roles: Comma-separated list of roles (code,research,writing,analysis)
            parallel: Run agents in parallel
            synthesize: Synthesize results with orchestrator

        Returns:
            JSON with swarm execution results
        """
        role_list = roles.split(",") if roles else None
        result = spawn_swarm(task, role_list, parallel, synthesize)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def swarm_create_agent(
        role: str = "general",
        model_id: Optional[str] = None
    ) -> str:
        """
        Create a specialized AI agent.

        Args:
            role: Agent role (code, research, writing, analysis, review, general)
            model_id: Specific model to use (auto-selects if None)

        Returns:
            JSON with created agent details
        """
        try:
            agent_role = AgentRole(role.lower())
        except ValueError:
            agent_role = AgentRole.GENERAL
        result = create_agent(agent_role, model_id)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def swarm_execute(
        agent_id: str,
        task: str,
        context: Optional[str] = None
    ) -> str:
        """
        Execute a task with a specific agent.

        Args:
            agent_id: The agent to use
            task: The task to execute
            context: Optional context for the task

        Returns:
            JSON with task execution result
        """
        result = execute_agent_task(agent_id, task, context)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def swarm_list() -> str:
        """
        List all agents in the swarm pool.

        Returns:
            JSON with all agents and their status
        """
        result = list_agents()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def swarm_results(swarm_id: str) -> str:
        """
        Get results from a previous swarm execution.

        Args:
            swarm_id: The swarm ID to get results for

        Returns:
            JSON with swarm results
        """
        result = get_swarm_results(swarm_id)
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 28: Continuous Background Agents
    # =========================================================================

    @mcp.tool()
    def bg_agent_create(
        agent_type: str,
        name: Optional[str] = None,
        interval_minutes: int = 60,
        auto_start: bool = False
    ) -> str:
        """
        Create a background monitoring agent.

        Args:
            agent_type: Type (code_sentinel, doc_keeper, research_watcher, meeting_prep)
            name: Custom name for the agent
            interval_minutes: How often to run
            auto_start: Start monitoring immediately

        Returns:
            JSON with created agent details
        """
        try:
            bg_type = BackgroundAgentType(agent_type.lower())
        except ValueError:
            bg_type = BackgroundAgentType.CUSTOM
        result = create_background_agent(bg_type, name, interval_minutes, None, auto_start)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def bg_agent_start(agent_id: str) -> str:
        """
        Start a background agent's monitoring loop.

        Args:
            agent_id: The agent to start

        Returns:
            JSON with start result
        """
        result = start_background_agent(agent_id)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def bg_agent_stop(agent_id: str) -> str:
        """
        Stop a background agent's monitoring loop.

        Args:
            agent_id: The agent to stop

        Returns:
            JSON with stop result
        """
        result = stop_background_agent(agent_id)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def bg_agent_list() -> str:
        """
        List all background agents.

        Returns:
            JSON with all background agents
        """
        result = list_background_agents()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def bg_agent_trigger(agent_id: str) -> str:
        """
        Manually trigger a background agent to run now.

        Args:
            agent_id: The agent to trigger

        Returns:
            JSON with trigger result
        """
        result = trigger_agent_run(agent_id)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def bg_alerts(
        severity: Optional[str] = None,
        unacknowledged_only: bool = False
    ) -> str:
        """
        Get alerts from background agents.

        Args:
            severity: Filter by severity (info, warning, error, critical)
            unacknowledged_only: Only show unacknowledged alerts

        Returns:
            JSON with alerts
        """
        result = get_alerts(None, severity, unacknowledged_only)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def bg_alert_ack(alert_id: str) -> str:
        """
        Acknowledge an alert.

        Args:
            alert_id: The alert to acknowledge

        Returns:
            JSON with acknowledgment result
        """
        result = acknowledge_alert(alert_id)
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 29: Semantic Response Cache
    # =========================================================================

    @mcp.tool()
    def sem_cache_store(
        prompt: str,
        response: str,
        model_id: str,
        ttl_hours: int = 24
    ) -> str:
        """
        Cache a response with semantic embedding.

        Args:
            prompt: The original prompt
            response: The model's response
            model_id: The model that generated the response
            ttl_hours: Time-to-live in hours

        Returns:
            JSON with cache entry details
        """
        result = semantic_cache_response(prompt, response, model_id, ttl_hours)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def sem_cache_find(
        prompt: str,
        threshold: float = 0.85
    ) -> str:
        """
        Find a semantically similar cached response.

        Args:
            prompt: The prompt to find similar responses for
            threshold: Similarity threshold (0-1)

        Returns:
            JSON with cache hit or miss details
        """
        result = find_similar_response(prompt, threshold)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def sem_cache_stats() -> str:
        """
        Get semantic cache statistics.

        Returns:
            JSON with cache statistics
        """
        result = semantic_get_cache_stats()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def sem_cache_list(limit: int = 50) -> str:
        """
        List cached entries.

        Args:
            limit: Maximum entries to return

        Returns:
            JSON with cache entries
        """
        result = list_cache_entries(limit)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def sem_cache_config(
        similarity_threshold: Optional[float] = None,
        max_cache_size: Optional[int] = None,
        default_ttl_hours: Optional[int] = None
    ) -> str:
        """
        Configure semantic cache settings.

        Args:
            similarity_threshold: Minimum similarity for cache hit (0-1)
            max_cache_size: Maximum cache entries
            default_ttl_hours: Default TTL for new entries

        Returns:
            JSON with updated configuration
        """
        result = configure_cache(similarity_threshold, max_cache_size, default_ttl_hours)
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 30: Predictive Model Pre-Loading
    # =========================================================================

    @mcp.tool()
    def predict_task() -> str:
        """
        Predict what type of task the user is likely to do next.

        Returns:
            JSON with task predictions based on usage patterns
        """
        result = predict_next_task()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def predict_models() -> str:
        """
        Recommend which models should be pre-loaded.

        Returns:
            JSON with model recommendations
        """
        result = recommend_models_to_load()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def predict_session_start() -> str:
        """
        Start a new usage session with predictions.

        Returns:
            JSON with session info and predictions
        """
        result = predictor_start_session()
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 31: Conversation Archaeology
    # =========================================================================

    @mcp.tool()
    def archaeology_search_tool(
        query: str,
        days: int = 90,
        limit: int = 50
    ) -> str:
        """
        Deep search across all historical conversations.

        Args:
            query: Search query
            days: Number of days to search back
            limit: Maximum results

        Returns:
            JSON with search results and context
        """
        result = archaeology_search(query, limit, days)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def archaeology_decisions(topic: Optional[str] = None, days: int = 90) -> str:
        """
        Find decisions made in conversations.

        Args:
            topic: Optional topic to filter decisions
            days: Number of days to search

        Returns:
            JSON with extracted decisions
        """
        result = find_decisions(topic, days)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def archaeology_timeline(topic: str, days: int = 90) -> str:
        """
        Build a timeline of discussions about a topic.

        Args:
            topic: Topic to track
            days: Number of days to search

        Returns:
            JSON with timeline of mentions
        """
        result = build_timeline(topic, days)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def archaeology_actions(
        conversation_id: Optional[str] = None,
        days: int = 7
    ) -> str:
        """
        Extract action items from conversations.

        Args:
            conversation_id: Specific conversation (optional, all if None)
            days: Number of days to search

        Returns:
            JSON with extracted action items
        """
        result = extract_action_items(conversation_id, days)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def archaeology_related(conversation_id: str) -> str:
        """
        Find conversations related to a specific one.

        Args:
            conversation_id: The conversation to find relations for

        Returns:
            JSON with related conversations
        """
        result = find_related_conversations(conversation_id)
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 32: A/B Testing Framework
    # =========================================================================

    @mcp.tool()
    def ab_create(
        name: str,
        prompts: str,
        models: str,
        description: str = ""
    ) -> str:
        """
        Create an A/B testing experiment.

        Args:
            name: Experiment name
            prompts: Pipe-separated prompts to test (prompt1|prompt2)
            models: Comma-separated models to compare (model1,model2)
            description: Experiment description

        Returns:
            JSON with created experiment details
        """
        prompt_list = prompts.split("|")
        model_list = models.split(",")
        result = create_experiment(name, prompt_list, model_list, description)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def ab_run(experiment_id: str) -> str:
        """
        Run an A/B testing experiment.

        Args:
            experiment_id: The experiment to run

        Returns:
            JSON with experiment results
        """
        result = run_experiment(experiment_id)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def ab_analyze(experiment_id: str) -> str:
        """
        Analyze results from an A/B experiment.

        Args:
            experiment_id: The experiment to analyze

        Returns:
            JSON with statistical analysis
        """
        result = analyze_experiment(experiment_id)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def ab_list() -> str:
        """
        List all A/B experiments.

        Returns:
            JSON with all experiments
        """
        result = list_experiments()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def ab_quick_compare(prompt: str, models: str) -> str:
        """
        Quick comparison of models on a single prompt.

        Args:
            prompt: The prompt to test
            models: Comma-separated models to compare

        Returns:
            JSON with comparison results
        """
        model_list = models.split(",")
        result = compare_models_quick(prompt, model_list)
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 33: Cascade Execution
    # =========================================================================

    @mcp.tool()
    def cascade_execute(
        prompt: str,
        start_tier: int = 1,
        max_tier: int = 3
    ) -> str:
        """
        Execute with cascade escalation based on confidence.

        Args:
            prompt: The prompt to execute
            start_tier: Starting tier (1-4)
            max_tier: Maximum tier to escalate to

        Returns:
            JSON with cascade execution results
        """
        result = execute_with_cascade(prompt, start_tier, max_tier)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def cascade_smart(prompt: str) -> str:
        """
        Smart execution that auto-selects starting tier.

        Args:
            prompt: The prompt to execute

        Returns:
            JSON with execution results and task analysis
        """
        result = smart_execute(prompt)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def cascade_config() -> str:
        """
        Get current cascade configuration.

        Returns:
            JSON with tier configuration
        """
        result = get_cascade_config()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def cascade_test_tiers() -> str:
        """
        Test which models are available at each tier.

        Returns:
            JSON with tier availability
        """
        result = test_cascade_tiers()
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 34: Cost Intelligence Dashboard
    # =========================================================================

    @mcp.tool()
    def cost_summary(days: int = 30) -> str:
        """
        Get cost and usage summary.

        Args:
            days: Number of days to summarize

        Returns:
            JSON with usage and cost summary
        """
        result = cost_usage_summary(days)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def cost_compare_local_cloud() -> str:
        """
        Compare local vs cloud costs.

        Returns:
            JSON with cost comparison and savings
        """
        result = compare_local_vs_cloud()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def cost_daily(days: int = 7) -> str:
        """
        Get day-by-day cost breakdown.

        Args:
            days: Number of days

        Returns:
            JSON with daily breakdown
        """
        result = get_daily_breakdown(days)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def cost_budget_set(session_id: str, budget: float) -> str:
        """
        Set a budget limit for a session.

        Args:
            session_id: Session identifier
            budget: Budget in dollars

        Returns:
            JSON with budget status
        """
        result = set_session_budget(session_id, budget)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def cost_alerts() -> str:
        """
        Get budget alerts.

        Returns:
            JSON with budget alerts
        """
        result = get_budget_alerts()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def cost_optimize() -> str:
        """
        Get cost optimization recommendations.

        Returns:
            JSON with optimization suggestions
        """
        result = get_optimization_recommendations()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def cost_projection(days_ahead: int = 30) -> str:
        """
        Project future costs based on usage.

        Args:
            days_ahead: Days to project

        Returns:
            JSON with cost projections
        """
        result = get_cost_projection(days_ahead)
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 35: Persona Fusion
    # =========================================================================

    @mcp.tool()
    def fusion_create(
        personas: str,
        weights: Optional[str] = None,
        task_context: Optional[str] = None,
        name: Optional[str] = None
    ) -> str:
        """
        Create a fused persona from multiple source personas.

        Args:
            personas: Comma-separated persona names (coder,researcher,writer)
            weights: Comma-separated weights matching personas (0.5,0.3,0.2)
            task_context: Optional context to influence fusion
            name: Custom name for the fused persona

        Returns:
            JSON with fused persona details
        """
        persona_list = [p.strip() for p in personas.split(",")]
        weight_dict = None
        if weights:
            weight_list = [float(w.strip()) for w in weights.split(",")]
            if len(weight_list) == len(persona_list):
                weight_dict = dict(zip(persona_list, weight_list))
        result = fuse_personas(persona_list, weight_dict, task_context, name)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def fusion_suggest(task_description: str) -> str:
        """
        Suggest which personas to fuse for a task.

        Args:
            task_description: Description of the task

        Returns:
            JSON with persona suggestions
        """
        result = suggest_fusion_for_task(task_description)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def fusion_quick(task_description: str) -> str:
        """
        Automatically create optimal persona fusion for a task.

        Args:
            task_description: Description of the task

        Returns:
            JSON with created fusion details
        """
        result = quick_fuse_for_task(task_description)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def fusion_list() -> str:
        """
        List all fused personas.

        Returns:
            JSON with fused personas
        """
        result = list_fused_personas()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def fusion_available() -> str:
        """
        List available persona templates.

        Returns:
            JSON with available personas
        """
        result = list_available_personas()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def fusion_compatibility(personas: str) -> str:
        """
        Analyze how well personas would work together.

        Args:
            personas: Comma-separated persona names to analyze

        Returns:
            JSON with compatibility analysis
        """
        persona_list = [p.strip() for p in personas.split(",")]
        result = analyze_persona_compatibility(persona_list)
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 36: Research Agent (Bridge to OpenClaw)
    # =========================================================================

    @mcp.tool()
    def research_mission(query: str) -> str:
        """
        Execute a systematic AI-powered research mission.
        Integrates ScraperAI for discovery and Msty for synthesis.

        Args:
            query: The research question or topic
            
        Returns:
            JSON with research report and methodology trace
        """
        if not RESEARCH_AVAILABLE:
            return json.dumps({
                "error": "Research Agent components not found in msty-openclaw-bridge.",
                "status": "failed"
            }, indent=2)
            
        try:
            orchestrator = ResearchOrchestrator()
            result = orchestrator.conduct_research(query)
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            return json.dumps({
                "error": f"Research execution failed: {str(e)}",
                "status": "failed"
            }, indent=2)

    @mcp.tool()
    def music_production(vibe: str) -> str:
        """
        Produce a high-quality song based on a vibe description.
        Generates lyrics locally via Msty and music via Suno.

        Args:
            vibe: Description of the song's style, mood, and topic
            
        Returns:
            JSON with song details and file paths
        """
        if not MUSIC_AVAILABLE:
            return json.dumps({
                "error": "Music Producer components not found in msty-openclaw-bridge.",
                "status": "failed"
            }, indent=2)
            
        try:
            producer = MusicProducer()
            result = producer.produce_song(vibe)
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            return json.dumps({
                "error": f"Music production failed: {str(e)}",
                "status": "failed"
            }, indent=2)

    @mcp.tool()
    def start_voice_assistant() -> str:
        """
        Start the local voice-controlled AI assistant.
        Enables hands-free control of browser, music, research, and system.
        
        Returns:
            Status message
        """
        if not VOICE_AVAILABLE:
            return "Voice Assistant components not found in msty-openclaw-bridge."
            
        try:
            orchestrator = VoiceOrchestrator()
            # Run in a background thread to avoid blocking the MCP server
            threading.Thread(target=orchestrator.run, daemon=True).start()
            return "Local voice assistant activated. You can now speak commands."
        except Exception as e:
            return f"Failed to start voice assistant: {str(e)}"

    logger.info("Registered 45 extension tools (Phases 26-36)")


__all__ = ["register_extension_tools_v3"]
