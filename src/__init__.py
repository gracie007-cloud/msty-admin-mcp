"""
Msty Admin MCP Server v9.0.0

AI-administered Msty Studio Desktop management system with database insights,
configuration management, local model orchestration, tiered AI workflows,
and advanced AI orchestration capabilities.

Phase 1: Foundational Tools (Read-Only)
Phase 2: Configuration Management
Phase 3: Automation Bridge (Sidecar Integration)
Phase 4: Intelligence Layer (Analytics)
Phase 5: Tiered AI Workflow (Calibration)
Phase 6: Advanced Model Management
Phase 7: Conversation Management
Phase 8: Prompt Templates & Automation
Phase 9: Backup & System Management
Phase 10: Knowledge Stack Management
Phase 11: Model Download/Delete
Phase 12: Claude ↔ Local Model Bridge
Phase 13: Turnstile Workflows
Phase 14: Live Context
Phase 15: Conversation Analytics
Phase 16: Shadow Persona Integration
Phase 17: Workspaces Management
Phase 18: Real-Time Web/Data Integration
Phase 19: Chat/Conversation Management
Phase 20: Folder Organization
Phase 21: PII Scrubbing Tools
Phase 22: Embedding Visualization
Phase 23: Health Monitoring Dashboard
Phase 24: Configuration Profiles
Phase 25: Automated Maintenance
Phase 26: Intelligent Auto-Router
Phase 27: Autonomous Agent Swarm
Phase 28: Continuous Background Agents
Phase 29: Semantic Response Cache
Phase 30: Predictive Model Pre-Loading
Phase 31: Conversation Archaeology
Phase 32: A/B Testing Framework
Phase 33: Cascade Execution
Phase 34: Cost Intelligence Dashboard
Phase 35: Persona Fusion

Total: 155 tools

Original author: Pineapple 🍍
Fork maintainer: DigitalKredit (v5.0.0+)
"""

from .constants import SERVER_VERSION

__version__ = SERVER_VERSION
__author__ = "DigitalKredit"
__original_author__ = "Pineapple 🍍"

# Import utility modules (no external dependencies)
from .models import MstyInstallation, MstyHealthReport, DatabaseStats, PersonaConfig
from .errors import ErrorCode, error_response, success_response
from .paths import get_msty_paths, sanitize_path, expand_path, read_claude_desktop_config
from .database import (
    get_database_connection, query_database, get_table_names,
    is_safe_table_name, validate_table_exists, safe_query_table,
    safe_count_table, get_table_row_count
)
from .network import (
    make_api_request, is_process_running, is_local_ai_available,
    get_available_service_ports
)
from .cache import ResponseCache, get_cached_models, cache_models, get_cache
from .tagging import MODEL_TAGS, get_model_tags, find_models_by_tag

# Import Phase 10-15 extension modules
from .knowledge_stacks import (
    list_knowledge_stacks, get_knowledge_stack_details,
    search_knowledge_stack, analyze_knowledge_stack, get_stack_statistics
)
from .model_management import (
    get_local_model_inventory, delete_model, find_duplicate_models,
    check_huggingface_model, get_download_instructions, analyze_model_storage
)
from .model_bridge import (
    select_best_model_for_task, delegate_to_local_model,
    multi_model_consensus, draft_and_refine, parallel_process_tasks
)
from .turnstiles import (
    list_turnstiles, get_turnstile_details, list_turnstile_templates,
    get_turnstile_template, execute_turnstile, analyze_turnstile_usage,
    suggest_turnstile_for_task, TURNSTILE_TEMPLATES
)
from .live_context import (
    get_system_context, get_datetime_context, get_environment_context,
    get_process_context, get_msty_context, get_full_live_context,
    format_context_for_prompt, get_cached_context, clear_context_cache,
    get_context_cache_stats
)
from .conversation_analytics import (
    get_conversations, get_messages, analyze_usage_patterns,
    analyze_conversation_content, analyze_model_performance,
    analyze_session_patterns, generate_analytics_report
)

# Import Phase 16-25 extension modules
from .shadow_personas import (
    list_shadow_personas, get_shadow_persona_details,
    analyze_shadow_conversation, synthesize_shadow_responses,
    compare_shadow_responses
)
from .workspaces import (
    list_workspaces, get_workspace_details,
    get_workspace_stats, export_workspace
)
from .realtime_data import (
    realtime_search, realtime_fetch, realtime_youtube_transcript
)
from .chat_management import (
    export_chat_thread, clone_chat, branch_chat, merge_chats
)
from .folders import (
    list_folders, get_folder_details, get_folder_stats,
    suggest_folder_organization
)
from .pii_tools import (
    scan_for_pii, scrub_pii, generate_pii_report, PII_PATTERNS
)
from .embeddings import (
    get_embeddings_from_stack, visualize_embeddings,
    cluster_embeddings, compare_embeddings, embedding_statistics
)
from .dashboard import (
    check_service_health, get_dashboard_status, get_active_alerts
)
from .profiles import (
    list_profiles, save_profile, load_profile, compare_profiles,
    export_profile
)
from .maintenance import (
    identify_cleanup_candidates, perform_cleanup,
    optimize_database, generate_maintenance_report
)

# Import Phase 26-35 extension modules (Advanced AI Orchestration)
from .smart_router import (
    classify_task, estimate_complexity, route_request,
    record_routing_outcome, get_routing_stats, get_model_recommendation,
    clear_routing_history, TASK_PATTERNS, MODEL_CAPABILITIES
)
from .agent_swarm import (
    create_agent, execute_agent_task, spawn_swarm,
    get_agent_status, list_agents, terminate_agent,
    clear_all_agents, get_swarm_results, AgentRole, AgentStatus,
    ROLE_SYSTEM_PROMPTS
)
from .background_agents import (
    create_background_agent, start_background_agent, stop_background_agent,
    get_alerts, acknowledge_alert, list_background_agents,
    delete_background_agent, trigger_agent_run,
    BackgroundAgentType, AlertSeverity, DEFAULT_CONFIGS
)
from .semantic_cache import (
    cache_response, find_similar_response, get_cache_stats,
    clear_cache, delete_cache_entry, list_cache_entries,
    configure_cache, cosine_similarity
)
from .predictive_loader import (
    record_usage, predict_next_task, recommend_models_to_load,
    get_usage_summary, get_hourly_breakdown, configure_prediction,
    clear_usage_history, start_session
)
from .conversation_archaeology import (
    search_conversations, find_decisions, build_timeline,
    extract_action_items, get_conversation_summary,
    find_related_conversations, get_archaeology_stats
)
from .ab_testing import (
    create_experiment, run_experiment, analyze_experiment,
    rate_result, get_experiment, list_experiments,
    delete_experiment, compare_models_quick, ExperimentStatus
)
from .cascade import (
    execute_with_cascade, smart_execute, estimate_response_confidence,
    get_cascade_config, test_cascade_tiers, DEFAULT_CASCADE_TIERS
)
from .cost_intelligence import (
    record_usage as cost_record_usage, get_usage_summary as cost_usage_summary,
    compare_local_vs_cloud, get_daily_breakdown, set_session_budget,
    get_budget_alerts, get_optimization_recommendations,
    get_cost_projection, export_usage_data, CLOUD_COSTS_PER_1K
)
from .persona_fusion import (
    fuse_personas, suggest_fusion_for_task, get_fused_persona,
    list_fused_personas, list_available_personas,
    analyze_persona_compatibility, delete_fused_persona,
    quick_fuse_for_task, PERSONA_TEMPLATES
)

# Lazy import for server module (requires mcp package)
# Use: from src.server import mcp, main
def _get_server():
    """Lazy import for server module (requires mcp package)"""
    from . import server
    return server

__all__ = [
    # Core (lazy loaded)
    "_get_server",
    "__version__",
    "__author__",
    # Models
    "MstyInstallation",
    "MstyHealthReport",
    "DatabaseStats",
    "PersonaConfig",
    # Errors
    "ErrorCode",
    "error_response",
    "success_response",
    # Paths
    "get_msty_paths",
    "sanitize_path",
    "expand_path",
    "read_claude_desktop_config",
    # Database
    "get_database_connection",
    "query_database",
    "get_table_names",
    "is_safe_table_name",
    "validate_table_exists",
    "safe_query_table",
    "safe_count_table",
    "get_table_row_count",
    # Network
    "make_api_request",
    "is_process_running",
    "is_local_ai_available",
    "get_available_service_ports",
    # Cache
    "ResponseCache",
    "get_cached_models",
    "cache_models",
    "get_cache",
    # Tagging
    "MODEL_TAGS",
    "get_model_tags",
    "find_models_by_tag",
    # Knowledge Stacks (Phase 10)
    "list_knowledge_stacks",
    "get_knowledge_stack_details",
    "search_knowledge_stack",
    "analyze_knowledge_stack",
    "get_stack_statistics",
    # Model Management (Phase 11)
    "get_local_model_inventory",
    "delete_model",
    "find_duplicate_models",
    "check_huggingface_model",
    "get_download_instructions",
    "analyze_model_storage",
    # Model Bridge (Phase 12)
    "select_best_model_for_task",
    "delegate_to_local_model",
    "multi_model_consensus",
    "draft_and_refine",
    "parallel_process_tasks",
    # Turnstiles (Phase 13)
    "list_turnstiles",
    "get_turnstile_details",
    "list_turnstile_templates",
    "get_turnstile_template",
    "execute_turnstile",
    "analyze_turnstile_usage",
    "suggest_turnstile_for_task",
    "TURNSTILE_TEMPLATES",
    # Live Context (Phase 14)
    "get_system_context",
    "get_datetime_context",
    "get_environment_context",
    "get_process_context",
    "get_msty_context",
    "get_full_live_context",
    "format_context_for_prompt",
    "get_cached_context",
    "clear_context_cache",
    "get_context_cache_stats",
    # Conversation Analytics (Phase 15)
    "get_conversations",
    "get_messages",
    "analyze_usage_patterns",
    "analyze_conversation_content",
    "analyze_model_performance",
    "analyze_session_patterns",
    "generate_analytics_report",
    # Shadow Personas (Phase 16)
    "list_shadow_personas",
    "get_shadow_persona_details",
    "analyze_shadow_conversation",
    "synthesize_shadow_responses",
    "compare_shadow_responses",
    # Workspaces (Phase 17)
    "list_workspaces",
    "get_workspace_details",
    "get_workspace_stats",
    "export_workspace",
    # Real-Time Data (Phase 18)
    "realtime_search",
    "realtime_fetch",
    "realtime_youtube_transcript",
    # Chat Management (Phase 19)
    "export_chat_thread",
    "clone_chat",
    "branch_chat",
    "merge_chats",
    # Folders (Phase 20)
    "list_folders",
    "get_folder_details",
    "get_folder_stats",
    "suggest_folder_organization",
    # PII Tools (Phase 21)
    "scan_for_pii",
    "scrub_pii",
    "generate_pii_report",
    "PII_PATTERNS",
    # Embeddings (Phase 22)
    "get_embeddings_from_stack",
    "visualize_embeddings",
    "cluster_embeddings",
    "compare_embeddings",
    "embedding_statistics",
    # Dashboard (Phase 23)
    "check_service_health",
    "get_dashboard_status",
    "get_active_alerts",
    # Profiles (Phase 24)
    "list_profiles",
    "save_profile",
    "load_profile",
    "compare_profiles",
    "export_profile",
    # Maintenance (Phase 25)
    "identify_cleanup_candidates",
    "perform_cleanup",
    "optimize_database",
    "generate_maintenance_report",
    # Smart Router (Phase 26)
    "classify_task",
    "estimate_complexity",
    "route_request",
    "record_routing_outcome",
    "get_routing_stats",
    "get_model_recommendation",
    "clear_routing_history",
    "TASK_PATTERNS",
    "MODEL_CAPABILITIES",
    # Agent Swarm (Phase 27)
    "create_agent",
    "execute_agent_task",
    "spawn_swarm",
    "get_agent_status",
    "list_agents",
    "terminate_agent",
    "clear_all_agents",
    "get_swarm_results",
    "AgentRole",
    "AgentStatus",
    "ROLE_SYSTEM_PROMPTS",
    # Background Agents (Phase 28)
    "create_background_agent",
    "start_background_agent",
    "stop_background_agent",
    "get_alerts",
    "acknowledge_alert",
    "list_background_agents",
    "delete_background_agent",
    "trigger_agent_run",
    "BackgroundAgentType",
    "AlertSeverity",
    "DEFAULT_CONFIGS",
    # Semantic Cache (Phase 29)
    "cache_response",
    "find_similar_response",
    "delete_cache_entry",
    "list_cache_entries",
    "configure_cache",
    "cosine_similarity",
    # Predictive Loader (Phase 30)
    "record_usage",
    "predict_next_task",
    "recommend_models_to_load",
    "configure_prediction",
    "start_session",
    # Conversation Archaeology (Phase 31)
    "search_conversations",
    "find_decisions",
    "build_timeline",
    "extract_action_items",
    "get_conversation_summary",
    "find_related_conversations",
    "get_archaeology_stats",
    # A/B Testing (Phase 32)
    "create_experiment",
    "run_experiment",
    "analyze_experiment",
    "rate_result",
    "get_experiment",
    "list_experiments",
    "delete_experiment",
    "compare_models_quick",
    "ExperimentStatus",
    # Cascade Execution (Phase 33)
    "execute_with_cascade",
    "smart_execute",
    "estimate_response_confidence",
    "get_cascade_config",
    "test_cascade_tiers",
    "DEFAULT_CASCADE_TIERS",
    # Cost Intelligence (Phase 34)
    "cost_record_usage",
    "cost_usage_summary",
    "compare_local_vs_cloud",
    "get_daily_breakdown",
    "set_session_budget",
    "get_budget_alerts",
    "get_optimization_recommendations",
    "get_cost_projection",
    "export_usage_data",
    "CLOUD_COSTS_PER_1K",
    # Persona Fusion (Phase 35)
    "fuse_personas",
    "suggest_fusion_for_task",
    "get_fused_persona",
    "list_fused_personas",
    "list_available_personas",
    "analyze_persona_compatibility",
    "delete_fused_persona",
    "quick_fuse_for_task",
    "PERSONA_TEMPLATES",
]