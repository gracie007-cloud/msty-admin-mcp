"""
Msty Admin MCP Server

AI-administered Msty Studio Desktop management system with database insights,
configuration management, local model orchestration, and tiered AI workflows.

Phase 1: Foundational Tools (Read-Only)
Phase 2: Configuration Management
Phase 3: Automation Bridge (Sidecar Integration)
Phase 4: Intelligence Layer (Analytics)
Phase 5: Tiered AI Workflow (Calibration)
Phase 6: Advanced Model Management
Phase 7: Conversation Management
Phase 8: Prompt Templates & Automation
Phase 9: Backup & System Management

Created by Pineapple 🍍
"""

from .constants import SERVER_VERSION

__version__ = SERVER_VERSION
__author__ = "Pineapple 🍍"

from .server import mcp, main
from .models import MstyInstallation, MstyHealthReport, DatabaseStats, PersonaConfig
from .errors import ErrorCode, error_response, success_response

# Utility modules (v6.5.0)
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

__all__ = [
    # Core
    "mcp",
    "main",
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
    "find_models_by_tag"
]