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

__all__ = [
    "mcp",
    "main",
    "__version__",
    "__author__",
    "MstyInstallation",
    "MstyHealthReport",
    "DatabaseStats",
    "PersonaConfig",
    "ErrorCode",
    "error_response",
    "success_response"
]