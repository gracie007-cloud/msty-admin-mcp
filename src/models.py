"""
Msty Admin MCP - Data Models

Data classes for structured data throughout the application.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MstyInstallation:
    """Msty Studio Desktop installation details"""
    installed: bool
    version: Optional[str] = None
    app_path: Optional[str] = None
    data_path: Optional[str] = None
    sidecar_path: Optional[str] = None
    database_path: Optional[str] = None
    mlx_models_path: Optional[str] = None
    is_running: bool = False
    sidecar_running: bool = False
    platform_info: dict = field(default_factory=dict)


@dataclass
class MstyHealthReport:
    """Msty Studio health analysis"""
    overall_status: str
    database_status: dict = field(default_factory=dict)
    storage_status: dict = field(default_factory=dict)
    model_cache_status: dict = field(default_factory=dict)
    recommendations: list = field(default_factory=list)
    timestamp: str = ""


@dataclass
class DatabaseStats:
    """Statistics from Msty database"""
    total_conversations: int = 0
    total_messages: int = 0
    total_personas: int = 0
    total_prompts: int = 0
    total_knowledge_stacks: int = 0
    total_tools: int = 0
    database_size_mb: float = 0.0
    last_activity: Optional[str] = None


@dataclass
class PersonaConfig:
    """Msty persona configuration structure"""
    name: str
    description: str = ""
    system_prompt: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    model_preference: Optional[str] = None
    knowledge_stacks: list = field(default_factory=list)
    tools_enabled: list = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


__all__ = [
    "MstyInstallation",
    "MstyHealthReport",
    "DatabaseStats",
    "PersonaConfig"
]
