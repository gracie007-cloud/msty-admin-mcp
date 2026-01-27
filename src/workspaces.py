"""
Msty Admin MCP - Workspaces Management

Msty supports multiple isolated workspaces for data separation.
This module provides tools for managing workspaces.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from .paths import get_msty_paths
from .database import get_database_connection

logger = logging.getLogger("msty-admin-mcp")


# ============================================================================
# WORKSPACE DATA ACCESS
# ============================================================================

def get_workspaces_path() -> Optional[Path]:
    """Get the path to Workspaces storage."""
    paths = get_msty_paths()
    data_dir = paths.get("data_dir")
    if data_dir:
        # Workspaces might be in different locations
        possible_paths = [
            Path(data_dir) / "Workspaces",
            Path(data_dir) / "Spaces",
            Path(data_dir).parent / "Workspaces",
        ]
        for p in possible_paths:
            if p.exists():
                return p
    return None


def list_workspaces() -> Dict[str, Any]:
    """
    List all available workspaces.

    Returns:
        Dict with workspaces list and metadata
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "workspaces": [],
        "total_count": 0,
        "active_workspace": None,
    }

    # Try file system
    workspaces_path = get_workspaces_path()
    if workspaces_path and workspaces_path.exists():
        for item in workspaces_path.iterdir():
            if item.is_dir():
                workspace_info = {
                    "id": item.name,
                    "name": item.name,
                    "path": str(item),
                    "source": "filesystem",
                }

                # Get workspace metadata if exists
                meta_file = item / "workspace.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, 'r') as f:
                            meta = json.load(f)
                            workspace_info.update(meta)
                    except:
                        pass

                # Calculate size
                try:
                    total_size = sum(
                        f.stat().st_size for f in item.rglob('*') if f.is_file()
                    )
                    workspace_info["size_mb"] = round(total_size / (1024 * 1024), 2)
                except:
                    workspace_info["size_mb"] = 0

                # Count items
                try:
                    workspace_info["file_count"] = len(list(item.rglob('*')))
                except:
                    workspace_info["file_count"] = 0

                result["workspaces"].append(workspace_info)

    # Try database for workspace info
    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND (
                    name LIKE '%workspace%' OR
                    name LIKE '%space%'
                )
            """)
            tables = [t[0] for t in cursor.fetchall()]

            for table in tables:
                try:
                    cursor.execute(f"SELECT * FROM {table} LIMIT 50")
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    for row in rows:
                        ws = dict(zip(columns, row))
                        ws["source"] = "database"
                        # Avoid duplicates
                        if not any(w.get("id") == ws.get("id") for w in result["workspaces"]):
                            result["workspaces"].append(ws)
                except:
                    pass

            # Check for active workspace
            try:
                cursor.execute("""
                    SELECT * FROM settings WHERE key = 'active_workspace'
                    OR key = 'current_workspace'
                    LIMIT 1
                """)
                row = cursor.fetchone()
                if row:
                    result["active_workspace"] = row[1] if len(row) > 1 else row[0]
            except:
                pass

            conn.close()
    except Exception as e:
        logger.debug(f"Workspace DB query: {e}")

    result["total_count"] = len(result["workspaces"])

    if not result["workspaces"]:
        result["note"] = "No workspaces found. You may be using the default workspace."
        result["workspaces"].append({
            "id": "default",
            "name": "Default Workspace",
            "is_default": True,
            "note": "This is the default workspace when no explicit workspaces are configured"
        })

    return result


def get_workspace_details(workspace_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific workspace.

    Args:
        workspace_id: The workspace identifier

    Returns:
        Dict with workspace details
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "workspace_id": workspace_id,
        "found": False,
    }

    all_workspaces = list_workspaces()

    for ws in all_workspaces.get("workspaces", []):
        if ws.get("id") == workspace_id or ws.get("name") == workspace_id:
            result.update(ws)
            result["found"] = True

            # Get additional details if path exists
            if "path" in ws:
                ws_path = Path(ws["path"])
                if ws_path.exists():
                    # List contents
                    result["contents"] = {
                        "directories": [],
                        "files": []
                    }
                    for item in ws_path.iterdir():
                        if item.is_dir():
                            result["contents"]["directories"].append(item.name)
                        else:
                            result["contents"]["files"].append(item.name)

            return result

    result["error"] = f"Workspace '{workspace_id}' not found"
    result["available_workspaces"] = [
        ws.get("name", ws.get("id"))
        for ws in all_workspaces.get("workspaces", [])
    ]

    return result


def get_workspace_stats(workspace_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get statistics for a workspace or all workspaces.

    Args:
        workspace_id: Specific workspace (optional, all if None)

    Returns:
        Dict with workspace statistics
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "workspace_id": workspace_id,
        "statistics": {},
    }

    all_workspaces = list_workspaces()

    if workspace_id:
        # Stats for specific workspace
        for ws in all_workspaces.get("workspaces", []):
            if ws.get("id") == workspace_id or ws.get("name") == workspace_id:
                result["statistics"] = {
                    "workspace": ws.get("name", workspace_id),
                    "size_mb": ws.get("size_mb", 0),
                    "file_count": ws.get("file_count", 0),
                }

                # Try to get conversation count
                try:
                    conn = get_database_connection()
                    if conn:
                        cursor = conn.cursor()
                        # This is workspace-specific if Msty supports it
                        cursor.execute("""
                            SELECT COUNT(*) FROM chats
                            WHERE workspace_id = ? OR workspace = ?
                        """, (workspace_id, workspace_id))
                        row = cursor.fetchone()
                        if row:
                            result["statistics"]["conversation_count"] = row[0]
                        conn.close()
                except:
                    pass

                return result

        result["error"] = f"Workspace '{workspace_id}' not found"
    else:
        # Stats for all workspaces
        result["statistics"] = {
            "total_workspaces": len(all_workspaces.get("workspaces", [])),
            "total_size_mb": sum(
                ws.get("size_mb", 0)
                for ws in all_workspaces.get("workspaces", [])
            ),
            "total_files": sum(
                ws.get("file_count", 0)
                for ws in all_workspaces.get("workspaces", [])
            ),
            "workspace_breakdown": [
                {
                    "name": ws.get("name", ws.get("id")),
                    "size_mb": ws.get("size_mb", 0),
                    "file_count": ws.get("file_count", 0)
                }
                for ws in all_workspaces.get("workspaces", [])
            ]
        }

    return result


def export_workspace(
    workspace_id: str,
    export_format: str = "json",
    include_conversations: bool = True,
    include_personas: bool = True,
    include_knowledge_stacks: bool = True
) -> Dict[str, Any]:
    """
    Export a workspace for backup or migration.

    Args:
        workspace_id: The workspace to export
        export_format: Output format (json, zip)
        include_conversations: Include conversation history
        include_personas: Include persona configurations
        include_knowledge_stacks: Include knowledge stack data

    Returns:
        Dict with export data or file path
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "workspace_id": workspace_id,
        "export_format": export_format,
        "exported": False,
    }

    workspace = get_workspace_details(workspace_id)

    if not workspace.get("found"):
        result["error"] = f"Workspace '{workspace_id}' not found"
        return result

    export_data = {
        "workspace_info": {
            "id": workspace_id,
            "name": workspace.get("name", workspace_id),
            "exported_at": datetime.now().isoformat(),
        },
        "components": {
            "conversations": [],
            "personas": [],
            "knowledge_stacks": [],
        }
    }

    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()

            # Export conversations
            if include_conversations:
                try:
                    cursor.execute("""
                        SELECT * FROM chats
                        WHERE workspace_id = ? OR workspace = ?
                        LIMIT 1000
                    """, (workspace_id, workspace_id))
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    for row in rows:
                        export_data["components"]["conversations"].append(
                            dict(zip(columns, row))
                        )
                except Exception as e:
                    logger.debug(f"Conversation export: {e}")

            # Export personas
            if include_personas:
                try:
                    cursor.execute("""
                        SELECT * FROM personas
                        WHERE workspace_id = ? OR workspace = ?
                        LIMIT 100
                    """, (workspace_id, workspace_id))
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    for row in rows:
                        export_data["components"]["personas"].append(
                            dict(zip(columns, row))
                        )
                except Exception as e:
                    logger.debug(f"Persona export: {e}")

            # Export knowledge stacks
            if include_knowledge_stacks:
                try:
                    cursor.execute("""
                        SELECT * FROM knowledge_stacks
                        WHERE workspace_id = ? OR workspace = ?
                        LIMIT 100
                    """, (workspace_id, workspace_id))
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    for row in rows:
                        export_data["components"]["knowledge_stacks"].append(
                            dict(zip(columns, row))
                        )
                except Exception as e:
                    logger.debug(f"Knowledge stack export: {e}")

            conn.close()

        result["exported"] = True
        result["export_data"] = export_data
        result["summary"] = {
            "conversations_exported": len(export_data["components"]["conversations"]),
            "personas_exported": len(export_data["components"]["personas"]),
            "knowledge_stacks_exported": len(export_data["components"]["knowledge_stacks"]),
        }

    except Exception as e:
        result["error"] = str(e)

    return result


__all__ = [
    "get_workspaces_path",
    "list_workspaces",
    "get_workspace_details",
    "get_workspace_stats",
    "export_workspace",
]
