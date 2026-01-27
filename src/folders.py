"""
Msty Admin MCP - Folder Organization

Tools for managing chat folders and organization structure.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from .paths import get_msty_paths
from .database import get_database_connection

logger = logging.getLogger("msty-admin-mcp")


# ============================================================================
# FOLDER LISTING
# ============================================================================

def list_folders() -> Dict[str, Any]:
    """
    List all chat folders.

    Returns:
        Dict with folders list and hierarchy
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "folders": [],
        "total_count": 0,
    }

    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()

            # Look for folder tables
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND (
                    name LIKE '%folder%' OR
                    name LIKE '%project%' OR
                    name LIKE '%category%'
                )
            """)
            tables = [t[0] for t in cursor.fetchall()]

            for table in tables:
                try:
                    cursor.execute(f"SELECT * FROM {table} LIMIT 200")
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    for row in rows:
                        folder = dict(zip(columns, row))
                        folder["_source_table"] = table
                        result["folders"].append(folder)
                except Exception as e:
                    logger.debug(f"Error reading {table}: {e}")

            conn.close()
    except Exception as e:
        logger.debug(f"Folder query error: {e}")

    # Build hierarchy
    folders_by_id = {f.get("id"): f for f in result["folders"]}
    root_folders = []
    child_folders = []

    for folder in result["folders"]:
        parent_id = folder.get("parent_id") or folder.get("parent")
        if parent_id and parent_id in folders_by_id:
            child_folders.append(folder)
        else:
            root_folders.append(folder)

    result["hierarchy"] = {
        "root_folders": len(root_folders),
        "nested_folders": len(child_folders),
        "max_depth": _calculate_folder_depth(result["folders"])
    }

    result["total_count"] = len(result["folders"])

    if not result["folders"]:
        result["note"] = "No folders found. Folders can be created in Msty's sidebar."
        result["folders"] = [
            {
                "id": "default",
                "name": "All Chats",
                "is_default": True,
                "note": "Default uncategorized folder"
            }
        ]

    return result


def _calculate_folder_depth(folders: List[Dict]) -> int:
    """Calculate maximum folder nesting depth."""
    folders_by_id = {f.get("id"): f for f in folders}
    max_depth = 0

    for folder in folders:
        depth = 0
        current = folder
        visited = set()

        while current and current.get("id") not in visited:
            visited.add(current.get("id"))
            parent_id = current.get("parent_id") or current.get("parent")
            if parent_id and parent_id in folders_by_id:
                depth += 1
                current = folders_by_id[parent_id]
            else:
                break

        max_depth = max(max_depth, depth)

    return max_depth


def get_folder_details(folder_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific folder.

    Args:
        folder_id: The folder identifier

    Returns:
        Dict with folder details including contents
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "folder_id": folder_id,
        "found": False,
    }

    all_folders = list_folders()

    for folder in all_folders.get("folders", []):
        if folder.get("id") == folder_id or folder.get("name") == folder_id:
            result.update(folder)
            result["found"] = True

            # Get chats in folder
            result["chats"] = []
            try:
                conn = get_database_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT id, title, model, created_at, updated_at
                        FROM chats
                        WHERE folder_id = ? OR folder = ? OR project_id = ?
                        ORDER BY updated_at DESC
                        LIMIT 100
                    """, (folder_id, folder_id, folder_id))
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    for row in rows:
                        result["chats"].append(dict(zip(columns, row)))
                    conn.close()
            except:
                pass

            result["chat_count"] = len(result["chats"])
            return result

    result["error"] = f"Folder '{folder_id}' not found"
    result["available_folders"] = [
        f.get("name", f.get("id"))
        for f in all_folders.get("folders", [])
    ]

    return result


def get_folder_stats() -> Dict[str, Any]:
    """
    Get statistics for all folders.

    Returns:
        Dict with folder statistics
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "statistics": {},
    }

    all_folders = list_folders()

    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()

            stats = {
                "total_folders": len(all_folders.get("folders", [])),
                "folders_with_chats": 0,
                "empty_folders": 0,
                "folder_breakdown": []
            }

            for folder in all_folders.get("folders", []):
                folder_id = folder.get("id")
                if not folder_id or folder.get("is_default"):
                    continue

                # Count chats in folder
                try:
                    cursor.execute("""
                        SELECT COUNT(*) FROM chats
                        WHERE folder_id = ? OR folder = ? OR project_id = ?
                    """, (folder_id, folder_id, folder_id))
                    row = cursor.fetchone()
                    chat_count = row[0] if row else 0

                    stats["folder_breakdown"].append({
                        "folder_id": folder_id,
                        "folder_name": folder.get("name", folder_id),
                        "chat_count": chat_count
                    })

                    if chat_count > 0:
                        stats["folders_with_chats"] += 1
                    else:
                        stats["empty_folders"] += 1
                except:
                    pass

            # Count uncategorized chats
            try:
                cursor.execute("""
                    SELECT COUNT(*) FROM chats
                    WHERE folder_id IS NULL AND folder IS NULL AND project_id IS NULL
                """)
                row = cursor.fetchone()
                stats["uncategorized_chats"] = row[0] if row else 0
            except:
                stats["uncategorized_chats"] = 0

            conn.close()

            # Sort breakdown by chat count
            stats["folder_breakdown"].sort(
                key=lambda x: x["chat_count"],
                reverse=True
            )

            result["statistics"] = stats

    except Exception as e:
        result["error"] = str(e)

    return result


def suggest_folder_organization() -> Dict[str, Any]:
    """
    Analyze chats and suggest folder organization.

    Returns:
        Dict with organization suggestions
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "suggestions": [],
        "analysis": {},
    }

    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()

            # Get uncategorized chats
            cursor.execute("""
                SELECT id, title, model, created_at
                FROM chats
                WHERE folder_id IS NULL AND folder IS NULL AND project_id IS NULL
                ORDER BY created_at DESC
                LIMIT 200
            """)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            uncategorized = [dict(zip(columns, row)) for row in rows]

            conn.close()

            result["analysis"] = {
                "uncategorized_count": len(uncategorized),
                "by_model": {},
                "by_keyword": {}
            }

            # Analyze by model
            for chat in uncategorized:
                model = chat.get("model", "unknown")
                if model not in result["analysis"]["by_model"]:
                    result["analysis"]["by_model"][model] = []
                result["analysis"]["by_model"][model].append(chat.get("id"))

            # Simple keyword detection for suggestions
            keywords = {
                "code": ["code", "programming", "debug", "function", "api", "script"],
                "writing": ["write", "essay", "article", "blog", "content", "story"],
                "analysis": ["analyze", "data", "report", "statistics", "research"],
                "chat": ["chat", "conversation", "help", "question"],
            }

            for chat in uncategorized:
                title = (chat.get("title") or "").lower()
                for category, words in keywords.items():
                    if any(word in title for word in words):
                        if category not in result["analysis"]["by_keyword"]:
                            result["analysis"]["by_keyword"][category] = []
                        result["analysis"]["by_keyword"][category].append(chat.get("id"))
                        break

            # Generate suggestions
            if len(uncategorized) > 10:
                result["suggestions"].append({
                    "type": "organization",
                    "priority": "high",
                    "message": f"You have {len(uncategorized)} uncategorized chats",
                    "recommendation": "Create folders to organize your conversations"
                })

            # Suggest model-based folders
            for model, chats in result["analysis"]["by_model"].items():
                if len(chats) >= 5:
                    result["suggestions"].append({
                        "type": "model_folder",
                        "priority": "medium",
                        "message": f"{len(chats)} chats use {model}",
                        "recommendation": f"Consider creating a '{model}' folder"
                    })

            # Suggest topic-based folders
            for topic, chats in result["analysis"]["by_keyword"].items():
                if len(chats) >= 3:
                    result["suggestions"].append({
                        "type": "topic_folder",
                        "priority": "medium",
                        "message": f"{len(chats)} chats appear to be about {topic}",
                        "recommendation": f"Consider creating a '{topic.title()}' folder"
                    })

            if not result["suggestions"]:
                result["suggestions"].append({
                    "type": "good_organization",
                    "priority": "low",
                    "message": "Your conversations appear well-organized!",
                    "recommendation": "Keep up the good work"
                })

    except Exception as e:
        result["error"] = str(e)

    return result


__all__ = [
    "list_folders",
    "get_folder_details",
    "get_folder_stats",
    "suggest_folder_organization",
]
