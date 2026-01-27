"""
Msty Admin MCP - Chat/Conversation Management

Advanced tools for managing individual conversations including
export, cloning, branching, and merging.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import hashlib

from .paths import get_msty_paths
from .database import get_database_connection

logger = logging.getLogger("msty-admin-mcp")


# ============================================================================
# CHAT EXPORT
# ============================================================================

def export_chat_thread(
    conversation_id: str,
    export_format: str = "markdown",
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Export a single conversation to various formats.

    Args:
        conversation_id: The conversation to export
        export_format: Output format (markdown, json, text, html)
        include_metadata: Include conversation metadata

    Returns:
        Dict with exported content
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "conversation_id": conversation_id,
        "export_format": export_format,
        "exported": False,
    }

    try:
        conn = get_database_connection()
        if not conn:
            result["error"] = "Could not connect to database"
            return result

        cursor = conn.cursor()

        # Get conversation metadata
        metadata = {}
        try:
            cursor.execute("""
                SELECT * FROM chats WHERE id = ? OR uuid = ? LIMIT 1
            """, (conversation_id, conversation_id))
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                metadata = dict(zip(columns, row))
        except:
            pass

        # Get messages
        messages = []
        message_tables = ["messages", "chat_messages"]
        for table in message_tables:
            try:
                cursor.execute(f"""
                    SELECT * FROM {table}
                    WHERE chat_id = ? OR conversation_id = ? OR session_id = ?
                    ORDER BY id ASC
                """, (conversation_id, conversation_id, conversation_id))
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    messages.append(dict(zip(columns, row)))
                if messages:
                    break
            except:
                pass

        conn.close()

        if not messages:
            result["error"] = f"No messages found for conversation '{conversation_id}'"
            return result

        # Format output
        if export_format == "json":
            result["content"] = {
                "metadata": metadata if include_metadata else {},
                "messages": messages,
                "message_count": len(messages)
            }

        elif export_format == "markdown":
            lines = []
            if include_metadata:
                lines.append(f"# Conversation Export")
                lines.append(f"")
                lines.append(f"- **ID**: {conversation_id}")
                lines.append(f"- **Title**: {metadata.get('title', 'Untitled')}")
                lines.append(f"- **Model**: {metadata.get('model', 'Unknown')}")
                lines.append(f"- **Created**: {metadata.get('created_at', 'Unknown')}")
                lines.append(f"- **Messages**: {len(messages)}")
                lines.append(f"")
                lines.append("---")
                lines.append("")

            for msg in messages:
                role = msg.get("role", msg.get("type", "unknown"))
                content = msg.get("content", msg.get("text", ""))

                if role in ["user", "human"]:
                    lines.append(f"## 👤 User")
                elif role in ["assistant", "ai", "bot"]:
                    lines.append(f"## 🤖 Assistant")
                else:
                    lines.append(f"## {role.title()}")

                lines.append("")
                lines.append(content)
                lines.append("")

            result["content"] = "\n".join(lines)

        elif export_format == "text":
            lines = []
            for msg in messages:
                role = msg.get("role", msg.get("type", "unknown"))
                content = msg.get("content", msg.get("text", ""))
                lines.append(f"[{role.upper()}]")
                lines.append(content)
                lines.append("")

            result["content"] = "\n".join(lines)

        elif export_format == "html":
            html_parts = ['<!DOCTYPE html><html><head><meta charset="utf-8">']
            html_parts.append(f'<title>Conversation {conversation_id}</title>')
            html_parts.append('<style>')
            html_parts.append('body { font-family: system-ui; max-width: 800px; margin: 0 auto; padding: 20px; }')
            html_parts.append('.message { margin: 20px 0; padding: 15px; border-radius: 10px; }')
            html_parts.append('.user { background: #e3f2fd; }')
            html_parts.append('.assistant { background: #f5f5f5; }')
            html_parts.append('.role { font-weight: bold; margin-bottom: 10px; }')
            html_parts.append('</style></head><body>')

            if include_metadata:
                html_parts.append(f'<h1>{metadata.get("title", "Conversation")}</h1>')

            for msg in messages:
                role = msg.get("role", msg.get("type", "unknown"))
                content = msg.get("content", msg.get("text", ""))
                css_class = "user" if role in ["user", "human"] else "assistant"
                # Escape HTML
                content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                content = content.replace('\n', '<br>')

                html_parts.append(f'<div class="message {css_class}">')
                html_parts.append(f'<div class="role">{role.title()}</div>')
                html_parts.append(f'<div class="content">{content}</div>')
                html_parts.append('</div>')

            html_parts.append('</body></html>')
            result["content"] = "\n".join(html_parts)

        else:
            result["error"] = f"Unknown export format: {export_format}"
            return result

        result["exported"] = True
        result["message_count"] = len(messages)

    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================================
# CHAT CLONING
# ============================================================================

def clone_chat(
    conversation_id: str,
    new_title: Optional[str] = None,
    include_all_messages: bool = True
) -> Dict[str, Any]:
    """
    Clone a conversation for A/B testing or experimentation.

    Note: This creates a logical clone in the result. Actual database
    insertion would require write permissions to Msty's database.

    Args:
        conversation_id: The conversation to clone
        new_title: Title for the cloned conversation
        include_all_messages: Include all messages or just the first exchange

    Returns:
        Dict with clone data ready for import
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "source_conversation_id": conversation_id,
        "cloned": False,
    }

    try:
        conn = get_database_connection()
        if not conn:
            result["error"] = "Could not connect to database"
            return result

        cursor = conn.cursor()

        # Get original conversation
        original = {}
        try:
            cursor.execute("""
                SELECT * FROM chats WHERE id = ? OR uuid = ? LIMIT 1
            """, (conversation_id, conversation_id))
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                original = dict(zip(columns, row))
        except:
            pass

        # Get messages
        messages = []
        message_tables = ["messages", "chat_messages"]
        for table in message_tables:
            try:
                cursor.execute(f"""
                    SELECT * FROM {table}
                    WHERE chat_id = ? OR conversation_id = ? OR session_id = ?
                    ORDER BY id ASC
                """, (conversation_id, conversation_id, conversation_id))
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    messages.append(dict(zip(columns, row)))
                if messages:
                    break
            except:
                pass

        conn.close()

        if not messages:
            result["error"] = f"No messages found for conversation '{conversation_id}'"
            return result

        # Create clone data
        clone_id = hashlib.md5(
            f"{conversation_id}-clone-{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        clone_data = {
            "id": f"clone_{clone_id}",
            "title": new_title or f"Clone of {original.get('title', conversation_id)}",
            "model": original.get("model"),
            "created_at": datetime.now().isoformat(),
            "cloned_from": conversation_id,
            "messages": []
        }

        # Clone messages
        if include_all_messages:
            clone_data["messages"] = messages
        else:
            # Just first exchange (first user + first assistant)
            for msg in messages:
                clone_data["messages"].append(msg)
                if msg.get("role") in ["assistant", "ai"]:
                    break

        result["cloned"] = True
        result["clone_data"] = clone_data
        result["message_count"] = len(clone_data["messages"])
        result["note"] = "Clone data generated. To persist, import into Msty or save to file."

    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================================
# CHAT BRANCHING
# ============================================================================

def branch_chat(
    conversation_id: str,
    branch_point: int,
    branch_title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a branch from a specific point in a conversation.

    Args:
        conversation_id: The conversation to branch from
        branch_point: Message index to branch from (0-based)
        branch_title: Title for the branch

    Returns:
        Dict with branch data
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "source_conversation_id": conversation_id,
        "branch_point": branch_point,
        "branched": False,
    }

    try:
        conn = get_database_connection()
        if not conn:
            result["error"] = "Could not connect to database"
            return result

        cursor = conn.cursor()

        # Get original conversation
        original = {}
        try:
            cursor.execute("""
                SELECT * FROM chats WHERE id = ? OR uuid = ? LIMIT 1
            """, (conversation_id, conversation_id))
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                original = dict(zip(columns, row))
        except:
            pass

        # Get messages
        messages = []
        message_tables = ["messages", "chat_messages"]
        for table in message_tables:
            try:
                cursor.execute(f"""
                    SELECT * FROM {table}
                    WHERE chat_id = ? OR conversation_id = ? OR session_id = ?
                    ORDER BY id ASC
                """, (conversation_id, conversation_id, conversation_id))
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    messages.append(dict(zip(columns, row)))
                if messages:
                    break
            except:
                pass

        conn.close()

        if not messages:
            result["error"] = f"No messages found for conversation '{conversation_id}'"
            return result

        if branch_point >= len(messages):
            result["error"] = f"Branch point {branch_point} exceeds message count {len(messages)}"
            return result

        # Create branch data
        branch_id = hashlib.md5(
            f"{conversation_id}-branch-{branch_point}-{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        branch_data = {
            "id": f"branch_{branch_id}",
            "title": branch_title or f"Branch from {original.get('title', conversation_id)} @ msg {branch_point}",
            "model": original.get("model"),
            "created_at": datetime.now().isoformat(),
            "branched_from": conversation_id,
            "branch_point": branch_point,
            "messages": messages[:branch_point + 1]  # Include up to and including branch point
        }

        result["branched"] = True
        result["branch_data"] = branch_data
        result["messages_included"] = len(branch_data["messages"])
        result["messages_excluded"] = len(messages) - len(branch_data["messages"])
        result["note"] = "Branch data generated. Continue the conversation from this point."

    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================================
# CHAT MERGING
# ============================================================================

def merge_chats(
    conversation_ids: List[str],
    merge_strategy: str = "sequential",
    merged_title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Merge multiple conversations into one.

    Args:
        conversation_ids: List of conversation IDs to merge
        merge_strategy: How to merge (sequential, interleaved, by_timestamp)
        merged_title: Title for merged conversation

    Returns:
        Dict with merged conversation data
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "source_conversations": conversation_ids,
        "merge_strategy": merge_strategy,
        "merged": False,
    }

    if len(conversation_ids) < 2:
        result["error"] = "Need at least 2 conversations to merge"
        return result

    try:
        conn = get_database_connection()
        if not conn:
            result["error"] = "Could not connect to database"
            return result

        cursor = conn.cursor()

        all_messages = {}
        conversation_titles = []

        for conv_id in conversation_ids:
            # Get conversation title
            try:
                cursor.execute("""
                    SELECT title FROM chats WHERE id = ? OR uuid = ? LIMIT 1
                """, (conv_id, conv_id))
                row = cursor.fetchone()
                if row:
                    conversation_titles.append(row[0] or conv_id)
                else:
                    conversation_titles.append(conv_id)
            except:
                conversation_titles.append(conv_id)

            # Get messages
            messages = []
            message_tables = ["messages", "chat_messages"]
            for table in message_tables:
                try:
                    cursor.execute(f"""
                        SELECT * FROM {table}
                        WHERE chat_id = ? OR conversation_id = ? OR session_id = ?
                        ORDER BY id ASC
                    """, (conv_id, conv_id, conv_id))
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    for row in rows:
                        msg = dict(zip(columns, row))
                        msg["_source_conversation"] = conv_id
                        messages.append(msg)
                    if messages:
                        break
                except:
                    pass

            all_messages[conv_id] = messages

        conn.close()

        # Merge based on strategy
        merged_messages = []

        if merge_strategy == "sequential":
            # Simply concatenate
            for conv_id in conversation_ids:
                merged_messages.extend(all_messages.get(conv_id, []))

        elif merge_strategy == "interleaved":
            # Interleave by taking one exchange from each
            max_len = max(len(msgs) for msgs in all_messages.values())
            for i in range(max_len):
                for conv_id in conversation_ids:
                    msgs = all_messages.get(conv_id, [])
                    if i < len(msgs):
                        merged_messages.append(msgs[i])

        elif merge_strategy == "by_timestamp":
            # Sort by timestamp
            all_msgs = []
            for conv_id, msgs in all_messages.items():
                all_msgs.extend(msgs)
            # Sort by created_at or id
            merged_messages = sorted(
                all_msgs,
                key=lambda x: x.get("created_at", x.get("id", 0))
            )

        else:
            result["error"] = f"Unknown merge strategy: {merge_strategy}"
            return result

        # Create merged data
        merge_id = hashlib.md5(
            f"merge-{'-'.join(conversation_ids)}-{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        merged_data = {
            "id": f"merged_{merge_id}",
            "title": merged_title or f"Merged: {', '.join(conversation_titles[:3])}{'...' if len(conversation_titles) > 3 else ''}",
            "created_at": datetime.now().isoformat(),
            "merged_from": conversation_ids,
            "merge_strategy": merge_strategy,
            "messages": merged_messages
        }

        result["merged"] = True
        result["merged_data"] = merged_data
        result["total_messages"] = len(merged_messages)
        result["source_message_counts"] = {
            conv_id: len(msgs)
            for conv_id, msgs in all_messages.items()
        }

    except Exception as e:
        result["error"] = str(e)

    return result


__all__ = [
    "export_chat_thread",
    "clone_chat",
    "branch_chat",
    "merge_chats",
]
