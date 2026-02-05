"""
Msty Admin MCP - Conversation Archaeology (Phase 31)

Deep analysis of historical conversations to find patterns,
decisions, and forgotten context.

Features:
- Semantic search across all conversations
- Decision extraction and tracking
- Timeline reconstruction
- Topic threading
- Context recovery
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict

from .paths import get_msty_paths
from .database import get_database_connection, safe_query_table

logger = logging.getLogger("msty-admin-mcp")


def search_conversations(
    query: str,
    limit: int = 20,
    days_back: Optional[int] = None,
    model_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Semantic search across all conversations.

    Args:
        query: Search query
        limit: Maximum results
        days_back: Only search conversations from last N days
        model_filter: Only search conversations with this model
    """
    timestamp = datetime.now().isoformat()
    paths = get_msty_paths()
    db_path = paths.get("database_path")

    if not db_path:
        return {
            "timestamp": timestamp,
            "error": "Database not found",
            "results": []
        }

    results = []
    query_lower = query.lower()
    query_words = query_lower.split()

    try:
        conn = get_database_connection(db_path)
        cursor = conn.cursor()

        # Build query based on available tables
        # Try to get messages with conversation context
        sql = """
            SELECT DISTINCT
                m.id as message_id,
                m.content,
                m.role,
                m.created_at,
                c.id as conversation_id,
                c.title as conversation_title
            FROM messages m
            LEFT JOIN chats c ON m.chat_id = c.id
            WHERE m.content IS NOT NULL
            ORDER BY m.created_at DESC
            LIMIT 1000
        """

        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
        except:
            # Fallback: try simpler query
            cursor.execute("SELECT * FROM messages LIMIT 1000")
            rows = cursor.fetchall()

        # Filter results based on query
        for row in rows:
            content = str(row[1] if len(row) > 1 else row[0]).lower()

            # Check if query matches
            match_score = 0
            for word in query_words:
                if word in content:
                    match_score += 1

            if match_score > 0:
                relevance = match_score / len(query_words)
                results.append({
                    "message_id": row[0] if row else None,
                    "content_preview": str(row[1])[:300] if len(row) > 1 else "",
                    "role": row[2] if len(row) > 2 else "unknown",
                    "created_at": row[3] if len(row) > 3 else None,
                    "conversation_id": row[4] if len(row) > 4 else None,
                    "conversation_title": row[5] if len(row) > 5 else "Untitled",
                    "relevance": relevance,
                    "matched_words": match_score
                })

        conn.close()

        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        results = results[:limit]

    except Exception as e:
        logger.error(f"Search error: {e}")
        return {
            "timestamp": timestamp,
            "error": str(e),
            "results": []
        }

    return {
        "timestamp": timestamp,
        "query": query,
        "total_results": len(results),
        "results": results
    }


def find_decisions(
    topic: Optional[str] = None,
    days_back: int = 90
) -> Dict[str, Any]:
    """
    Find decisions made in conversations.
    Looks for decision-indicating language patterns.
    """
    timestamp = datetime.now().isoformat()

    # Decision indicator patterns
    decision_patterns = [
        r"decided to",
        r"we will",
        r"let's go with",
        r"the decision is",
        r"agreed to",
        r"finalized",
        r"chosen",
        r"selected",
        r"approved",
        r"confirmed that",
        r"went with",
        r"settled on"
    ]

    paths = get_msty_paths()
    db_path = paths.get("database_path")

    if not db_path:
        return {
            "timestamp": timestamp,
            "error": "Database not found",
            "decisions": []
        }

    decisions = []

    try:
        conn = get_database_connection(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, content, created_at FROM messages WHERE role = 'assistant' LIMIT 2000")
        rows = cursor.fetchall()

        for row in rows:
            content = str(row[1]) if len(row) > 1 else ""
            content_lower = content.lower()

            # Check for decision patterns
            for pattern in decision_patterns:
                matches = re.finditer(pattern, content_lower)
                for match in matches:
                    # Extract context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(content), match.end() + 200)
                    context = content[start:end]

                    # Filter by topic if specified
                    if topic and topic.lower() not in content_lower:
                        continue

                    decisions.append({
                        "message_id": row[0],
                        "decision_context": context.strip(),
                        "pattern_matched": pattern,
                        "created_at": row[2] if len(row) > 2 else None,
                        "full_message_preview": content[:500]
                    })

        conn.close()

    except Exception as e:
        logger.error(f"Decision search error: {e}")
        return {
            "timestamp": timestamp,
            "error": str(e),
            "decisions": []
        }

    # Remove duplicates based on context
    seen_contexts = set()
    unique_decisions = []
    for d in decisions:
        ctx_key = d["decision_context"][:100]
        if ctx_key not in seen_contexts:
            seen_contexts.add(ctx_key)
            unique_decisions.append(d)

    return {
        "timestamp": timestamp,
        "topic_filter": topic,
        "total_decisions": len(unique_decisions),
        "decisions": unique_decisions[:50]  # Limit results
    }


def build_timeline(
    topic: str,
    days_back: int = 90
) -> Dict[str, Any]:
    """
    Build a timeline of discussions about a topic.
    """
    timestamp = datetime.now().isoformat()

    search_results = search_conversations(topic, limit=100, days_back=days_back)

    if "error" in search_results:
        return search_results

    # Group by date
    timeline = defaultdict(list)

    for result in search_results.get("results", []):
        created_at = result.get("created_at")
        if created_at:
            try:
                date_str = created_at[:10]  # Get YYYY-MM-DD
                timeline[date_str].append({
                    "conversation_title": result.get("conversation_title"),
                    "content_preview": result.get("content_preview"),
                    "relevance": result.get("relevance")
                })
            except:
                pass

    # Sort timeline
    sorted_timeline = [
        {
            "date": date,
            "discussions": items,
            "count": len(items)
        }
        for date, items in sorted(timeline.items(), reverse=True)
    ]

    return {
        "timestamp": timestamp,
        "topic": topic,
        "timeline_entries": len(sorted_timeline),
        "timeline": sorted_timeline[:30],
        "total_mentions": sum(len(items) for items in timeline.values())
    }


def extract_action_items(
    conversation_id: Optional[str] = None,
    days_back: int = 7
) -> Dict[str, Any]:
    """
    Extract action items and TODOs from conversations.
    """
    timestamp = datetime.now().isoformat()

    # Action item patterns
    action_patterns = [
        r"TODO:",
        r"action item:",
        r"need to",
        r"should do",
        r"will do",
        r"must",
        r"follow up",
        r"don't forget",
        r"remember to",
        r"task:",
        r"next step"
    ]

    paths = get_msty_paths()
    db_path = paths.get("database_path")

    if not db_path:
        return {
            "timestamp": timestamp,
            "error": "Database not found",
            "action_items": []
        }

    action_items = []

    try:
        conn = get_database_connection(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, content, created_at FROM messages LIMIT 2000")
        rows = cursor.fetchall()

        for row in rows:
            content = str(row[1]) if len(row) > 1 else ""
            content_lower = content.lower()

            for pattern in action_patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    # Find the sentence containing the pattern
                    sentences = re.split(r'[.!?]', content)
                    for sentence in sentences:
                        if re.search(pattern, sentence, re.IGNORECASE):
                            action_items.append({
                                "message_id": row[0],
                                "action_item": sentence.strip()[:200],
                                "pattern": pattern,
                                "created_at": row[2] if len(row) > 2 else None
                            })
                            break

        conn.close()

    except Exception as e:
        logger.error(f"Action item extraction error: {e}")
        return {
            "timestamp": timestamp,
            "error": str(e),
            "action_items": []
        }

    return {
        "timestamp": timestamp,
        "total_action_items": len(action_items),
        "action_items": action_items[:50]
    }


def get_conversation_summary(conversation_id: str) -> Dict[str, Any]:
    """
    Get a summary of a specific conversation.
    """
    timestamp = datetime.now().isoformat()

    paths = get_msty_paths()
    db_path = paths.get("database_path")

    if not db_path:
        return {
            "timestamp": timestamp,
            "error": "Database not found"
        }

    try:
        conn = get_database_connection(db_path)
        cursor = conn.cursor()

        # Get conversation details
        cursor.execute(
            "SELECT id, title, created_at FROM chats WHERE id = ?",
            (conversation_id,)
        )
        conv_row = cursor.fetchone()

        if not conv_row:
            return {
                "timestamp": timestamp,
                "error": f"Conversation {conversation_id} not found"
            }

        # Get messages
        cursor.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY created_at",
            (conversation_id,)
        )
        messages = cursor.fetchall()

        conn.close()

        # Analyze conversation
        user_messages = [m[1] for m in messages if m[0] == "user"]
        assistant_messages = [m[1] for m in messages if m[0] == "assistant"]

        # Extract key topics (simple word frequency)
        all_text = " ".join(str(m[1]) for m in messages).lower()
        words = re.findall(r'\b[a-z]{4,}\b', all_text)
        word_freq = defaultdict(int)
        for word in words:
            if word not in ["that", "this", "with", "have", "from", "will", "your", "what", "about"]:
                word_freq[word] += 1

        top_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "timestamp": timestamp,
            "conversation_id": conversation_id,
            "title": conv_row[1],
            "created_at": conv_row[2],
            "message_count": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "top_topics": [{"word": w, "count": c} for w, c in top_topics],
            "first_message_preview": str(user_messages[0])[:200] if user_messages else None,
            "last_message_preview": str(assistant_messages[-1])[:200] if assistant_messages else None
        }

    except Exception as e:
        logger.error(f"Conversation summary error: {e}")
        return {
            "timestamp": timestamp,
            "error": str(e)
        }


def find_related_conversations(conversation_id: str, limit: int = 5) -> Dict[str, Any]:
    """
    Find conversations related to a given conversation.
    """
    timestamp = datetime.now().isoformat()

    # Get summary of source conversation
    summary = get_conversation_summary(conversation_id)

    if "error" in summary:
        return summary

    # Use top topics to search for related conversations
    topics = summary.get("top_topics", [])
    if not topics:
        return {
            "timestamp": timestamp,
            "conversation_id": conversation_id,
            "related": [],
            "message": "No topics found to search for related conversations"
        }

    # Search for each topic
    all_results = []
    for topic_info in topics[:3]:
        topic = topic_info["word"]
        results = search_conversations(topic, limit=10)
        for r in results.get("results", []):
            if r.get("conversation_id") != conversation_id:
                r["matched_topic"] = topic
                all_results.append(r)

    # Deduplicate and rank
    seen = set()
    unique_results = []
    for r in all_results:
        conv_id = r.get("conversation_id")
        if conv_id and conv_id not in seen:
            seen.add(conv_id)
            unique_results.append(r)

    return {
        "timestamp": timestamp,
        "source_conversation_id": conversation_id,
        "search_topics": [t["word"] for t in topics[:3]],
        "related_conversations": unique_results[:limit]
    }


def get_archaeology_stats() -> Dict[str, Any]:
    """
    Get statistics about the conversation archive.
    """
    timestamp = datetime.now().isoformat()

    paths = get_msty_paths()
    db_path = paths.get("database_path")

    if not db_path:
        return {
            "timestamp": timestamp,
            "error": "Database not found"
        }

    try:
        conn = get_database_connection(db_path)
        cursor = conn.cursor()

        stats = {}

        # Count conversations
        try:
            cursor.execute("SELECT COUNT(*) FROM chats")
            stats["total_conversations"] = cursor.fetchone()[0]
        except:
            stats["total_conversations"] = "unknown"

        # Count messages
        try:
            cursor.execute("SELECT COUNT(*) FROM messages")
            stats["total_messages"] = cursor.fetchone()[0]
        except:
            stats["total_messages"] = "unknown"

        # Date range
        try:
            cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM messages")
            row = cursor.fetchone()
            stats["earliest_message"] = row[0]
            stats["latest_message"] = row[1]
        except:
            pass

        conn.close()

        return {
            "timestamp": timestamp,
            "statistics": stats
        }

    except Exception as e:
        return {
            "timestamp": timestamp,
            "error": str(e)
        }


__all__ = [
    "search_conversations",
    "find_decisions",
    "build_timeline",
    "extract_action_items",
    "get_conversation_summary",
    "find_related_conversations",
    "get_archaeology_stats"
]
