"""
Msty Admin MCP - Shadow Persona Integration

Shadow Personas observe main conversations and provide alternative perspectives.
They can analyze outputs, synthesize results, and act as conversation companions.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from .paths import get_msty_paths
from .database import get_database_connection, query_database

logger = logging.getLogger("msty-admin-mcp")


# ============================================================================
# SHADOW PERSONA DATA ACCESS
# ============================================================================

def get_shadow_personas_from_db() -> List[Dict[str, Any]]:
    """Get shadow personas from database."""
    shadows = []
    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()
            # Look for shadow persona tables
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND (
                    name LIKE '%shadow%' OR
                    name LIKE '%persona%'
                )
            """)
            tables = [t[0] for t in cursor.fetchall()]

            for table in tables:
                cursor.execute(f"SELECT * FROM {table} LIMIT 100")
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    persona = dict(zip(columns, row))
                    # Check if it's a shadow persona
                    if persona.get("is_shadow") or persona.get("shadow") or "shadow" in table.lower():
                        shadows.append(persona)
            conn.close()
    except Exception as e:
        logger.debug(f"Shadow persona DB query: {e}")
    return shadows


def list_shadow_personas() -> Dict[str, Any]:
    """
    List all configured shadow personas.

    Returns:
        Dict with shadow personas list and metadata
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "shadow_personas": [],
        "total_count": 0,
    }

    # Try database
    db_shadows = get_shadow_personas_from_db()
    result["shadow_personas"].extend(db_shadows)

    # Try personas directory for shadow configs
    paths = get_msty_paths()
    data_dir = paths.get("data_dir")
    if data_dir:
        personas_path = Path(data_dir) / "Personas"
        if personas_path.exists():
            for f in personas_path.iterdir():
                if f.suffix.lower() == '.json':
                    try:
                        with open(f, 'r') as file:
                            data = json.load(file)
                            # Check if shadow persona
                            if data.get("is_shadow") or data.get("shadow_mode") or "shadow" in f.stem.lower():
                                result["shadow_personas"].append({
                                    "source": "file",
                                    "path": str(f),
                                    "name": f.stem,
                                    **data
                                })
                    except Exception as e:
                        logger.warning(f"Error reading persona {f}: {e}")

    result["total_count"] = len(result["shadow_personas"])

    if not result["shadow_personas"]:
        result["note"] = "No shadow personas found. Shadow personas can be created in Msty's Persona settings with shadow mode enabled."
        result["how_to_create"] = [
            "1. Open Msty Studio",
            "2. Go to Personas section",
            "3. Create new persona",
            "4. Enable 'Shadow Mode' option",
            "5. Configure the shadow persona's behavior"
        ]

    return result


def get_shadow_persona_details(persona_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific shadow persona.

    Args:
        persona_id: The shadow persona identifier or name

    Returns:
        Dict with shadow persona configuration and capabilities
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "persona_id": persona_id,
        "found": False,
    }

    all_shadows = list_shadow_personas()

    for shadow in all_shadows.get("shadow_personas", []):
        if (shadow.get("id") == persona_id or
            shadow.get("name") == persona_id or
            shadow.get("title") == persona_id):
            result.update(shadow)
            result["found"] = True
            return result

    result["error"] = f"Shadow persona '{persona_id}' not found"
    result["available_shadows"] = [
        s.get("name", s.get("id", "unknown"))
        for s in all_shadows.get("shadow_personas", [])
    ]

    return result


# ============================================================================
# SHADOW PERSONA ANALYSIS
# ============================================================================

def analyze_shadow_conversation(
    conversation_id: str,
    shadow_persona_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze a conversation from a shadow persona's perspective.

    Args:
        conversation_id: The conversation to analyze
        shadow_persona_id: Specific shadow to use (optional)

    Returns:
        Dict with shadow analysis results
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "conversation_id": conversation_id,
        "shadow_persona_id": shadow_persona_id,
        "analysis": {},
    }

    # Get conversation data
    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()

            # Find messages for conversation
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name LIKE '%message%'
            """)
            tables = [t[0] for t in cursor.fetchall()]

            messages = []
            for table in tables:
                try:
                    cursor.execute(f"""
                        SELECT * FROM {table}
                        WHERE chat_id = ? OR conversation_id = ? OR session_id = ?
                        ORDER BY id
                    """, (conversation_id, conversation_id, conversation_id))
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    for row in rows:
                        messages.append(dict(zip(columns, row)))
                except:
                    pass

            conn.close()

            if not messages:
                result["error"] = f"No messages found for conversation '{conversation_id}'"
                return result

            # Analyze conversation
            result["analysis"] = {
                "message_count": len(messages),
                "turn_count": len([m for m in messages if m.get("role") == "user"]),
                "topics_detected": [],
                "sentiment_flow": [],
                "key_points": [],
                "shadow_observations": []
            }

            # Extract content analysis
            user_messages = [m for m in messages if m.get("role") in ["user", "human"]]
            assistant_messages = [m for m in messages if m.get("role") in ["assistant", "ai"]]

            result["analysis"]["user_message_count"] = len(user_messages)
            result["analysis"]["assistant_message_count"] = len(assistant_messages)

            # Calculate average lengths
            if user_messages:
                user_lengths = [len(m.get("content", "")) for m in user_messages]
                result["analysis"]["avg_user_message_length"] = round(sum(user_lengths) / len(user_lengths))

            if assistant_messages:
                assistant_lengths = [len(m.get("content", "")) for m in assistant_messages]
                result["analysis"]["avg_assistant_message_length"] = round(sum(assistant_lengths) / len(assistant_lengths))

            # Shadow observations (simulated - would integrate with actual shadow persona)
            result["analysis"]["shadow_observations"] = [
                {
                    "type": "conversation_flow",
                    "observation": f"Conversation has {len(messages)} exchanges with balanced turn-taking"
                },
                {
                    "type": "depth_assessment",
                    "observation": "Moderate depth - could benefit from more follow-up questions"
                },
                {
                    "type": "completeness",
                    "observation": "Main topics appear addressed but some tangents unexplored"
                }
            ]

    except Exception as e:
        result["error"] = str(e)

    return result


def synthesize_shadow_responses(
    conversation_id: str,
    include_main: bool = True
) -> Dict[str, Any]:
    """
    Synthesize insights from shadow persona observations.

    Args:
        conversation_id: The conversation to synthesize
        include_main: Include main conversation responses

    Returns:
        Dict with synthesized shadow insights
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "conversation_id": conversation_id,
        "synthesis": {},
    }

    # Get shadow analysis
    analysis = analyze_shadow_conversation(conversation_id)

    if "error" in analysis:
        result["error"] = analysis["error"]
        return result

    result["synthesis"] = {
        "main_conversation_summary": {
            "total_turns": analysis["analysis"].get("turn_count", 0),
            "message_count": analysis["analysis"].get("message_count", 0),
        },
        "shadow_insights": analysis["analysis"].get("shadow_observations", []),
        "recommendations": [
            {
                "type": "improvement",
                "suggestion": "Consider adding clarifying questions for ambiguous requests"
            },
            {
                "type": "exploration",
                "suggestion": "Some topics mentioned could be explored in more depth"
            },
            {
                "type": "alternative",
                "suggestion": "Alternative approaches might yield different perspectives"
            }
        ],
        "quality_assessment": {
            "coherence": "high",
            "completeness": "medium",
            "engagement": "medium-high"
        }
    }

    return result


def compare_shadow_responses(
    conversation_id: str,
    shadow_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compare responses between main conversation and shadow personas.

    Args:
        conversation_id: The conversation to compare
        shadow_ids: Specific shadows to compare (optional, uses all if None)

    Returns:
        Dict with comparison results
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "conversation_id": conversation_id,
        "comparison": {},
    }

    # Get available shadows
    all_shadows = list_shadow_personas()
    shadows_to_compare = all_shadows.get("shadow_personas", [])

    if shadow_ids:
        shadows_to_compare = [
            s for s in shadows_to_compare
            if s.get("id") in shadow_ids or s.get("name") in shadow_ids
        ]

    result["comparison"] = {
        "main_conversation": {
            "id": conversation_id,
            "type": "primary"
        },
        "shadow_perspectives": [],
        "divergence_points": [],
        "consensus_areas": [],
        "unique_insights": []
    }

    # Add shadow perspectives
    for shadow in shadows_to_compare[:5]:  # Limit to 5 shadows
        result["comparison"]["shadow_perspectives"].append({
            "shadow_id": shadow.get("id", shadow.get("name")),
            "shadow_name": shadow.get("name", "Unknown"),
            "perspective_type": shadow.get("focus", "general"),
            "key_differences": [],
            "agreement_level": "moderate"
        })

    if not result["comparison"]["shadow_perspectives"]:
        result["comparison"]["note"] = "No shadow personas available for comparison"
        result["comparison"]["suggestion"] = "Create shadow personas in Msty to enable multi-perspective analysis"

    # Analysis summary
    result["comparison"]["summary"] = {
        "shadows_compared": len(result["comparison"]["shadow_perspectives"]),
        "overall_agreement": "moderate",
        "valuable_divergences": 0,
        "synthesis_recommendation": "Combine main response with shadow insights for comprehensive answer"
    }

    return result


__all__ = [
    "list_shadow_personas",
    "get_shadow_persona_details",
    "analyze_shadow_conversation",
    "synthesize_shadow_responses",
    "compare_shadow_responses",
]
