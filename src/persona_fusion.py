"""
Msty Admin MCP - Persona Fusion (Phase 35)

Dynamically combine multiple personas for complex tasks requiring
diverse expertise. Create hybrid personas on-the-fly.

Features:
- Combine multiple personas
- Weighted trait merging
- Dynamic persona creation
- Context-aware persona selection
- Persona compatibility analysis
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from .paths import get_msty_paths

logger = logging.getLogger("msty-admin-mcp")


def _safe_query_personas(query: str, limit: int = 100) -> Dict[str, Any]:
    """Safely query Msty personas database if available."""
    try:
        from .database import query_database
        paths = get_msty_paths()
        db_path = paths.get("database_path")
        if db_path:
            rows = query_database(db_path, query)
            return {"rows": rows[:limit] if rows else []}
    except Exception:
        pass
    return {"rows": []}


@dataclass
class PersonaTrait:
    """A single trait of a persona."""
    name: str
    weight: float  # 0.0 to 1.0
    description: str
    keywords: List[str] = field(default_factory=list)


@dataclass
class FusedPersona:
    """A dynamically created fused persona."""
    id: str
    name: str
    source_personas: List[str]
    weights: Dict[str, float]
    system_prompt: str
    created_at: str
    traits: List[PersonaTrait] = field(default_factory=list)
    use_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# Storage
_fused_personas: Dict[str, FusedPersona] = {}
_persona_cache: Dict[str, Dict[str, Any]] = {}


# Predefined persona templates
PERSONA_TEMPLATES = {
    "coder": {
        "traits": ["technical", "precise", "methodical", "debugging-oriented"],
        "keywords": ["code", "programming", "debug", "implement", "function", "class", "algorithm"],
        "base_prompt": "You are an expert programmer focused on writing clean, efficient code."
    },
    "researcher": {
        "traits": ["analytical", "thorough", "citation-focused", "fact-checking"],
        "keywords": ["research", "analyze", "investigate", "sources", "evidence", "data"],
        "base_prompt": "You are a meticulous researcher who values accuracy and comprehensive analysis."
    },
    "writer": {
        "traits": ["creative", "eloquent", "narrative-focused", "stylistic"],
        "keywords": ["write", "draft", "compose", "story", "article", "creative"],
        "base_prompt": "You are a skilled writer with a talent for engaging prose."
    },
    "analyst": {
        "traits": ["logical", "data-driven", "systematic", "pattern-recognition"],
        "keywords": ["analyze", "metrics", "trends", "data", "insights", "evaluate"],
        "base_prompt": "You are an analytical expert who excels at finding patterns and insights."
    },
    "teacher": {
        "traits": ["patient", "explanatory", "step-by-step", "encouraging"],
        "keywords": ["explain", "teach", "learn", "understand", "how", "why"],
        "base_prompt": "You are a patient educator who explains concepts clearly and thoroughly."
    },
    "critic": {
        "traits": ["discerning", "constructive", "detailed-feedback", "improvement-focused"],
        "keywords": ["review", "critique", "improve", "feedback", "issues", "suggestions"],
        "base_prompt": "You are a constructive critic who provides detailed, actionable feedback."
    },
    "strategist": {
        "traits": ["big-picture", "planning-oriented", "risk-aware", "goal-focused"],
        "keywords": ["strategy", "plan", "approach", "goal", "roadmap", "direction"],
        "base_prompt": "You are a strategic thinker who plans for long-term success."
    },
    "creative": {
        "traits": ["imaginative", "innovative", "unconventional", "brainstorming"],
        "keywords": ["creative", "ideas", "brainstorm", "innovative", "novel", "unique"],
        "base_prompt": "You are a creative innovator who generates novel ideas and approaches."
    }
}


def _load_msty_personas() -> List[Dict[str, Any]]:
    """Load personas from Msty database."""
    try:
        result = _safe_query_personas(
            "SELECT id, name, system_prompt FROM personas LIMIT 100"
        )
        return result.get("rows", [])
    except Exception as e:
        logger.debug(f"Could not load Msty personas: {e}")
        return []


def _get_persona_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Get a persona by name (from cache, templates, or Msty)."""
    # Check cache
    if name in _persona_cache:
        return _persona_cache[name]

    # Check templates
    if name.lower() in PERSONA_TEMPLATES:
        template = PERSONA_TEMPLATES[name.lower()]
        persona = {
            "id": f"template_{name.lower()}",
            "name": name.lower(),
            "system_prompt": template["base_prompt"],
            "traits": template["traits"],
            "keywords": template["keywords"]
        }
        _persona_cache[name] = persona
        return persona

    # Try Msty database
    try:
        result = _safe_query_personas(
            f"SELECT id, name, system_prompt FROM personas WHERE name LIKE '%{name}%' LIMIT 1"
        )
        if result.get("rows"):
            row = result["rows"][0]
            persona = {
                "id": row.get("id"),
                "name": row.get("name"),
                "system_prompt": row.get("system_prompt", ""),
                "traits": [],
                "keywords": []
            }
            _persona_cache[name] = persona
            return persona
    except Exception:
        pass

    return None


def fuse_personas(
    persona_names: List[str],
    weights: Optional[Dict[str, float]] = None,
    task_context: Optional[str] = None,
    custom_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a fused persona from multiple source personas.

    Args:
        persona_names: List of persona names to fuse
        weights: Optional weight for each persona (0-1, defaults to equal)
        task_context: Optional context to influence fusion
        custom_name: Optional custom name for the fused persona
    """
    timestamp = datetime.now()
    fusion_id = f"fusion_{uuid.uuid4().hex[:8]}"

    # Load source personas
    source_personas = []
    for name in persona_names:
        persona = _get_persona_by_name(name)
        if persona:
            source_personas.append(persona)
        else:
            # Create minimal persona from template if available
            if name.lower() in PERSONA_TEMPLATES:
                template = PERSONA_TEMPLATES[name.lower()]
                source_personas.append({
                    "id": f"template_{name.lower()}",
                    "name": name.lower(),
                    "system_prompt": template["base_prompt"],
                    "traits": template["traits"],
                    "keywords": template["keywords"]
                })

    if not source_personas:
        return {
            "timestamp": timestamp.isoformat(),
            "error": "No valid personas found to fuse",
            "requested": persona_names
        }

    # Calculate weights
    if weights is None:
        # Equal weights
        weights = {p["name"]: 1.0 / len(source_personas) for p in source_personas}
    else:
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

    # Fuse system prompts
    fused_prompt_parts = []
    fused_prompt_parts.append("You are a versatile AI assistant combining multiple areas of expertise:\n")

    for persona in source_personas:
        weight = weights.get(persona["name"], 1.0 / len(source_personas))
        if weight >= 0.1:  # Only include significant contributions
            prompt = persona.get("system_prompt", "")
            if prompt:
                # Weight-based emphasis
                emphasis = "primarily" if weight > 0.5 else "also" if weight > 0.25 else "with some"
                fused_prompt_parts.append(f"\n{emphasis.title()} {persona['name'].title()}: {prompt}")

    # Add task context if provided
    if task_context:
        fused_prompt_parts.append(f"\n\nCurrent task context: {task_context}")

    fused_prompt = "\n".join(fused_prompt_parts)

    # Collect all traits
    all_traits = []
    for persona in source_personas:
        for trait in persona.get("traits", []):
            all_traits.append(PersonaTrait(
                name=trait,
                weight=weights.get(persona["name"], 0.5),
                description=f"From {persona['name']}",
                keywords=persona.get("keywords", [])
            ))

    # Create fused persona
    fused = FusedPersona(
        id=fusion_id,
        name=custom_name or f"Fused: {', '.join(p['name'] for p in source_personas[:3])}",
        source_personas=[p["name"] for p in source_personas],
        weights=weights,
        system_prompt=fused_prompt,
        created_at=timestamp.isoformat(),
        traits=all_traits
    )

    _fused_personas[fusion_id] = fused

    return {
        "timestamp": timestamp.isoformat(),
        "fusion_id": fusion_id,
        "name": fused.name,
        "source_personas": fused.source_personas,
        "weights": weights,
        "system_prompt": fused_prompt,
        "trait_count": len(all_traits),
        "created": True
    }


def suggest_fusion_for_task(task_description: str) -> Dict[str, Any]:
    """
    Suggest which personas to fuse for a given task.
    """
    task_lower = task_description.lower()

    # Score each persona template
    scores = {}
    for name, template in PERSONA_TEMPLATES.items():
        score = 0
        for keyword in template["keywords"]:
            if keyword in task_lower:
                score += 1

        # Boost based on certain task patterns
        if "code" in task_lower or "program" in task_lower:
            if name == "coder":
                score += 3
        if "write" in task_lower or "draft" in task_lower:
            if name == "writer":
                score += 3
        if "analyze" in task_lower or "data" in task_lower:
            if name == "analyst":
                score += 3
        if "research" in task_lower or "investigate" in task_lower:
            if name == "researcher":
                score += 3
        if "explain" in task_lower or "teach" in task_lower:
            if name == "teacher":
                score += 3
        if "review" in task_lower or "critique" in task_lower:
            if name == "critic":
                score += 3
        if "plan" in task_lower or "strategy" in task_lower:
            if name == "strategist":
                score += 3
        if "creative" in task_lower or "ideas" in task_lower:
            if name == "creative":
                score += 3

        if score > 0:
            scores[name] = score

    if not scores:
        # Default suggestion
        return {
            "timestamp": datetime.now().isoformat(),
            "task": task_description[:100],
            "suggestion": {
                "personas": ["analyst"],
                "weights": {"analyst": 1.0},
                "reasoning": "Default suggestion for general tasks"
            }
        }

    # Sort by score
    sorted_personas = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Take top 2-3 personas
    top_personas = sorted_personas[:min(3, len(sorted_personas))]

    # Calculate suggested weights based on scores
    total_score = sum(s for _, s in top_personas)
    suggested_weights = {
        name: score / total_score
        for name, score in top_personas
    }

    return {
        "timestamp": datetime.now().isoformat(),
        "task": task_description[:100],
        "suggestion": {
            "personas": [name for name, _ in top_personas],
            "weights": suggested_weights,
            "reasoning": f"Based on task keywords matching {len(top_personas)} persona types",
            "scores": dict(sorted_personas[:5])
        }
    }


def get_fused_persona(fusion_id: str) -> Dict[str, Any]:
    """Get details of a fused persona."""
    if fusion_id not in _fused_personas:
        return {"error": f"Fused persona {fusion_id} not found"}

    fused = _fused_personas[fusion_id]

    return {
        "timestamp": datetime.now().isoformat(),
        "fusion_id": fusion_id,
        "name": fused.name,
        "source_personas": fused.source_personas,
        "weights": fused.weights,
        "system_prompt": fused.system_prompt,
        "created_at": fused.created_at,
        "use_count": fused.use_count,
        "traits": [
            {"name": t.name, "weight": t.weight, "description": t.description}
            for t in fused.traits
        ]
    }


def list_fused_personas() -> Dict[str, Any]:
    """List all fused personas."""
    personas_list = []
    for fusion_id, fused in _fused_personas.items():
        personas_list.append({
            "fusion_id": fusion_id,
            "name": fused.name,
            "source_personas": fused.source_personas,
            "created_at": fused.created_at,
            "use_count": fused.use_count
        })

    return {
        "timestamp": datetime.now().isoformat(),
        "total_fused": len(personas_list),
        "fused_personas": personas_list
    }


def list_available_personas() -> Dict[str, Any]:
    """List all available persona templates and Msty personas."""
    templates = []
    for name, template in PERSONA_TEMPLATES.items():
        templates.append({
            "name": name,
            "type": "template",
            "traits": template["traits"],
            "keywords": template["keywords"][:5],
            "base_prompt_preview": template["base_prompt"][:100]
        })

    # Try to load Msty personas
    msty_personas = []
    try:
        result = _safe_query_personas(
            "SELECT id, name, system_prompt FROM personas LIMIT 20"
        )
        for row in result.get("rows", []):
            msty_personas.append({
                "name": row.get("name"),
                "type": "msty",
                "id": row.get("id"),
                "prompt_preview": (row.get("system_prompt") or "")[:100]
            })
    except Exception:
        pass

    return {
        "timestamp": datetime.now().isoformat(),
        "templates": templates,
        "template_count": len(templates),
        "msty_personas": msty_personas,
        "msty_count": len(msty_personas)
    }


def analyze_persona_compatibility(persona_names: List[str]) -> Dict[str, Any]:
    """
    Analyze how well personas would work together.
    """
    if len(persona_names) < 2:
        return {
            "timestamp": datetime.now().isoformat(),
            "error": "Need at least 2 personas to analyze compatibility"
        }

    # Load personas
    personas = []
    for name in persona_names:
        if name.lower() in PERSONA_TEMPLATES:
            template = PERSONA_TEMPLATES[name.lower()]
            personas.append({
                "name": name.lower(),
                "traits": set(template["traits"]),
                "keywords": set(template["keywords"])
            })

    if len(personas) < 2:
        return {
            "timestamp": datetime.now().isoformat(),
            "error": "Could not find enough personas to analyze"
        }

    # Calculate compatibility metrics
    compatibility_scores = []

    for i, p1 in enumerate(personas):
        for j, p2 in enumerate(personas):
            if i >= j:
                continue

            # Trait overlap
            trait_overlap = len(p1["traits"] & p2["traits"])
            trait_total = len(p1["traits"] | p2["traits"])
            trait_similarity = trait_overlap / trait_total if trait_total > 0 else 0

            # Keyword overlap
            keyword_overlap = len(p1["keywords"] & p2["keywords"])
            keyword_total = len(p1["keywords"] | p2["keywords"])
            keyword_similarity = keyword_overlap / keyword_total if keyword_total > 0 else 0

            # Complementary score (different traits can be good)
            complementary = 1 - trait_similarity

            compatibility_scores.append({
                "pair": [p1["name"], p2["name"]],
                "trait_similarity": trait_similarity,
                "keyword_overlap": keyword_similarity,
                "complementary_score": complementary,
                "recommendation": "excellent" if complementary > 0.6 else "good" if complementary > 0.3 else "overlapping"
            })

    # Overall assessment
    avg_complementary = sum(s["complementary_score"] for s in compatibility_scores) / len(compatibility_scores)

    return {
        "timestamp": datetime.now().isoformat(),
        "personas_analyzed": [p["name"] for p in personas],
        "pair_analysis": compatibility_scores,
        "overall_compatibility": avg_complementary,
        "assessment": "highly complementary" if avg_complementary > 0.6 else "moderately compatible" if avg_complementary > 0.3 else "significant overlap",
        "recommendation": "Great combination for diverse tasks" if avg_complementary > 0.5 else "Consider adding contrasting personas for more diversity"
    }


def delete_fused_persona(fusion_id: str) -> Dict[str, Any]:
    """Delete a fused persona."""
    if fusion_id not in _fused_personas:
        return {"error": f"Fused persona {fusion_id} not found"}

    fused = _fused_personas.pop(fusion_id)

    return {
        "timestamp": datetime.now().isoformat(),
        "fusion_id": fusion_id,
        "deleted": True,
        "name": fused.name
    }


def record_fusion_usage(fusion_id: str) -> Dict[str, Any]:
    """Record that a fused persona was used."""
    if fusion_id not in _fused_personas:
        return {"error": f"Fused persona {fusion_id} not found"}

    _fused_personas[fusion_id].use_count += 1

    return {
        "timestamp": datetime.now().isoformat(),
        "fusion_id": fusion_id,
        "use_count": _fused_personas[fusion_id].use_count
    }


def quick_fuse_for_task(task_description: str) -> Dict[str, Any]:
    """
    Convenience function: suggest and create a fused persona for a task.
    """
    # Get suggestion
    suggestion = suggest_fusion_for_task(task_description)

    if "error" in suggestion:
        return suggestion

    # Create the fusion
    personas = suggestion["suggestion"]["personas"]
    weights = suggestion["suggestion"]["weights"]

    fusion_result = fuse_personas(
        persona_names=personas,
        weights=weights,
        task_context=task_description
    )

    return {
        "timestamp": datetime.now().isoformat(),
        "task": task_description[:100],
        "suggestion": suggestion["suggestion"],
        "fusion_result": fusion_result
    }


__all__ = [
    "fuse_personas",
    "suggest_fusion_for_task",
    "get_fused_persona",
    "list_fused_personas",
    "list_available_personas",
    "analyze_persona_compatibility",
    "delete_fused_persona",
    "record_fusion_usage",
    "quick_fuse_for_task",
    "PERSONA_TEMPLATES"
]
