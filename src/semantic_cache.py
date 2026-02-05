"""
Msty Admin MCP - Semantic Response Cache (Phase 29)

Embedding-based similarity caching - finds "close enough" answers
instead of requiring exact prompt matches.

Features:
- Semantic similarity matching using embeddings
- Configurable similarity threshold
- Cost savings tracking
- Cache statistics and management
- TTL-based expiration
"""

import json
import logging
import hashlib
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from .paths import get_msty_paths
from .network import make_api_request, get_available_service_ports

logger = logging.getLogger("msty-admin-mcp")


@dataclass
class CacheEntry:
    """A cached response with its embedding."""
    id: str
    prompt: str
    prompt_hash: str
    response: str
    model_id: str
    embedding: List[float]
    created_at: str
    expires_at: str
    hit_count: int = 0
    last_accessed: Optional[str] = None
    tokens_saved: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# Cache storage
_semantic_cache: Dict[str, CacheEntry] = {}
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "total_tokens_saved": 0,
    "estimated_cost_saved": 0.0
}

# Default configuration
DEFAULT_CONFIG = {
    "similarity_threshold": 0.85,  # Minimum similarity to consider a cache hit
    "max_cache_size": 1000,
    "default_ttl_hours": 24,
    "embedding_model": None,  # Auto-detect
    "track_savings": True
}

_config = DEFAULT_CONFIG.copy()


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(v1) != len(v2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(b * b for b in v2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def get_embedding(text: str, model_id: Optional[str] = None) -> Optional[List[float]]:
    """
    Get embedding for text using available embedding model.
    """
    ports = get_available_service_ports()

    # Try each service
    for service, port in ports.items():
        try:
            # First check if there's an embedding model available
            models_response = make_api_request(f"http://127.0.0.1:{port}/v1/models")
            if not models_response or "data" not in models_response:
                continue

            # Find embedding model
            embedding_model = model_id
            if not embedding_model:
                for model in models_response["data"]:
                    model_name = model.get("id", "").lower()
                    if any(kw in model_name for kw in ["embed", "bge", "nomic", "e5"]):
                        embedding_model = model["id"]
                        break

            if not embedding_model:
                # Use first available model with embeddings endpoint
                embedding_model = models_response["data"][0]["id"] if models_response["data"] else None

            if not embedding_model:
                continue

            # Get embedding
            response = make_api_request(
                f"http://127.0.0.1:{port}/v1/embeddings",
                method="POST",
                data={
                    "model": embedding_model,
                    "input": text[:8000]  # Limit input size
                }
            )

            if response and "data" in response and response["data"]:
                return response["data"][0].get("embedding")

        except Exception as e:
            logger.debug(f"Could not get embedding from {service}: {e}")
            continue

    return None


def _generate_simple_embedding(text: str) -> List[float]:
    """
    Generate a simple text-based embedding when no embedding model is available.
    This is a fallback that uses character-level features.
    """
    # Simple bag-of-characters approach (not as good as real embeddings)
    text_lower = text.lower()

    # 256-dimensional embedding based on character frequencies
    embedding = [0.0] * 256

    for char in text_lower:
        if ord(char) < 256:
            embedding[ord(char)] += 1

    # Normalize
    total = sum(embedding) or 1
    embedding = [v / total for v in embedding]

    return embedding


def cache_response(
    prompt: str,
    response: str,
    model_id: str,
    ttl_hours: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Cache a response with its semantic embedding.

    Args:
        prompt: The original prompt
        response: The model's response
        model_id: The model that generated the response
        ttl_hours: Time-to-live in hours (default from config)
        metadata: Additional metadata to store
    """
    timestamp = datetime.now()
    ttl = ttl_hours or _config["default_ttl_hours"]

    # Generate embedding
    embedding = get_embedding(prompt)
    if embedding is None:
        embedding = _generate_simple_embedding(prompt)

    # Create cache entry
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    entry_id = f"cache_{prompt_hash[:12]}_{timestamp.strftime('%Y%m%d%H%M%S')}"

    entry = CacheEntry(
        id=entry_id,
        prompt=prompt,
        prompt_hash=prompt_hash,
        response=response,
        model_id=model_id,
        embedding=embedding,
        created_at=timestamp.isoformat(),
        expires_at=(timestamp + timedelta(hours=ttl)).isoformat(),
        tokens_saved=len(response.split()) * 2,  # Rough estimate
        metadata=metadata or {}
    )

    # Check cache size limit
    if len(_semantic_cache) >= _config["max_cache_size"]:
        _evict_oldest_entries(1)

    _semantic_cache[entry_id] = entry

    return {
        "timestamp": timestamp.isoformat(),
        "cached": True,
        "entry_id": entry_id,
        "ttl_hours": ttl,
        "expires_at": entry.expires_at,
        "embedding_size": len(embedding)
    }


def find_similar_response(
    prompt: str,
    threshold: Optional[float] = None,
    model_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Find a cached response similar to the given prompt.

    Args:
        prompt: The prompt to find similar responses for
        threshold: Similarity threshold (0-1, higher = more similar)
        model_filter: Only consider responses from this model
    """
    timestamp = datetime.now()
    threshold = threshold or _config["similarity_threshold"]

    # Get embedding for query
    query_embedding = get_embedding(prompt)
    if query_embedding is None:
        query_embedding = _generate_simple_embedding(prompt)

    # Clean expired entries
    _clean_expired_entries()

    # Find best match
    best_match = None
    best_similarity = 0.0

    for entry_id, entry in _semantic_cache.items():
        # Apply model filter
        if model_filter and entry.model_id != model_filter:
            continue

        # Calculate similarity
        similarity = cosine_similarity(query_embedding, entry.embedding)

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = entry

    # Check if we found a match above threshold
    if best_match and best_similarity >= threshold:
        # Update stats
        best_match.hit_count += 1
        best_match.last_accessed = timestamp.isoformat()
        _cache_stats["hits"] += 1
        _cache_stats["total_tokens_saved"] += best_match.tokens_saved
        _cache_stats["estimated_cost_saved"] += best_match.tokens_saved * 0.00001  # Rough estimate

        return {
            "timestamp": timestamp.isoformat(),
            "cache_hit": True,
            "similarity": best_similarity,
            "threshold": threshold,
            "entry_id": best_match.id,
            "original_prompt": best_match.prompt[:200] + "..." if len(best_match.prompt) > 200 else best_match.prompt,
            "response": best_match.response,
            "model_id": best_match.model_id,
            "hit_count": best_match.hit_count,
            "tokens_saved": best_match.tokens_saved,
            "created_at": best_match.created_at
        }
    else:
        _cache_stats["misses"] += 1
        return {
            "timestamp": timestamp.isoformat(),
            "cache_hit": False,
            "best_similarity": best_similarity if best_match else 0.0,
            "threshold": threshold,
            "suggestion": "No sufficiently similar response found"
        }


def get_or_generate(
    prompt: str,
    generator_fn,
    threshold: Optional[float] = None,
    ttl_hours: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get cached response or generate and cache a new one.

    Args:
        prompt: The prompt to process
        generator_fn: Function to call if cache miss (should return response string)
        threshold: Similarity threshold
        ttl_hours: TTL for new cache entry
    """
    # Try cache first
    cache_result = find_similar_response(prompt, threshold)

    if cache_result.get("cache_hit"):
        cache_result["source"] = "cache"
        return cache_result

    # Generate new response
    try:
        response = generator_fn(prompt)

        # Cache the response
        cache_response(prompt, response, "generated", ttl_hours)

        return {
            "timestamp": datetime.now().isoformat(),
            "cache_hit": False,
            "source": "generated",
            "response": response,
            "cached_for_future": True
        }
    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "cache_hit": False,
            "source": "error",
            "error": str(e)
        }


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    _clean_expired_entries()

    total_requests = _cache_stats["hits"] + _cache_stats["misses"]
    hit_rate = _cache_stats["hits"] / total_requests if total_requests > 0 else 0.0

    return {
        "timestamp": datetime.now().isoformat(),
        "total_entries": len(_semantic_cache),
        "max_entries": _config["max_cache_size"],
        "hits": _cache_stats["hits"],
        "misses": _cache_stats["misses"],
        "hit_rate": hit_rate,
        "total_tokens_saved": _cache_stats["total_tokens_saved"],
        "estimated_cost_saved": f"${_cache_stats['estimated_cost_saved']:.4f}",
        "similarity_threshold": _config["similarity_threshold"],
        "default_ttl_hours": _config["default_ttl_hours"]
    }


def clear_cache() -> Dict[str, Any]:
    """Clear the entire cache."""
    count = len(_semantic_cache)
    _semantic_cache.clear()

    return {
        "timestamp": datetime.now().isoformat(),
        "cleared": True,
        "entries_removed": count
    }


def delete_cache_entry(entry_id: str) -> Dict[str, Any]:
    """Delete a specific cache entry."""
    if entry_id in _semantic_cache:
        del _semantic_cache[entry_id]
        return {
            "timestamp": datetime.now().isoformat(),
            "deleted": True,
            "entry_id": entry_id
        }
    return {"error": f"Entry {entry_id} not found"}


def list_cache_entries(limit: int = 50) -> Dict[str, Any]:
    """List cache entries with metadata."""
    _clean_expired_entries()

    entries = []
    for entry_id, entry in list(_semantic_cache.items())[:limit]:
        entries.append({
            "entry_id": entry_id,
            "prompt_preview": entry.prompt[:100] + "..." if len(entry.prompt) > 100 else entry.prompt,
            "model_id": entry.model_id,
            "hit_count": entry.hit_count,
            "created_at": entry.created_at,
            "expires_at": entry.expires_at,
            "last_accessed": entry.last_accessed
        })

    return {
        "timestamp": datetime.now().isoformat(),
        "total_entries": len(_semantic_cache),
        "showing": len(entries),
        "entries": entries
    }


def configure_cache(
    similarity_threshold: Optional[float] = None,
    max_cache_size: Optional[int] = None,
    default_ttl_hours: Optional[int] = None
) -> Dict[str, Any]:
    """Update cache configuration."""
    if similarity_threshold is not None:
        _config["similarity_threshold"] = max(0.0, min(1.0, similarity_threshold))

    if max_cache_size is not None:
        _config["max_cache_size"] = max(10, max_cache_size)

    if default_ttl_hours is not None:
        _config["default_ttl_hours"] = max(1, default_ttl_hours)

    return {
        "timestamp": datetime.now().isoformat(),
        "config": _config.copy()
    }


def _clean_expired_entries():
    """Remove expired cache entries."""
    now = datetime.now()
    expired = [
        entry_id for entry_id, entry in _semantic_cache.items()
        if datetime.fromisoformat(entry.expires_at) < now
    ]

    for entry_id in expired:
        del _semantic_cache[entry_id]


def _evict_oldest_entries(count: int):
    """Evict oldest entries to make room."""
    # Sort by last_accessed (or created_at if never accessed)
    sorted_entries = sorted(
        _semantic_cache.items(),
        key=lambda x: x[1].last_accessed or x[1].created_at
    )

    for entry_id, _ in sorted_entries[:count]:
        del _semantic_cache[entry_id]


__all__ = [
    "cache_response",
    "find_similar_response",
    "get_or_generate",
    "get_cache_stats",
    "clear_cache",
    "delete_cache_entry",
    "list_cache_entries",
    "configure_cache",
    "cosine_similarity"
]
