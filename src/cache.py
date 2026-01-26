"""
Msty Admin MCP - Response Caching

Simple TTL cache for API responses.
"""

import time
from typing import Any, Dict, Optional


class ResponseCache:
    """Simple TTL cache for API responses"""

    def __init__(self, default_ttl: int = 30):
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Cache a value with TTL"""
        ttl = ttl or self._default_ttl
        self._cache[key] = (value, time.time() + ttl)

    def invalidate(self, key: str = None) -> None:
        """Clear specific key or entire cache"""
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()

    def stats(self) -> dict:
        """Get cache statistics"""
        now = time.time()
        valid = sum(1 for _, (_, exp) in self._cache.items() if exp > now)
        return {"total_entries": len(self._cache), "valid_entries": valid}


# Global cache instance
_response_cache = ResponseCache(default_ttl=30)


def get_cached_models() -> Optional[dict]:
    """Get cached model list"""
    return _response_cache.get("models_list")


def cache_models(models_data: dict, ttl: int = 60) -> None:
    """Cache model list for TTL seconds"""
    _response_cache.set("models_list", models_data, ttl)


def get_cache() -> ResponseCache:
    """Get the global cache instance"""
    return _response_cache


__all__ = [
    "ResponseCache",
    "get_cached_models",
    "cache_models",
    "get_cache"
]
