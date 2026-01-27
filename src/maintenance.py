"""
Msty Admin MCP - Automated Maintenance

Tools for automated cleanup, optimization, and maintenance tasks.
"""

import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import psutil

from .paths import get_msty_paths
from .database import get_database_connection

logger = logging.getLogger("msty-admin-mcp")


# ============================================================================
# CLEANUP OPERATIONS
# ============================================================================

def identify_cleanup_candidates() -> Dict[str, Any]:
    """
    Identify files and data that can be cleaned up.

    Returns:
        Dict with cleanup candidates and potential space savings
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "candidates": [],
        "total_potential_savings_mb": 0,
    }

    paths = get_msty_paths()
    data_dir = paths.get("data_dir")

    if not data_dir:
        result["error"] = "Could not locate Msty data directory"
        return result

    data_path = Path(data_dir)

    # Check for cache directories
    cache_patterns = ["cache", "Cache", ".cache", "tmp", "temp", "Temp"]
    for pattern in cache_patterns:
        cache_path = data_path / pattern
        if cache_path.exists():
            try:
                size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
                size_mb = round(size / (1024 * 1024), 2)
                if size_mb > 0.1:  # Only report if > 0.1 MB
                    result["candidates"].append({
                        "type": "cache",
                        "path": str(cache_path),
                        "size_mb": size_mb,
                        "file_count": len(list(cache_path.rglob('*'))),
                        "safe_to_delete": True,
                        "priority": "low"
                    })
                    result["total_potential_savings_mb"] += size_mb
            except Exception as e:
                logger.debug(f"Error checking {cache_path}: {e}")

    # Check for log files
    log_patterns = ["*.log", "*.log.*", "logs"]
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.log') or '.log.' in file:
                file_path = Path(root) / file
                try:
                    size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
                    if size_mb > 1:  # Only report logs > 1 MB
                        result["candidates"].append({
                            "type": "log",
                            "path": str(file_path),
                            "size_mb": size_mb,
                            "safe_to_delete": True,
                            "priority": "low"
                        })
                        result["total_potential_savings_mb"] += size_mb
                except:
                    pass

    # Check for old conversation exports
    exports_path = data_path / "Exports"
    if exports_path.exists():
        cutoff = datetime.now() - timedelta(days=30)
        for f in exports_path.iterdir():
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime < cutoff:
                    size_mb = round(f.stat().st_size / (1024 * 1024), 2)
                    result["candidates"].append({
                        "type": "old_export",
                        "path": str(f),
                        "size_mb": size_mb,
                        "age_days": (datetime.now() - mtime).days,
                        "safe_to_delete": True,
                        "priority": "medium"
                    })
                    result["total_potential_savings_mb"] += size_mb
            except:
                pass

    # Check for orphaned model files
    models_path = data_path / "Models"
    if models_path.exists():
        for f in models_path.rglob('*.gguf.part*'):
            # Incomplete downloads
            try:
                size_mb = round(f.stat().st_size / (1024 * 1024), 2)
                result["candidates"].append({
                    "type": "incomplete_download",
                    "path": str(f),
                    "size_mb": size_mb,
                    "safe_to_delete": True,
                    "priority": "high"
                })
                result["total_potential_savings_mb"] += size_mb
            except:
                pass

    # Sort by size
    result["candidates"].sort(key=lambda x: x["size_mb"], reverse=True)

    result["summary"] = {
        "total_candidates": len(result["candidates"]),
        "by_type": {},
        "by_priority": {
            "high": sum(1 for c in result["candidates"] if c.get("priority") == "high"),
            "medium": sum(1 for c in result["candidates"] if c.get("priority") == "medium"),
            "low": sum(1 for c in result["candidates"] if c.get("priority") == "low"),
        }
    }

    for candidate in result["candidates"]:
        ctype = candidate["type"]
        if ctype not in result["summary"]["by_type"]:
            result["summary"]["by_type"][ctype] = {"count": 0, "size_mb": 0}
        result["summary"]["by_type"][ctype]["count"] += 1
        result["summary"]["by_type"][ctype]["size_mb"] += candidate["size_mb"]

    result["total_potential_savings_mb"] = round(result["total_potential_savings_mb"], 2)

    return result


def perform_cleanup(
    cleanup_types: Optional[List[str]] = None,
    max_age_days: int = 30,
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    Perform cleanup operations.

    Args:
        cleanup_types: Types to clean (cache, log, old_export, incomplete_download)
        max_age_days: Maximum age for time-based cleanup
        dry_run: If True, only report what would be deleted

    Returns:
        Dict with cleanup results
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "dry_run": dry_run,
        "cleanup_types": cleanup_types or ["cache", "log", "incomplete_download"],
        "deleted": [],
        "errors": [],
        "space_freed_mb": 0,
    }

    candidates = identify_cleanup_candidates()

    for candidate in candidates.get("candidates", []):
        ctype = candidate["type"]

        # Check if this type should be cleaned
        if cleanup_types and ctype not in cleanup_types:
            continue

        # Check age for time-based items
        if "age_days" in candidate:
            if candidate["age_days"] < max_age_days:
                continue

        path = Path(candidate["path"])

        if dry_run:
            result["deleted"].append({
                "path": str(path),
                "type": ctype,
                "size_mb": candidate["size_mb"],
                "status": "would_delete"
            })
            result["space_freed_mb"] += candidate["size_mb"]
        else:
            try:
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)

                result["deleted"].append({
                    "path": str(path),
                    "type": ctype,
                    "size_mb": candidate["size_mb"],
                    "status": "deleted"
                })
                result["space_freed_mb"] += candidate["size_mb"]

            except Exception as e:
                result["errors"].append({
                    "path": str(path),
                    "error": str(e)
                })

    result["space_freed_mb"] = round(result["space_freed_mb"], 2)
    result["summary"] = {
        "items_processed": len(result["deleted"]),
        "errors": len(result["errors"]),
    }

    if dry_run:
        result["note"] = "Dry run - no files were actually deleted. Set dry_run=False to perform cleanup."

    return result


# ============================================================================
# OPTIMIZATION
# ============================================================================

def optimize_database() -> Dict[str, Any]:
    """
    Optimize the Msty database.

    Returns:
        Dict with optimization results
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "optimizations": [],
        "before": {},
        "after": {},
    }

    try:
        conn = get_database_connection()
        if not conn:
            result["error"] = "Could not connect to database"
            return result

        cursor = conn.cursor()

        # Get database stats before
        cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
        row = cursor.fetchone()
        result["before"]["size_bytes"] = row[0] if row else 0
        result["before"]["size_mb"] = round(result["before"]["size_bytes"] / (1024*1024), 2)

        # Run VACUUM
        try:
            cursor.execute("VACUUM")
            result["optimizations"].append({
                "operation": "VACUUM",
                "status": "success",
                "description": "Rebuilt database to reclaim space"
            })
        except Exception as e:
            result["optimizations"].append({
                "operation": "VACUUM",
                "status": "failed",
                "error": str(e)
            })

        # Run ANALYZE
        try:
            cursor.execute("ANALYZE")
            result["optimizations"].append({
                "operation": "ANALYZE",
                "status": "success",
                "description": "Updated query optimizer statistics"
            })
        except Exception as e:
            result["optimizations"].append({
                "operation": "ANALYZE",
                "status": "failed",
                "error": str(e)
            })

        # Check integrity
        try:
            cursor.execute("PRAGMA integrity_check")
            integrity = cursor.fetchone()
            result["optimizations"].append({
                "operation": "INTEGRITY_CHECK",
                "status": "success" if integrity[0] == "ok" else "warning",
                "result": integrity[0]
            })
        except Exception as e:
            result["optimizations"].append({
                "operation": "INTEGRITY_CHECK",
                "status": "failed",
                "error": str(e)
            })

        # Get database stats after
        cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
        row = cursor.fetchone()
        result["after"]["size_bytes"] = row[0] if row else 0
        result["after"]["size_mb"] = round(result["after"]["size_bytes"] / (1024*1024), 2)

        conn.close()

        # Calculate savings
        result["space_saved_mb"] = round(
            result["before"]["size_mb"] - result["after"]["size_mb"], 2
        )

    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================================
# MAINTENANCE REPORT
# ============================================================================

def generate_maintenance_report() -> Dict[str, Any]:
    """
    Generate a comprehensive maintenance report with recommendations.

    Returns:
        Dict with maintenance report
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "report": {
            "storage": {},
            "database": {},
            "performance": {},
        },
        "recommendations": [],
        "health_score": 100,
    }

    # Storage analysis
    cleanup = identify_cleanup_candidates()
    result["report"]["storage"] = {
        "cleanup_candidates": len(cleanup.get("candidates", [])),
        "potential_savings_mb": cleanup.get("total_potential_savings_mb", 0),
        "by_type": cleanup.get("summary", {}).get("by_type", {}),
    }

    if cleanup.get("total_potential_savings_mb", 0) > 1000:
        result["recommendations"].append({
            "category": "storage",
            "priority": "high",
            "issue": f"Over 1 GB of cleanable data found",
            "action": "Run perform_cleanup() to free space"
        })
        result["health_score"] -= 15
    elif cleanup.get("total_potential_savings_mb", 0) > 100:
        result["recommendations"].append({
            "category": "storage",
            "priority": "medium",
            "issue": f"Over 100 MB of cleanable data found",
            "action": "Consider running cleanup for cache and logs"
        })
        result["health_score"] -= 5

    # Database analysis
    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()

            # Database size
            cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
            row = cursor.fetchone()
            db_size_mb = round(row[0] / (1024*1024), 2) if row else 0

            # Fragmentation check
            cursor.execute("PRAGMA freelist_count")
            freelist = cursor.fetchone()
            freelist_count = freelist[0] if freelist else 0

            result["report"]["database"] = {
                "size_mb": db_size_mb,
                "freelist_pages": freelist_count,
                "needs_vacuum": freelist_count > 100,
            }

            if freelist_count > 100:
                result["recommendations"].append({
                    "category": "database",
                    "priority": "medium",
                    "issue": f"Database has {freelist_count} free pages",
                    "action": "Run optimize_database() to reclaim space"
                })
                result["health_score"] -= 5

            conn.close()
    except Exception as e:
        result["report"]["database"]["error"] = str(e)

    # System performance
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        result["report"]["performance"] = {
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_free_gb": round(disk.free / (1024**3), 2),
        }

        if memory.percent > 85:
            result["recommendations"].append({
                "category": "performance",
                "priority": "high",
                "issue": f"Memory usage is {memory.percent}%",
                "action": "Close unused applications or unload models"
            })
            result["health_score"] -= 20

        if disk.percent > 90:
            result["recommendations"].append({
                "category": "performance",
                "priority": "high",
                "issue": f"Disk usage is {disk.percent}%",
                "action": "Free up disk space immediately"
            })
            result["health_score"] -= 25

    except Exception as e:
        result["report"]["performance"]["error"] = str(e)

    # Sort recommendations by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    result["recommendations"].sort(
        key=lambda x: priority_order.get(x.get("priority"), 3)
    )

    # Ensure health score doesn't go below 0
    result["health_score"] = max(0, result["health_score"])

    # Health status
    if result["health_score"] >= 90:
        result["health_status"] = "excellent"
    elif result["health_score"] >= 70:
        result["health_status"] = "good"
    elif result["health_score"] >= 50:
        result["health_status"] = "fair"
    else:
        result["health_status"] = "needs_attention"

    return result


__all__ = [
    "identify_cleanup_candidates",
    "perform_cleanup",
    "optimize_database",
    "generate_maintenance_report",
]
