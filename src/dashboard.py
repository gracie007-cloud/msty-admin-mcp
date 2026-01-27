"""
Msty Admin MCP - Health Monitoring Dashboard

Unified health monitoring and alerting for all Msty services.
"""

import json
import logging
import urllib.request
import urllib.error
from datetime import datetime
from typing import Optional, List, Dict, Any

import psutil

from .paths import get_msty_paths
from .database import get_database_connection
from .network import is_process_running, get_available_service_ports
from .constants import (
    LOCAL_AI_SERVICE_PORT,
    MLX_SERVICE_PORT,
    LLAMACPP_SERVICE_PORT,
    VIBE_PROXY_PORT,
    SIDECAR_HOST
)

logger = logging.getLogger("msty-admin-mcp")


# ============================================================================
# SERVICE STATUS
# ============================================================================

def check_service_health(host: str, port: int, timeout: int = 3) -> Dict[str, Any]:
    """Check health of a single service."""
    result = {
        "port": port,
        "host": host,
        "healthy": False,
        "response_time_ms": None,
        "error": None
    }

    try:
        import time
        start = time.time()
        url = f"http://{host}:{port}/v1/models"

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result["response_time_ms"] = round((time.time() - start) * 1000, 2)
            result["healthy"] = response.status == 200
            result["status_code"] = response.status

            # Try to get model count
            try:
                data = json.loads(response.read().decode())
                if "data" in data:
                    result["model_count"] = len(data["data"])
            except:
                pass

    except urllib.error.URLError as e:
        result["error"] = f"Connection failed: {e.reason}"
    except Exception as e:
        result["error"] = str(e)

    return result


def get_dashboard_status() -> Dict[str, Any]:
    """
    Get combined health status of all Msty services.

    Returns:
        Dict with comprehensive status dashboard
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "unknown",
        "services": {},
        "system": {},
        "database": {},
        "alerts": []
    }

    # Check all services
    services = {
        "local_ai": {"port": LOCAL_AI_SERVICE_PORT, "name": "Local AI Service"},
        "mlx": {"port": MLX_SERVICE_PORT, "name": "MLX Service"},
        "llamacpp": {"port": LLAMACPP_SERVICE_PORT, "name": "LLaMA.cpp Service"},
        "vibe_proxy": {"port": VIBE_PROXY_PORT, "name": "Vibe CLI Proxy"},
    }

    healthy_services = 0
    total_services = len(services)

    for service_id, config in services.items():
        health = check_service_health(SIDECAR_HOST, config["port"])
        health["name"] = config["name"]
        result["services"][service_id] = health

        if health["healthy"]:
            healthy_services += 1
        elif health.get("error"):
            result["alerts"].append({
                "severity": "warning",
                "service": service_id,
                "message": f"{config['name']} unavailable: {health['error']}"
            })

    # Check Msty process
    msty_running = is_process_running("msty")
    result["services"]["msty_app"] = {
        "name": "Msty Studio",
        "running": msty_running,
        "healthy": msty_running
    }
    if msty_running:
        healthy_services += 1
    else:
        result["alerts"].append({
            "severity": "critical",
            "service": "msty_app",
            "message": "Msty Studio is not running"
        })

    total_services += 1

    # System resources
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        disk = psutil.disk_usage('/')

        result["system"] = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2),
        }

        # Resource alerts
        if memory.percent > 90:
            result["alerts"].append({
                "severity": "critical",
                "service": "system",
                "message": f"Memory usage critical: {memory.percent}%"
            })
        elif memory.percent > 75:
            result["alerts"].append({
                "severity": "warning",
                "service": "system",
                "message": f"Memory usage high: {memory.percent}%"
            })

        if disk.percent > 90:
            result["alerts"].append({
                "severity": "critical",
                "service": "system",
                "message": f"Disk usage critical: {disk.percent}%"
            })

    except Exception as e:
        result["system"]["error"] = str(e)

    # Database status
    try:
        conn = get_database_connection()
        if conn:
            result["database"]["connected"] = True
            cursor = conn.cursor()

            # Get database size
            cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
            row = cursor.fetchone()
            if row:
                result["database"]["size_mb"] = round(row[0] / (1024 * 1024), 2)

            # Get table count
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            row = cursor.fetchone()
            if row:
                result["database"]["table_count"] = row[0]

            conn.close()
        else:
            result["database"]["connected"] = False
            result["alerts"].append({
                "severity": "warning",
                "service": "database",
                "message": "Could not connect to Msty database"
            })
    except Exception as e:
        result["database"]["error"] = str(e)

    # Calculate overall status
    health_ratio = healthy_services / total_services if total_services > 0 else 0

    critical_alerts = sum(1 for a in result["alerts"] if a["severity"] == "critical")
    warning_alerts = sum(1 for a in result["alerts"] if a["severity"] == "warning")

    if critical_alerts > 0:
        result["overall_status"] = "critical"
    elif warning_alerts > 0 or health_ratio < 0.8:
        result["overall_status"] = "degraded"
    elif health_ratio >= 0.8:
        result["overall_status"] = "healthy"
    else:
        result["overall_status"] = "unknown"

    result["health_summary"] = {
        "healthy_services": healthy_services,
        "total_services": total_services,
        "health_ratio": round(health_ratio, 2),
        "critical_alerts": critical_alerts,
        "warning_alerts": warning_alerts
    }

    return result


# ============================================================================
# ALERTS
# ============================================================================

def get_active_alerts() -> Dict[str, Any]:
    """
    Get current active alerts and issues.

    Returns:
        Dict with active alerts
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "alerts": [],
        "by_severity": {
            "critical": [],
            "warning": [],
            "info": []
        }
    }

    # Get dashboard status for alerts
    dashboard = get_dashboard_status()
    result["alerts"] = dashboard.get("alerts", [])

    # Categorize by severity
    for alert in result["alerts"]:
        severity = alert.get("severity", "info")
        if severity in result["by_severity"]:
            result["by_severity"][severity].append(alert)

    # Add recommendations
    result["recommendations"] = []

    for alert in result["alerts"]:
        if alert.get("severity") == "critical":
            if "not running" in alert.get("message", "").lower():
                result["recommendations"].append({
                    "for_alert": alert["message"],
                    "action": "Start Msty Studio application",
                    "priority": "high"
                })
            elif "memory" in alert.get("message", "").lower():
                result["recommendations"].append({
                    "for_alert": alert["message"],
                    "action": "Close unused applications or unload models",
                    "priority": "high"
                })
            elif "disk" in alert.get("message", "").lower():
                result["recommendations"].append({
                    "for_alert": alert["message"],
                    "action": "Free up disk space by removing unused models or files",
                    "priority": "high"
                })

    result["summary"] = {
        "total_alerts": len(result["alerts"]),
        "critical_count": len(result["by_severity"]["critical"]),
        "warning_count": len(result["by_severity"]["warning"]),
        "requires_action": len(result["by_severity"]["critical"]) > 0
    }

    return result


# ============================================================================
# REAL-TIME METRICS
# ============================================================================

def get_realtime_metrics() -> Dict[str, Any]:
    """
    Get real-time performance metrics.

    Returns:
        Dict with current metrics
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "metrics": {},
    }

    # CPU metrics
    try:
        cpu_times = psutil.cpu_times_percent(interval=0.5)
        result["metrics"]["cpu"] = {
            "user_percent": cpu_times.user,
            "system_percent": cpu_times.system,
            "idle_percent": cpu_times.idle,
            "total_percent": round(100 - cpu_times.idle, 1),
            "core_count": psutil.cpu_count()
        }
    except Exception as e:
        result["metrics"]["cpu"] = {"error": str(e)}

    # Memory metrics
    try:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        result["metrics"]["memory"] = {
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent": memory.percent,
            "swap_used_gb": round(swap.used / (1024**3), 2),
            "swap_percent": swap.percent
        }
    except Exception as e:
        result["metrics"]["memory"] = {"error": str(e)}

    # Disk I/O
    try:
        disk_io = psutil.disk_io_counters()
        if disk_io:
            result["metrics"]["disk_io"] = {
                "read_mb": round(disk_io.read_bytes / (1024**2), 2),
                "write_mb": round(disk_io.write_bytes / (1024**2), 2),
                "read_count": disk_io.read_count,
                "write_count": disk_io.write_count
            }
    except Exception as e:
        result["metrics"]["disk_io"] = {"error": str(e)}

    # Network I/O
    try:
        net_io = psutil.net_io_counters()
        if net_io:
            result["metrics"]["network"] = {
                "bytes_sent_mb": round(net_io.bytes_sent / (1024**2), 2),
                "bytes_recv_mb": round(net_io.bytes_recv / (1024**2), 2),
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
    except Exception as e:
        result["metrics"]["network"] = {"error": str(e)}

    # Service response times
    result["metrics"]["service_latency"] = {}
    services = {
        "local_ai": LOCAL_AI_SERVICE_PORT,
        "mlx": MLX_SERVICE_PORT,
    }
    for name, port in services.items():
        health = check_service_health(SIDECAR_HOST, port, timeout=2)
        if health["response_time_ms"]:
            result["metrics"]["service_latency"][name] = health["response_time_ms"]

    return result


__all__ = [
    "check_service_health",
    "get_dashboard_status",
    "get_active_alerts",
    "get_realtime_metrics",
]
