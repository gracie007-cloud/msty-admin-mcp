"""
Msty Admin MCP - Network Utilities

API request helpers and service availability checks.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

import psutil

from .constants import (
    SIDECAR_HOST,
    SIDECAR_TIMEOUT,
    LOCAL_AI_SERVICE_PORT,
    MLX_SERVICE_PORT,
    LLAMACPP_SERVICE_PORT,
    VIBE_PROXY_PORT
)

logger = logging.getLogger("msty-admin-mcp")


def make_api_request(
    endpoint: str,
    port: int = LOCAL_AI_SERVICE_PORT,
    method: str = "GET",
    data: Optional[Dict] = None,
    timeout: int = SIDECAR_TIMEOUT,
    host: str = None
) -> Dict[str, Any]:
    """Make HTTP request to Sidecar or Local AI Service API"""
    host = host or SIDECAR_HOST
    url = f"http://{host}:{port}{endpoint}"

    try:
        if method == "GET":
            req = urllib.request.Request(url)
        else:
            json_data = json.dumps(data).encode('utf-8') if data else None
            req = urllib.request.Request(url, data=json_data, method=method)
            req.add_header('Content-Type', 'application/json')

        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_data = response.read().decode('utf-8')
            return {
                "success": True,
                "status_code": response.status,
                "data": json.loads(response_data) if response_data else None
            }
    except urllib.error.URLError as e:
        logger.warning(f"Connection failed to {url}: {e.reason}")
        return {"success": False, "error": f"Connection failed: {e.reason}"}
    except urllib.error.HTTPError as e:
        # Capture response body for better debugging
        try:
            error_body = e.read().decode('utf-8', errors='ignore')[:200]
            logger.warning(f"HTTP {e.code} on {endpoint}: {error_body}")
        except (OSError, UnicodeDecodeError):
            error_body = None
        return {
            "success": False,
            "error": f"HTTP {e.code}: {e.reason}",
            "status_code": e.code,
            "error_body": error_body
        }
    except json.JSONDecodeError:
        return {"success": True, "status_code": 200, "data": response_data}
    except Exception as e:
        logger.error(f"Unexpected error calling {url}: {e}")
        return {"success": False, "error": str(e)}


def is_process_running(process_name: str) -> bool:
    """Check if a process is running by name"""
    for proc in psutil.process_iter(['name']):
        try:
            if process_name.lower() in proc.info['name'].lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def is_local_ai_available(port: int = None, timeout: int = 2) -> bool:
    """
    Check if Local AI Service is available by attempting to connect.
    Works with Msty 2.4.0+ where services are built into main app.
    """
    port = port or LOCAL_AI_SERVICE_PORT
    try:
        url = f"http://{SIDECAR_HOST}:{port}/v1/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        return False


def get_available_service_ports() -> dict:
    """
    Check which Msty AI services are available (Msty 2.4.0+).
    Returns dict with service status and ports.
    """
    services = {
        "local_ai": {"port": LOCAL_AI_SERVICE_PORT, "available": False},
        "mlx": {"port": MLX_SERVICE_PORT, "available": False},
        "llamacpp": {"port": LLAMACPP_SERVICE_PORT, "available": False},
        "vibe_proxy": {"port": VIBE_PROXY_PORT, "available": False},
    }

    for name, info in services.items():
        services[name]["available"] = is_local_ai_available(info["port"])

    return services


__all__ = [
    "make_api_request",
    "is_process_running",
    "is_local_ai_available",
    "get_available_service_ports"
]
