"""
Msty Admin MCP - Path Resolution Utilities

Functions for finding Msty Studio paths on macOS and Windows.
"""

import os
import sys
from pathlib import Path
from typing import Optional


def _get_windows_msty_paths() -> dict:
    """Detect Msty Studio paths on Windows (Msty 1.x and 2.x).

    Msty 1.x stores data in %APPDATA%\\Msty\\  with SharedStorage as the
    Electron/SQLite database file (same name convention as macOS builds).
    """
    home = Path.home()
    appdata = os.environ.get("APPDATA", str(home / "AppData" / "Roaming"))
    localappdata = os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local"))

    # Candidate data directories — checked in priority order
    win_data_candidates = [
        Path(appdata) / "Msty",             # Msty 1.x (confirmed)
        Path(appdata) / "MstyStudio",       # possible future name
        Path(appdata) / "Msty Studio",
        Path(localappdata) / "Msty",
        Path(localappdata) / "MstyStudio",
        Path(localappdata) / "Msty Studio",
    ]

    # Common executable locations
    win_app_candidates = [
        Path(localappdata) / "Programs" / "Msty" / "Msty.exe",
        Path(localappdata) / "Programs" / "MstyStudio" / "MstyStudio.exe",
        Path("C:/Program Files/Msty/Msty.exe"),
        Path("C:/Program Files/MstyStudio/MstyStudio.exe"),
    ]

    resolved: dict = {"app": None, "app_alt": None, "data": None,
                       "sidecar": None, "database": None, "database_type": None,
                       "mlx_models": None, "platform": "windows"}

    for cand in win_app_candidates:
        if cand.exists():
            resolved["app"] = str(cand)
            break

    for cand in win_data_candidates:
        if cand.exists():
            resolved["data"] = str(cand)
            break

    # DB detection inside the data dir — ordered by specificity (msty.db first,
    # SharedStorage last to avoid picking Electron's session/renderer cache)
    db_patterns = [
        "msty.db",            # Msty 1.9.x confirmed Windows app database
        "msty.sqlite",
        "msty.sqlite3",
        "data.db",
        "app.db",
        "SharedStorage",      # Fallback: older Msty builds used this name
    ]
    if resolved["data"]:
        data_path = Path(resolved["data"])
        for pattern in db_patterns:
            db_file = data_path / pattern
            if db_file.exists() and db_file.is_file():
                resolved["database"] = str(db_file)
                resolved["database_type"] = "windows_direct"
                break

    return resolved


def get_msty_paths() -> dict:
    """Get all relevant Msty Studio paths (cross-platform: Windows + macOS)."""
    home = Path.home()

    resolved: dict = {}

    # ── 1. Environment variable override (highest priority on all platforms) ──
    env_db_path = os.environ.get("MSTY_DATABASE_PATH")
    if env_db_path and Path(env_db_path).exists():
        resolved["database"] = env_db_path
        resolved["database_type"] = "env_override"
    else:
        resolved["database"] = None
        resolved["database_type"] = None

    # ── 2. Windows paths ──────────────────────────────────────────────────────
    if sys.platform == "win32":
        win = _get_windows_msty_paths()
        resolved.update(win)
        # env override still wins
        if env_db_path and Path(env_db_path).exists():
            resolved["database"] = env_db_path
            resolved["database_type"] = "env_override"

        # MLX models don't exist on Windows — keep None
        resolved["mlx_models"] = None
        return resolved

    # ── 3. macOS paths ────────────────────────────────────────────────────────
    mac_paths = {
        "app": Path("/Applications/MstyStudio.app"),
        "app_alt": Path("/Applications/Msty Studio.app"),
        "data": home / "Library/Application Support/MstyStudio",
        "sidecar": home / "Library/Application Support/MstySidecar",
    }
    for key, path in mac_paths.items():
        resolved[key] = str(path) if path.exists() else None

    # Check old Sidecar path
    if not resolved["database"] and resolved.get("sidecar"):
        sidecar_db = Path(resolved["sidecar"]) / "SharedStorage"
        if sidecar_db.exists():
            resolved["database"] = str(sidecar_db)
            resolved["database_type"] = "sidecar_shared"

    # Check for SharedStorage in main data directory (Msty 2.4.0+)
    if not resolved["database"] and resolved.get("data"):
        shared_storage = Path(resolved["data"]) / "SharedStorage"
        if shared_storage.exists() and shared_storage.is_file():
            resolved["database"] = str(shared_storage)
            resolved["database_type"] = "shared_storage"

    # Search for database files in data directory (Msty 2.4.0+)
    if not resolved["database"] and resolved.get("data"):
        data_path = Path(resolved["data"])
        db_patterns = [
            "SharedStorage",
            "msty.db",
            "msty.sqlite",
            "msty.sqlite3",
            "data.db",
            "app.db",
            "storage.db",
            "MstyStudio.db",
            "*.db",
            "**/*.db",
            "databases/*.db",
            "db/*.db",
            "storage/*.db",
        ]
        for pattern in db_patterns:
            if "*" in pattern:
                matches = list(data_path.glob(pattern))
                if matches:
                    matches.sort(key=lambda p: p.stat().st_size, reverse=True)
                    resolved["database"] = str(matches[0])
                    resolved["database_type"] = f"glob:{pattern}"
                    resolved["all_databases"] = [str(m) for m in matches[:5]]
                    break
            else:
                db_file = data_path / pattern
                if db_file.exists():
                    resolved["database"] = str(db_file)
                    resolved["database_type"] = "direct"
                    break

    # Check sandboxed app containers
    if not resolved["database"]:
        containers_path = home / "Library/Containers/ai.msty.MstyStudio/Data/Library/Application Support"
        if containers_path.exists():
            for db_file in containers_path.glob("**/*.db"):
                resolved["database"] = str(db_file)
                resolved["database_type"] = "container"
                break

    mlx_path = Path(resolved["data"]) / "models-mlx" if resolved.get("data") else None
    resolved["mlx_models"] = str(mlx_path) if mlx_path and mlx_path.exists() else None

    return resolved


def sanitize_path(path: str) -> str:
    """Replace home directory with $HOME for portability"""
    home = str(Path.home())
    if path and path.startswith(home):
        return path.replace(home, "$HOME", 1)
    return path


def expand_path(path: str) -> str:
    """Expand $HOME and ~ in paths"""
    if path:
        path = path.replace("$HOME", str(Path.home()))
        path = os.path.expanduser(path)
    return path


def read_claude_desktop_config() -> dict:
    """Read Claude Desktop's MCP configuration"""
    import json
    config_path = Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    if not config_path.exists():
        return {"error": "Claude Desktop config not found"}
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


__all__ = [
    "get_msty_paths",
    "sanitize_path",
    "expand_path",
    "read_claude_desktop_config"
]
