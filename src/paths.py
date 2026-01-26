"""
Msty Admin MCP - Path Resolution Utilities

Functions for finding Msty Studio paths on macOS.
"""

import os
from pathlib import Path
from typing import Optional


def get_msty_paths() -> dict:
    """Get all relevant Msty Studio paths for macOS"""
    home = Path.home()

    paths = {
        "app": Path("/Applications/MstyStudio.app"),
        "app_alt": Path("/Applications/Msty Studio.app"),
        "data": home / "Library/Application Support/MstyStudio",
        "sidecar": home / "Library/Application Support/MstySidecar",
    }

    resolved = {}
    for key, path in paths.items():
        resolved[key] = str(path) if path.exists() else None

    # Enhanced database detection for Msty 2.4.0+
    resolved["database"] = None
    resolved["database_type"] = None

    # Check environment variable override first
    env_db_path = os.environ.get("MSTY_DATABASE_PATH")
    if env_db_path and Path(env_db_path).exists():
        resolved["database"] = env_db_path
        resolved["database_type"] = "env_override"

    # Check old Sidecar path
    if not resolved["database"] and resolved["sidecar"]:
        sidecar_db = Path(resolved["sidecar"]) / "SharedStorage"
        if sidecar_db.exists():
            resolved["database"] = str(sidecar_db)
            resolved["database_type"] = "sidecar_shared"

    # Check for SharedStorage in main data directory (Msty 2.4.0+)
    # This is a SQLite file without .db extension
    if not resolved["database"] and resolved["data"]:
        shared_storage = Path(resolved["data"]) / "SharedStorage"
        if shared_storage.exists() and shared_storage.is_file():
            resolved["database"] = str(shared_storage)
            resolved["database_type"] = "shared_storage"

    # Search for database files in data directory (Msty 2.4.0+)
    if not resolved["database"] and resolved["data"]:
        data_path = Path(resolved["data"])

        # Common database file patterns for Msty 2.4.0+
        db_patterns = [
            "SharedStorage",  # Msty 2.4.0+ main database (no extension)
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
                # Glob pattern
                matches = list(data_path.glob(pattern))
                if matches:
                    # Prefer larger files (more likely to be the main database)
                    matches.sort(key=lambda p: p.stat().st_size, reverse=True)
                    resolved["database"] = str(matches[0])
                    resolved["database_type"] = f"glob:{pattern}"
                    resolved["all_databases"] = [str(m) for m in matches[:5]]
                    break
            else:
                # Exact path
                db_file = data_path / pattern
                if db_file.exists():
                    resolved["database"] = str(db_file)
                    resolved["database_type"] = "direct"
                    break

    # Also check for SQLite files in Containers (sandboxed apps)
    if not resolved["database"]:
        containers_path = home / "Library/Containers/ai.msty.MstyStudio/Data/Library/Application Support"
        if containers_path.exists():
            for db_file in containers_path.glob("**/*.db"):
                resolved["database"] = str(db_file)
                resolved["database_type"] = "container"
                break

    if resolved["data"]:
        mlx_path = Path(resolved["data"]) / "models-mlx"
        resolved["mlx_models"] = str(mlx_path) if mlx_path.exists() else None
    else:
        resolved["mlx_models"] = None

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
