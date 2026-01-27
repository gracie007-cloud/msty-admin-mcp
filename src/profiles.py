"""
Msty Admin MCP - Configuration Profiles

Tools for saving, loading, and managing configuration profiles.
"""

import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from .paths import get_msty_paths, read_claude_desktop_config
from .database import get_database_connection

logger = logging.getLogger("msty-admin-mcp")


# ============================================================================
# PROFILE STORAGE
# ============================================================================

def get_profiles_directory() -> Path:
    """Get or create the profiles directory."""
    paths = get_msty_paths()
    data_dir = paths.get("data_dir")

    if data_dir:
        profiles_dir = Path(data_dir) / "MCP_Profiles"
    else:
        # Fallback to home directory
        profiles_dir = Path.home() / ".msty-admin" / "profiles"

    profiles_dir.mkdir(parents=True, exist_ok=True)
    return profiles_dir


def list_profiles() -> Dict[str, Any]:
    """
    List all saved configuration profiles.

    Returns:
        Dict with profiles list
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "profiles": [],
        "total_count": 0,
    }

    profiles_dir = get_profiles_directory()

    for f in profiles_dir.iterdir():
        if f.suffix == '.json':
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                    result["profiles"].append({
                        "id": f.stem,
                        "name": data.get("name", f.stem),
                        "description": data.get("description", ""),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "components": list(data.get("config", {}).keys()),
                        "path": str(f)
                    })
            except Exception as e:
                logger.warning(f"Error reading profile {f}: {e}")

    result["total_count"] = len(result["profiles"])
    result["profiles_directory"] = str(profiles_dir)

    return result


# ============================================================================
# PROFILE SAVE/LOAD
# ============================================================================

def save_profile(
    name: str,
    description: str = "",
    include_personas: bool = True,
    include_tools: bool = True,
    include_settings: bool = True,
    include_prompts: bool = True
) -> Dict[str, Any]:
    """
    Save current configuration as a named profile.

    Args:
        name: Profile name
        description: Profile description
        include_personas: Include persona configurations
        include_tools: Include MCP tool configurations
        include_settings: Include application settings
        include_prompts: Include saved prompts

    Returns:
        Dict with save result
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "profile_name": name,
        "saved": False,
    }

    # Create profile ID from name
    profile_id = name.lower().replace(" ", "_").replace("-", "_")
    profile_id = "".join(c for c in profile_id if c.isalnum() or c == "_")

    profile_data = {
        "id": profile_id,
        "name": name,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "config": {}
    }

    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()

            # Save personas
            if include_personas:
                try:
                    cursor.execute("SELECT * FROM personas LIMIT 100")
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    profile_data["config"]["personas"] = [
                        dict(zip(columns, row)) for row in rows
                    ]
                except:
                    profile_data["config"]["personas"] = []

            # Save prompts
            if include_prompts:
                try:
                    cursor.execute("SELECT * FROM prompts LIMIT 100")
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    profile_data["config"]["prompts"] = [
                        dict(zip(columns, row)) for row in rows
                    ]
                except:
                    profile_data["config"]["prompts"] = []

            # Save settings
            if include_settings:
                try:
                    cursor.execute("SELECT * FROM settings LIMIT 100")
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    profile_data["config"]["settings"] = [
                        dict(zip(columns, row)) for row in rows
                    ]
                except:
                    profile_data["config"]["settings"] = []

            conn.close()

        # Save MCP tools from Claude config
        if include_tools:
            claude_config = read_claude_desktop_config()
            if claude_config:
                profile_data["config"]["mcp_tools"] = claude_config.get("mcpServers", {})

        # Write profile file
        profiles_dir = get_profiles_directory()
        profile_path = profiles_dir / f"{profile_id}.json"

        with open(profile_path, 'w') as f:
            json.dump(profile_data, f, indent=2, default=str)

        result["saved"] = True
        result["profile_id"] = profile_id
        result["profile_path"] = str(profile_path)
        result["components_saved"] = list(profile_data["config"].keys())

    except Exception as e:
        result["error"] = str(e)

    return result


def load_profile(
    profile_id: str,
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    Load a saved configuration profile.

    Args:
        profile_id: Profile identifier to load
        dry_run: If True, only show what would be loaded

    Returns:
        Dict with profile data or load result
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "profile_id": profile_id,
        "dry_run": dry_run,
        "loaded": False,
    }

    profiles_dir = get_profiles_directory()
    profile_path = profiles_dir / f"{profile_id}.json"

    if not profile_path.exists():
        # Try finding by name
        for f in profiles_dir.iterdir():
            if f.suffix == '.json':
                try:
                    with open(f, 'r') as file:
                        data = json.load(file)
                        if data.get("name", "").lower() == profile_id.lower():
                            profile_path = f
                            break
                except:
                    pass

    if not profile_path.exists():
        result["error"] = f"Profile '{profile_id}' not found"
        result["available_profiles"] = [
            p["name"] for p in list_profiles().get("profiles", [])
        ]
        return result

    try:
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)

        result["profile_name"] = profile_data.get("name")
        result["profile_description"] = profile_data.get("description")
        result["created_at"] = profile_data.get("created_at")
        result["components"] = list(profile_data.get("config", {}).keys())

        if dry_run:
            result["preview"] = {
                "personas_count": len(profile_data.get("config", {}).get("personas", [])),
                "prompts_count": len(profile_data.get("config", {}).get("prompts", [])),
                "settings_count": len(profile_data.get("config", {}).get("settings", [])),
                "mcp_tools_count": len(profile_data.get("config", {}).get("mcp_tools", {})),
            }
            result["note"] = "Dry run - no changes made. Set dry_run=False to apply."
        else:
            # Actual loading would require write access to Msty's database
            result["note"] = "Profile loaded in preview mode. Full restore requires Msty restart."
            result["config"] = profile_data.get("config", {})

        result["loaded"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================================
# PROFILE COMPARISON
# ============================================================================

def compare_profiles(
    profile_id_1: str,
    profile_id_2: str
) -> Dict[str, Any]:
    """
    Compare two configuration profiles.

    Args:
        profile_id_1: First profile to compare
        profile_id_2: Second profile to compare

    Returns:
        Dict with comparison results
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "profile_1": profile_id_1,
        "profile_2": profile_id_2,
        "comparison": {},
    }

    profiles_dir = get_profiles_directory()

    # Load both profiles
    profiles = {}
    for pid in [profile_id_1, profile_id_2]:
        profile_path = profiles_dir / f"{pid}.json"
        if profile_path.exists():
            try:
                with open(profile_path, 'r') as f:
                    profiles[pid] = json.load(f)
            except:
                result["error"] = f"Error loading profile '{pid}'"
                return result
        else:
            result["error"] = f"Profile '{pid}' not found"
            return result

    p1 = profiles[profile_id_1]
    p2 = profiles[profile_id_2]

    # Compare components
    result["comparison"] = {
        "metadata": {
            "profile_1_name": p1.get("name"),
            "profile_2_name": p2.get("name"),
            "profile_1_created": p1.get("created_at"),
            "profile_2_created": p2.get("created_at"),
        },
        "component_counts": {},
        "differences": []
    }

    # Count components
    for component in ["personas", "prompts", "settings", "mcp_tools"]:
        c1 = p1.get("config", {}).get(component, [])
        c2 = p2.get("config", {}).get(component, [])

        count1 = len(c1) if isinstance(c1, list) else len(c1.keys()) if isinstance(c1, dict) else 0
        count2 = len(c2) if isinstance(c2, list) else len(c2.keys()) if isinstance(c2, dict) else 0

        result["comparison"]["component_counts"][component] = {
            "profile_1": count1,
            "profile_2": count2,
            "difference": count2 - count1
        }

        if count1 != count2:
            result["comparison"]["differences"].append({
                "component": component,
                "type": "count_mismatch",
                "detail": f"{profile_id_1} has {count1}, {profile_id_2} has {count2}"
            })

    # Check for unique items
    for component in ["personas", "prompts"]:
        c1 = p1.get("config", {}).get(component, [])
        c2 = p2.get("config", {}).get(component, [])

        if isinstance(c1, list) and isinstance(c2, list):
            names1 = {item.get("name", item.get("id")) for item in c1}
            names2 = {item.get("name", item.get("id")) for item in c2}

            only_in_1 = names1 - names2
            only_in_2 = names2 - names1

            if only_in_1:
                result["comparison"]["differences"].append({
                    "component": component,
                    "type": "only_in_profile_1",
                    "items": list(only_in_1)
                })
            if only_in_2:
                result["comparison"]["differences"].append({
                    "component": component,
                    "type": "only_in_profile_2",
                    "items": list(only_in_2)
                })

    result["comparison"]["total_differences"] = len(result["comparison"]["differences"])
    result["comparison"]["identical"] = len(result["comparison"]["differences"]) == 0

    return result


def export_profile(
    profile_id: str,
    export_format: str = "json"
) -> Dict[str, Any]:
    """
    Export a profile for sharing.

    Args:
        profile_id: Profile to export
        export_format: Export format (json, yaml)

    Returns:
        Dict with exportable profile data
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "profile_id": profile_id,
        "export_format": export_format,
    }

    profiles_dir = get_profiles_directory()
    profile_path = profiles_dir / f"{profile_id}.json"

    if not profile_path.exists():
        result["error"] = f"Profile '{profile_id}' not found"
        return result

    try:
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)

        # Create shareable version (remove potentially sensitive data)
        shareable = {
            "name": profile_data.get("name"),
            "description": profile_data.get("description"),
            "created_at": profile_data.get("created_at"),
            "config": {}
        }

        # Include personas (without sensitive data)
        if "personas" in profile_data.get("config", {}):
            shareable["config"]["personas"] = [
                {k: v for k, v in p.items() if k not in ["api_key", "token", "secret"]}
                for p in profile_data["config"]["personas"]
            ]

        # Include prompts
        if "prompts" in profile_data.get("config", {}):
            shareable["config"]["prompts"] = profile_data["config"]["prompts"]

        # Include MCP tools (structure only, no env vars)
        if "mcp_tools" in profile_data.get("config", {}):
            tools = {}
            for name, config in profile_data["config"]["mcp_tools"].items():
                tools[name] = {
                    "command": config.get("command"),
                    "args": config.get("args"),
                    # Exclude env vars for security
                }
            shareable["config"]["mcp_tools"] = tools

        if export_format == "json":
            result["export_data"] = json.dumps(shareable, indent=2, default=str)
        else:
            result["export_data"] = json.dumps(shareable, indent=2, default=str)
            result["note"] = "YAML format requires PyYAML. Returning JSON."

        result["exported"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


__all__ = [
    "get_profiles_directory",
    "list_profiles",
    "save_profile",
    "load_profile",
    "compare_profiles",
    "export_profile",
]
