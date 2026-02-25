"""Fast probe: find Msty Studio 1.9.2 data directory on Windows."""
import os, sys
from pathlib import Path

appdata = os.environ.get("APPDATA", "")
localappdata = os.environ.get("LOCALAPPDATA", "")
home = Path.home()

candidates = [
    # Msty Studio Desktop (Tauri-based) — typical Windows paths
    Path(appdata) / "com.msty.studio",
    Path(appdata) / "MstyStudio",
    Path(appdata) / "Msty Studio",
    Path(appdata) / "msty",
    Path(appdata) / "msty-studio",
    Path(localappdata) / "com.msty.studio",
    Path(localappdata) / "MstyStudio",
    Path(localappdata) / "Msty Studio",
    Path(localappdata) / "msty",
    Path(localappdata) / "msty-studio",
    # Tauri stores data under Roaming\<bundle_id>
    Path(appdata) / "ai.msty.studio",
    Path(appdata) / "ai.msty.MstyStudio",
    # Some electron apps use %USERPROFILE%\AppData\Roaming
    home / "AppData" / "Roaming" / "MstyStudio",
    home / "AppData" / "Roaming" / "Msty Studio",
    home / "AppData" / "Local" / "MstyStudio",
    home / "AppData" / "Local" / "Msty Studio",
]

print("Checking candidate paths:")
for c in candidates:
    exists = c.exists()
    print(f"  {'FOUND' if exists else '    -'} {c}")
    if exists:
        print(f"         Contents: {[x.name for x in list(c.iterdir())[:10]]}")

