#!/bin/bash
# Msty Admin MCP Server Launcher
# Automatically detects script directory for portability

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
source .venv/bin/activate
exec python -m src.server
