# Msty Admin MCP - Development Guide

Guide for contributing to Msty Admin MCP.

## Table of Contents

- [Getting Started](#getting-started)
- [Project Architecture](#project-architecture)
- [Running Tests](#running-tests)
- [Adding New Tools](#adding-new-tools)
- [Code Style](#code-style)
- [Release Process](#release-process)

---

## Getting Started

### Prerequisites

- Python 3.10+
- macOS (for full testing with Msty Studio)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/DBSS/msty-admin-mcp.git
cd msty-admin-mcp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest pytest-asyncio
```

### Running the Server

```bash
# From the project root
python -m src.server

# Or use the shell script (recommended for Claude Desktop)
./run_msty_server.sh
```

---

## Project Architecture

### Module Overview

```
src/
├── __init__.py         # Package exports (lazy load for server)
├── constants.py        # Configuration constants (single source of truth)
├── models.py           # Data classes (no external deps)
├── errors.py           # Standardized error handling
├── paths.py            # macOS path resolution
├── database.py         # SQL operations with injection protection
├── network.py          # HTTP requests, process detection
├── cache.py            # TTL-based response caching
├── tagging.py          # Model tagging system (60+ overrides)
├── server.py           # Main MCP server (42 tools)
└── phase4_5_tools.py   # Metrics DB and calibration utilities
```

### Module Dependencies

```
constants.py ──┐
               ├──> models.py (no deps)
errors.py ─────┤
               ├──> paths.py (uses constants)
               ├──> database.py (uses constants)
               ├──> network.py (uses constants)
               ├──> cache.py (no deps)
               ├──> tagging.py (no deps)
               │
               └──> server.py (uses all modules + MCP)
```

**Key Design Principles:**

1. **No Circular Imports**: Modules only import from "lower" modules
2. **Lazy Server Import**: `__init__.py` doesn't import `server.py` directly (requires MCP package)
3. **Single Source of Truth**: All configuration in `constants.py`
4. **Testable Utilities**: Utility modules have no MCP dependency

---

## Running Tests

### Full Test Suite

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Individual Test Files

```bash
# Test specific module
pytest tests/test_database.py -v
pytest tests/test_tagging.py -v
pytest tests/test_cache.py -v
```

### Test Categories

| File | Tests | Description |
|------|-------|-------------|
| `test_constants.py` | 13 | Version format, ports, table allowlist |
| `test_paths.py` | 14 | Path resolution, sanitization |
| `test_database.py` | 14 | SQL injection protection, connections |
| `test_network.py` | 15 | API requests, process detection |
| `test_cache.py` | 15 | TTL expiry, cache operations |
| `test_tagging.py` | 30 | Model tags, patterns, overrides |
| `test_server.py` | 8 | Integration tests (requires MCP) |

**Total: 109 tests**

### Writing New Tests

```python
# tests/test_example.py
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.module_name import function_to_test


class TestFunctionName:
    """Tests for function_to_test"""

    def test_basic_functionality(self):
        """Verify basic behavior"""
        result = function_to_test("input")
        assert result == expected_output

    def test_edge_case(self):
        """Verify edge case handling"""
        result = function_to_test("")
        assert result is None

    def test_error_handling(self):
        """Verify errors are handled gracefully"""
        result = function_to_test(None)
        assert "error" in result
```

---

## Adding New Tools

### Step 1: Choose the Right Phase

| Phase | Purpose | Examples |
|-------|---------|----------|
| 1 | Read-only inspection | `detect_msty_installation`, `read_msty_database` |
| 2 | Configuration management | `export_tool_config`, `generate_persona` |
| 3 | Model interaction | `chat_with_local_model`, `list_available_models` |
| 4 | Analytics | `get_model_performance_metrics`, `compare_model_responses` |
| 5 | Calibration | `run_calibration_test`, `evaluate_response_quality` |
| 6 | Model management | `benchmark_model`, `get_model_details` |
| 7 | Conversations | `export_conversations`, `search_conversations` |
| 8 | Templates | `create_prompt_template`, `run_prompt_template` |
| 9 | System | `backup_configuration`, `get_system_resources` |

### Step 2: Add the Tool Function

```python
# In src/server.py

@mcp.tool()
def my_new_tool(
    required_param: str,
    optional_param: int = 10
) -> str:
    """
    Brief description of what this tool does.

    Args:
        required_param: Description of this parameter
        optional_param: Description with default value

    Returns:
        JSON string with the results
    """
    try:
        # Your implementation here
        result = do_something(required_param, optional_param)

        return success_response({
            "data": result,
            "count": len(result)
        })

    except SomeError as e:
        return error_response(
            ErrorCode.APPROPRIATE_CODE,
            f"Human-readable message: {e}",
            suggestion="How to fix this"
        )
```

### Step 3: Use Standardized Responses

```python
from .errors import error_response, success_response, ErrorCode

# Success
return success_response({
    "key": "value",
    "count": 42
}, message="Optional message")

# Error
return error_response(
    ErrorCode.DATABASE_NOT_FOUND,
    "Msty database not found",
    suggestion="Run detect_msty_installation first"
)
```

### Step 4: Update Documentation

1. Add tool to `README.md` in appropriate phase table
2. Add detailed docs to `docs/API.md` if complex
3. Update `CHANGELOG.md`

### Step 5: Add Tests

```python
# tests/test_server.py

def test_my_new_tool_returns_json(self):
    """Verify my_new_tool returns valid JSON"""
    from src.server import my_new_tool
    result = my_new_tool("test_input")

    data = json.loads(result)
    assert isinstance(data, dict)
    assert "data" in data
```

---

## Code Style

### Python Style

- Follow PEP 8
- Use type hints for function signatures
- Docstrings for all public functions
- Maximum line length: 100 characters

### Imports

```python
# Standard library
import json
import os
from pathlib import Path
from typing import Optional, Dict, List

# Third-party
import psutil
from mcp.server.fastmcp import FastMCP

# Local modules
from .constants import SERVER_VERSION, SIDECAR_HOST
from .errors import error_response, success_response, ErrorCode
from .database import query_database, is_safe_table_name
```

### Error Handling

```python
# Good: Specific exceptions
try:
    conn = sqlite3.connect(db_path)
except sqlite3.Error as e:
    logger.error(f"Database error: {e}")
    return error_response(ErrorCode.DATABASE_CONNECTION_ERROR, str(e))

# Bad: Bare except
try:
    conn = sqlite3.connect(db_path)
except:  # Never do this!
    pass
```

### Logging

```python
import logging
logger = logging.getLogger("msty-admin-mcp")

# Levels
logger.debug("Detailed diagnostic info")
logger.info("General operation info")
logger.warning("Something unexpected but handled")
logger.error("Operation failed")
```

---

## Release Process

### Version Numbering

Follow semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes

### Release Checklist

1. **Update Version**
   ```python
   # src/constants.py
   SERVER_VERSION = "6.6.0"
   ```

2. **Update CHANGELOG.md**
   ```markdown
   ## [6.6.0] - 2026-01-27

   ### Added
   - New feature description

   ### Changed
   - Modified behavior

   ### Fixed
   - Bug fix description
   ```

3. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

4. **Commit and Tag**
   ```bash
   git add .
   git commit -m "v6.6.0: Brief description

   Detailed changes...

   Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

   git tag -a v6.6.0 -m "Version 6.6.0"
   git push origin main --tags
   ```

5. **Update README badges** if needed

---

## Debugging

### Enable Debug Logging

```bash
export MSTY_DEBUG=1
python -m src.server
```

### Common Issues

**MCP Import Error:**
```
ModuleNotFoundError: No module named 'mcp'
```
Solution: Install with `pip install mcp`

**Database Not Found:**
```
DATABASE_NOT_FOUND: Msty database not found
```
Solution: Run `detect_msty_installation` to check paths, or set `MSTY_DATABASE_PATH`

**Service Unavailable:**
```
SERVICE_UNAVAILABLE: Local AI service not running
```
Solution: Open Msty Studio and enable Local AI service

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/DBSS/msty-admin-mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DBSS/msty-admin-mcp/discussions)
- **MCP Protocol**: [modelcontextprotocol.io](https://modelcontextprotocol.io)
