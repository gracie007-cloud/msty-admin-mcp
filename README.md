# 🍍 Msty Admin MCP

**AI-Administered Msty Studio Desktop Management System**

An MCP (Model Context Protocol) server that enables Claude Desktop to act as an intelligent system administrator for Msty Studio Desktop, providing database insights, configuration management, local AI orchestration, and seamless sync capabilities.

[![Version](https://img.shields.io/badge/version-3.0.1-blue.svg)](https://github.com/M-Pineapple/msty-admin-mcp)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-yellow.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://apple.com)

## 🎯 Overview

Msty Admin MCP bridges the gap between Claude Desktop and Msty Studio Desktop, enabling:

- **Database Insights**: Query conversations, personas, prompts, and tools directly
- **Health Monitoring**: Comprehensive system health analysis and recommendations
- **Configuration Sync**: Export/import tools between Claude Desktop and Msty
- **Local AI Orchestration**: Chat with local models via Sidecar API
- **Hardware Optimization**: Apple Silicon-optimized model recommendations

## 🚀 Features

### Phase 1: Foundational Tools ✅

| Tool | Description |
|------|-------------|
| `detect_msty_installation` | Find Msty Studio, verify version, locate data paths |
| `read_msty_database` | Query SQLite for conversations, personas, prompts, tools |
| `list_configured_tools` | Read MCP toolbox configuration |
| `get_model_providers` | List configured AI providers and local models |
| `analyse_msty_health` | Check database integrity, storage, model cache status |
| `get_server_status` | Server info and available capabilities |

### Phase 2: Configuration Management ✅

| Tool | Description |
|------|-------------|
| `export_tool_config` | Export MCP tool configurations for backup/sync |
| `sync_claude_preferences` | Convert Claude Desktop preferences to Msty persona |
| `generate_persona` | Create persona configurations from templates |
| `import_tool_config` | Import and validate tool configurations |

### Phase 3: Automation Bridge ✅

| Tool | Description |
|------|-------------|
| `get_sidecar_status` | Comprehensive Sidecar and Local AI Service health check |
| `list_available_models` | Query models available via Sidecar API |
| `query_local_ai_service` | Low-level access to Ollama-compatible API |
| `chat_with_local_model` | Send messages to local models (supports thinking models) |
| `recommend_model` | Hardware-aware model recommendations by use case |

**API Integration:**
- Local AI Service: `http://127.0.0.1:11964` (Ollama-compatible)
- Sidecar Proxy: `http://127.0.0.1:11932`

### Phase 4: Intelligence Layer (Planning)

| Tool | Description |
|------|-------------|
| `analyse_conversation_patterns` | Usage insights from chat history |
| `get_model_performance_metrics` | Track tokens/sec, latency per model |
| `optimise_knowledge_stacks` | Performance recommendations |
| `suggest_persona_improvements` | AI-powered persona optimization |
| `compare_model_responses` | Side-by-side quality comparison |

### Phase 5: Tiered AI Workflow (Future)

The ultimate goal: run local MLX models that perform at Claude Opus level for routine tasks, with seamless escalation to Claude when complexity demands it.

```
┌─────────────────────────────────────────────────┐
│                 Task Incoming                    │
└─────────────────────┬───────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌─────────────────┐      ┌─────────────────┐
│   Local MLX     │      │   Claude Opus   │
│   (Routine)     │      │   (Complex)     │
│                 │      │                 │
│ • File ops      │      │ • Architecture  │
│ • Simple code   │      │ • Deep analysis │
│ • Status checks │      │ • Creative work │
│ • Data queries  │      │ • Multi-step    │
└────────┬────────┘      └────────┬────────┘
         │                        │
         └────────────┬───────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│              Shared MCP Ecosystem                │
│  Memory • Trello • GitHub • Filesystem • More   │
└─────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- macOS (Apple Silicon recommended)
- Python 3.10+
- Msty Studio Desktop installed
- Claude Desktop with MCP support

### Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/M-Pineapple/msty-admin-mcp.git
cd msty-admin-mcp

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Claude Desktop Configuration

Add to your `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "msty-admin": {
      "command": "/path/to/msty-admin-mcp/.venv/bin/python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/msty-admin-mcp"
    }
  }
}
```

Or using uvx:

```json
{
  "mcpServers": {
    "msty-admin": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/M-Pineapple/msty-admin-mcp", "msty-admin"]
    }
  }
}
```

## 🔧 Usage

Once configured, ask Claude to use the Msty Admin tools:

### Check Installation

```
"Check if Msty Studio is installed and get its status"
```

### Database Insights

```
"Show me statistics about my Msty database"
"List all my configured personas in Msty"
"What MCP tools do I have in Msty's Toolbox?"
```

### Health Check

```
"Run a health check on my Msty installation"
"Is Sidecar running? What models are available?"
```

### Chat with Local Models

```
"Send 'Hello world' to qwen3:0.6b via Msty"
"What local models are available for coding tasks?"
"Recommend a model for fast responses on my hardware"
```

### Configuration Sync

```
"Export my Claude Desktop MCP tools for Msty"
"Generate an Opus-style persona for Msty"
```

## 📁 Project Structure

```
msty-admin-mcp/
├── src/
│   ├── __init__.py          # Package initialization
│   └── server.py            # Main FastMCP server (all phases)
├── tests/
│   └── test_server.py       # Unit tests
├── pyproject.toml           # Modern Python packaging
├── requirements.txt         # Dependencies
├── LICENSE                  # MIT License
└── README.md               # This file
```

## 🗂️ Msty Studio Paths

The MCP server automatically detects these locations:

| Component | macOS Path |
|-----------|------------|
| Application | `/Applications/MstyStudio.app` |
| Data Directory | `~/Library/Application Support/MstyStudio/` |
| Sidecar | `~/Library/Application Support/MstySidecar/` |
| Database | `~/Library/Application Support/MstySidecar/SharedStorage` |
| MLX Models | `~/Library/Application Support/MstyStudio/models-mlx/` |

## 🔒 Security

- **Read-Only Database Access**: Database queries use read-only connections
- **API Keys Redacted**: Provider queries automatically redact sensitive credentials
- **Local Only**: All operations happen on your local machine
- **No Telemetry**: Zero data collection or external communication

## 🐛 Troubleshooting

### "Msty database not found"

Ensure Msty Studio Desktop has been run at least once to create the database.

### "Sidecar not running"

Start Sidecar from Terminal:

```bash
open -a MstySidecar
```

### "Local AI Service not responding"

Check that Sidecar is running and models are loaded in Msty Studio.

### Empty responses from thinking models (qwen3)

Fixed in v3.0.1 - the tool now correctly handles models that return reasoning in a separate field.

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🍍 Credits

Created by **Pineapple 🍍** as part of the AI-Administered Local System project.

Built for seamless integration between Claude Desktop and Msty Studio Desktop.

---

**Current Version**: 3.0.1 (Phase 3 Complete)  
**Tools Available**: 15  
**Last Updated**: 19 December 2025  
**Platform**: macOS (Apple Silicon optimized)
