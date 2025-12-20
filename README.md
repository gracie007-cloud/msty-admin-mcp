# 🍍 Msty Admin MCP v4.0.0

**AI-Administered Msty Studio Desktop Management System**

An MCP (Model Context Protocol) server that enables Claude Desktop to act as an intelligent system administrator for Msty Studio Desktop, providing database insights, configuration management, local model orchestration, and tiered AI workflow capabilities.

[![Version](https://img.shields.io/badge/version-4.0.0-blue.svg)](https://github.com/M-Pineapple/msty-admin-mcp)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-yellow.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://apple.com)

## 🎯 Overview

Msty Admin MCP bridges the gap between Claude Desktop and Msty Studio Desktop, enabling:

- **Database Insights**: Query conversations, personas, prompts, and tools directly
- **Health Monitoring**: Comprehensive system health analysis and recommendations
- **Configuration Sync**: Export/import tools between Claude Desktop and Msty
- **Sidecar Integration**: Direct access to local AI models via Ollama-compatible API
- **Intelligence Layer**: Performance metrics, conversation analytics, model comparison
- **Tiered AI Workflow**: Local model calibration with Claude escalation triggers

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
| `export_tool_config` | Export MCP tool configurations for backup or sync |
| `sync_claude_preferences` | Convert Claude Desktop preferences to Msty persona |
| `generate_persona` | Create persona configurations from templates |
| `import_tool_config` | Validate and prepare tools for Msty import |

### Phase 3: Automation Bridge ✅

| Tool | Description |
|------|-------------|
| `get_sidecar_status` | Check Sidecar and Local AI Service status |
| `list_available_models` | List all models via Sidecar's Ollama-compatible API |
| `query_local_ai_service` | Direct API access to Local AI Service |
| `chat_with_local_model` | Send messages to local models with metric tracking |
| `recommend_model` | Get model recommendations based on use case |

### Phase 4: Intelligence Layer ✅ **NEW**

| Tool | Description |
|------|-------------|
| `get_model_performance_metrics` | Aggregated metrics: tokens/sec, latency, error rates |
| `analyse_conversation_patterns` | Privacy-respecting usage analytics from Msty database |
| `compare_model_responses` | Send same prompt to multiple models, compare quality/speed |
| `optimise_knowledge_stacks` | Analyse and recommend knowledge stack improvements |
| `suggest_persona_improvements` | AI-powered suggestions for persona optimization |

### Phase 5: Tiered AI Workflow ✅ **NEW**

| Tool | Description |
|------|-------------|
| `run_calibration_test` | Run quality tests on local models by category |
| `evaluate_response_quality` | Score any response using heuristic evaluation |
| `identify_handoff_triggers` | Track patterns that should escalate to Claude |
| `get_calibration_history` | Historical test results with trends and statistics |

## 📊 Tiered AI Architecture

The goal: Run local models for routine tasks with seamless escalation to Claude when complexity demands it.

```
┌─────────────────────────────────────────────────┐
│                 Task Incoming                    │
└─────────────────────┬───────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌─────────────────┐      ┌─────────────────┐
│   Local Model   │      │   Claude        │
│   (Routine)     │      │   (Complex)     │
│                 │      │                 │
│ • Simple tasks  │      │ • Architecture  │
│ • Quick queries │      │ • Deep analysis │
│ • Status checks │      │ • Creative work │
│ • Data lookup   │      │ • Multi-step    │
└────────┬────────┘      └────────┬────────┘
         │                        │
         │    Handoff Triggers    │
         │    ← ─ ─ ─ ─ ─ ─ ─ ─→  │
         └────────────┬───────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│              Shared MCP Ecosystem                │
│  Memory • Trello • GitHub • Filesystem • More   │
└─────────────────────────────────────────────────┘
```

### Calibration Categories

The calibration system tests models across these categories:

- **Reasoning**: Logic puzzles, mathematical problems
- **Coding**: Code generation, algorithm implementation
- **Writing**: Professional writing, British English
- **Analysis**: Critical thinking, trade-off analysis
- **Creative**: Story writing, creative tasks

### Metrics Database

Performance metrics are stored in `~/.msty-admin/msty_admin_metrics.db`:

- **model_metrics**: Per-request performance tracking
- **calibration_tests**: Test results with quality scores
- **handoff_triggers**: Patterns requiring escalation
- **conversation_analytics**: Aggregated usage statistics

## 📦 Installation

### Prerequisites

- macOS (Apple Silicon or Intel)
- Python 3.10+
- [Msty Studio Desktop](https://msty.ai) installed
- Msty Sidecar running (for Phase 3-5 features)

### Setup

```bash
# Clone the repository
git clone https://github.com/M-Pineapple/msty-admin-mcp.git
cd msty-admin-mcp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

## 🔧 Usage Examples

### Check Msty Installation

```
"What's the status of my Msty installation?"
→ Uses detect_msty_installation
```

### Query Database

```
"Show me my Msty personas"
→ Uses read_msty_database with query_type="personas"
```

### Chat with Local Model

```
"Ask my local Qwen model to explain async/await in Python"
→ Uses chat_with_local_model
```

### Compare Models

```
"Compare all my local models on a coding task"
→ Uses compare_model_responses
```

### Run Calibration

```
"Run calibration tests on my local model for coding tasks"
→ Uses run_calibration_test with category="coding"
```

### Check Performance

```
"Show me performance metrics for my local models"
→ Uses get_model_performance_metrics
```

## 🏗️ Project Structure

```
msty-admin-mcp/
├── src/
│   ├── __init__.py
│   ├── server.py              # Main MCP server (24 tools)
│   ├── phase4_5_tools.py      # Intelligence & Calibration utilities
│   ├── intelligence/          # Phase 4 modules
│   └── calibration/           # Phase 5 modules
├── tests/
│   └── test_server.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 📈 Changelog

### v4.0.0 (2025-12-20)

- **Phase 4: Intelligence Layer** - 5 new tools for analytics and optimization
- **Phase 5: Tiered AI Workflow** - 4 new calibration tools
- Metrics database system for performance tracking
- Heuristic response quality evaluation
- Handoff trigger identification and tracking
- Total tools: 24

### v3.0.1 (2025-12-19)

- Fixed compatibility with qwen3:0.6b thinking model
- Improved response handling for reasoning-only outputs
- Enhanced error handling in chat_with_local_model

### v3.0.0 (2025-12-19)

- Phase 3: Automation Bridge complete
- Sidecar API integration via Ollama-compatible endpoints
- Direct local model chat capability
- Model recommendations based on use case

### v2.0.0 (2025-12-18)

- Phase 2: Configuration Management complete
- Tool export/import between Claude Desktop and Msty
- Persona generation with templates

### v1.0.0 (2025-12-17)

- Initial release
- Phase 1: Foundational Tools

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines before submitting PRs.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🍍 Author

Created by **Pineapple** 🍍 AI Administration System

---

*"Making local AI models work smarter, not harder."*
