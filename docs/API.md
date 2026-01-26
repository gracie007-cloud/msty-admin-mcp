# Msty Admin MCP - API Reference

Complete API reference for Msty Admin MCP v6.5.0.

## Table of Contents

- [Response Formats](#response-formats)
- [Error Codes](#error-codes)
- [Tool Reference](#tool-reference)
- [Security](#security)

---

## Response Formats

### Success Response

All successful tool responses follow this JSON structure:

```json
{
  "success": true,
  "data": { ... },
  "message": "Operation completed successfully",
  "timestamp": "2026-01-26T18:30:00Z"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Always `true` for successful responses |
| `data` | object | Tool-specific response data |
| `message` | string | Optional human-readable message |
| `timestamp` | string | ISO 8601 timestamp |

### Error Response

All error responses follow this standardized format:

```json
{
  "success": false,
  "error": {
    "code": "DATABASE_NOT_FOUND",
    "message": "Msty database not found at expected location",
    "suggestion": "Run detect_msty_installation first to verify paths"
  },
  "timestamp": "2026-01-26T18:30:00Z"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Always `false` for errors |
| `error.code` | string | Machine-readable error code |
| `error.message` | string | Human-readable error description |
| `error.suggestion` | string | Optional remediation suggestion |
| `timestamp` | string | ISO 8601 timestamp |

---

## Error Codes

### Database Errors

| Code | Description | Common Causes |
|------|-------------|---------------|
| `DATABASE_NOT_FOUND` | Msty database not detected | Msty not installed, wrong path, SharedStorage file missing |
| `DATABASE_CONNECTION_ERROR` | Failed to connect to database | File locked, permissions, corruption |
| `QUERY_ERROR` | Database query failed | Invalid SQL, table missing |
| `INVALID_TABLE` | Table name not in allowlist | SQL injection attempt, typo in table name |

### Service Errors

| Code | Description | Common Causes |
|------|-------------|---------------|
| `SERVICE_UNAVAILABLE` | Local AI service not running | Msty Studio not open, service disabled |
| `API_ERROR` | External API call failed | Network issue, timeout, service error |
| `TIMEOUT_ERROR` | Request timed out | Service overloaded, network issue |

### Parameter Errors

| Code | Description | Common Causes |
|------|-------------|---------------|
| `INVALID_PARAMETER` | Invalid function parameter | Wrong type, out of range, unsupported value |
| `MISSING_PARAMETER` | Required parameter missing | Parameter not provided |
| `VALIDATION_ERROR` | Input validation failed | Format error, constraint violation |

### File Errors

| Code | Description | Common Causes |
|------|-------------|---------------|
| `FILE_NOT_FOUND` | File does not exist | Wrong path, file deleted |
| `PERMISSION_ERROR` | Insufficient permissions | Read/write access denied |
| `IO_ERROR` | File I/O operation failed | Disk full, file locked |

### Model Errors

| Code | Description | Common Causes |
|------|-------------|---------------|
| `MODEL_NOT_FOUND` | Specified model not available | Model not loaded, wrong ID |
| `MODEL_ERROR` | Model inference failed | Out of memory, invalid input |

---

## Tool Reference

### Phase 1: Installation & Health

#### `detect_msty_installation()`

Detect and analyze Msty Studio Desktop installation.

**Parameters:** None

**Returns:**
```json
{
  "installed": true,
  "version": "2.4.1",
  "is_running": true,
  "paths": {
    "app": "/Applications/MstyStudio.app",
    "data": "/Users/name/Library/Application Support/MstyStudio",
    "database": "/Users/name/Library/Application Support/MstyStudio/SharedStorage"
  },
  "platform_info": {
    "system": "Darwin",
    "machine": "arm64",
    "is_apple_silicon": true
  }
}
```

#### `read_msty_database(query_type, table_name?, limit?)`

Query the Msty Studio database.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query_type` | string | Yes | - | One of: `stats`, `tables`, `conversations`, `personas`, `prompts`, `tools`, `custom` |
| `table_name` | string | No | - | Table name for `custom` query type (must be in allowlist) |
| `limit` | int | No | 50 | Maximum rows to return |

**Returns:** Query-specific data

#### `analyse_msty_health()`

Perform comprehensive health analysis.

**Returns:**
```json
{
  "overall_status": "healthy",
  "database_status": {
    "exists": true,
    "size_mb": 45.2,
    "integrity": "ok"
  },
  "service_status": {
    "local_ai": { "available": true, "port": 11964 },
    "mlx": { "available": true, "port": 11973 },
    "llamacpp": { "available": false, "port": 11454 },
    "vibe_proxy": { "available": true, "port": 8317 }
  },
  "recommendations": []
}
```

---

### Phase 3: Local Model Integration

#### `chat_with_local_model(message, model?, system_prompt?, temperature?, max_tokens?)`

Send a chat message to a local model.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `message` | string | Yes | - | User message to send |
| `model` | string | No | auto | Model ID (auto-selects first available) |
| `system_prompt` | string | No | - | Optional system prompt |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-1.0) |
| `max_tokens` | int | No | 2048 | Maximum tokens in response |

**Returns:**
```json
{
  "response": "Here's my answer...",
  "model": "llama-3.3-70b-instruct",
  "tokens": {
    "prompt": 45,
    "completion": 128,
    "total": 173
  },
  "timing": {
    "total_ms": 3420,
    "tokens_per_second": 37.4
  }
}
```

#### `find_model_by_tag(tag, prefer_fast?, exclude_embedding?)`

Find models matching a specific tag.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `tag` | string | Yes | - | Tag to search for |
| `prefer_fast` | bool | No | false | Sort smaller/faster models first |
| `exclude_embedding` | bool | No | true | Exclude embedding models |

**Available Tags:** `fast`, `quality`, `coding`, `creative`, `reasoning`, `embedding`, `vision`, `long_context`, `large`, `medium`, `small`, `general`

---

### Phase 6: Advanced Model Management

#### `benchmark_model(model_id, num_runs?, prompt_lengths?)`

Run performance benchmarks on a model.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `model_id` | string | Yes | - | Model to benchmark |
| `num_runs` | int | No | 3 | Test runs per prompt length |
| `prompt_lengths` | list | No | [50, 200, 500] | Context sizes to test |

**Returns:**
```json
{
  "model_id": "llama-3.3-70b-instruct",
  "results": [
    { "prompt_length": 50, "avg_tokens_per_sec": 42.3, "avg_latency_ms": 1200 },
    { "prompt_length": 200, "avg_tokens_per_sec": 38.7, "avg_latency_ms": 5200 },
    { "prompt_length": 500, "avg_tokens_per_sec": 35.1, "avg_latency_ms": 14300 }
  ],
  "overall_avg_tokens_per_sec": 38.7
}
```

---

### Phase 8: Prompt Templates

#### `create_prompt_template(name, template, description?, variables?, preferred_model?, category?)`

Create a reusable prompt template.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `name` | string | Yes | - | Unique template name |
| `template` | string | Yes | - | Template with `{{variable}}` placeholders |
| `description` | string | No | - | What this template is for |
| `variables` | list | No | auto | Variable names (auto-detected if not provided) |
| `preferred_model` | string | No | - | Recommended model |
| `category` | string | No | general | One of: `general`, `coding`, `writing`, `analysis`, `creative` |

**Example:**
```json
{
  "name": "code-review",
  "template": "Review this {{language}} code for {{focus}}:\n\n```{{language}}\n{{code}}\n```\n\nProvide specific suggestions.",
  "description": "Code review template",
  "category": "coding"
}
```

#### `run_prompt_template(template_name, variables, model?)`

Execute a saved template.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `template_name` | string | Yes | - | Name of template to run |
| `variables` | dict | Yes | - | Variable values to substitute |
| `model` | string | No | - | Override model |

---

## Security

### SQL Injection Protection

All database operations use multiple layers of protection:

1. **Table Allowlist**: Only tables in `ALLOWED_TABLE_NAMES` can be queried
2. **Parameterized Queries**: All user input is passed as parameters, never interpolated
3. **Read-Only Connections**: Database opened in read-only mode by default
4. **Input Validation**: Table names validated before any query execution

**Allowed Tables:**
```python
ALLOWED_TABLE_NAMES = frozenset([
    "chats", "messages", "personas", "prompts", "mcp_tools", "tools",
    "knowledge_stacks", "models", "settings", "conversations", "users",
    "attachments", "embeddings", "tags", "folders", "providers",
    "chat_sessions", "chat_messages", "prompt_library"
])
```

### API Key Handling

- API keys are **NEVER** logged or returned in responses
- Provider queries automatically redact sensitive fields
- Environment variables used for configuration (never hardcoded)

### Network Security

- All local service calls use `127.0.0.1` (localhost only)
- Configurable timeouts prevent hanging requests
- No external network calls made by default
- Service availability checked before API calls

### Path Sanitization

- Home directory replaced with `$HOME` for portability
- Path expansion handles `~` and `$HOME` safely
- No shell injection possible through path parameters

---

## Rate Limiting

The MCP server does not implement rate limiting, but local AI services may have their own limits:

- **Ollama API**: Typically no rate limit for local requests
- **MLX Service**: Limited by hardware (GPU memory)
- **LLaMA.cpp**: Single request at a time (queued)

For heavy workloads, consider using the response cache (`get_cache_stats`, `clear_cache`).

---

## Versioning

This API follows semantic versioning:

- **Major** (6.x.x): Breaking changes to response formats or tool signatures
- **Minor** (x.5.x): New tools or non-breaking enhancements
- **Patch** (x.x.0): Bug fixes and documentation updates

Current version: **6.5.0**
