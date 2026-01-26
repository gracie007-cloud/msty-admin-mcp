"""
Msty Admin MCP - Model Tagging System

Smart model categorization for intelligent selection.
"""

import re
from typing import List, Optional

from .cache import get_cached_models
from .network import get_available_service_ports, make_api_request


# Model tags based on known model characteristics
MODEL_TAGS = {
    # By name patterns - expanded for comprehensive tagging
    "patterns": {
        "fast": ["granite", "phi", "gemma-2b", "qwen.*0.6b", "qwen.*1.5b", "tiny", "small", "mini", "flash", "lite", "haiku"],
        "quality": ["70b", "72b", "80b", "405b", "235b", "253b", "120b", "opus", "sonnet", "pro", "ultra", "nemotron", "v3"],
        "coding": ["coder", "codex", "deepseek-coder", "starcoder", "code", "dev", "kimi-dev", "oswe", "grok-code"],
        "creative": ["creative", "writer", "story", "hermes", "nous"],
        "reasoning": ["r1", "thinking", "reason", "o1", "deepseek-r1", "qwen3", "glm", "nemotron"],
        "embedding": ["embed", "bge", "nomic", "e5", "gte", "jina-embed"],
        "vision": ["vision", "llava", "image", "visual", "pro-image"],
        "long_context": ["longcat", "yarn", "longrope", "nemo", "mistral-nemo"],
    },
    # Manual overrides for specific models - comprehensive list
    "overrides": {
        # MLX Models
        "mlx-community/granite-3.3-2b-instruct-4bit": ["fast", "general"],
        "mlx-community/Qwen3-32B-MLX-4bit": ["quality", "general", "reasoning", "medium"],
        "mlx-community/Qwen3-235B-A22B-8bit": ["quality", "reasoning", "large"],
        "mlx-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit": ["quality", "reasoning", "large"],
        "mlx-community/Hermes-4-405B-MLX-6bit": ["quality", "creative", "reasoning", "large"],
        "mlx-community/Hermes-4-70B-MLX-4bit": ["quality", "creative", "reasoning", "large"],
        "mlx-community/Kimi-Dev-72B-4bit-DWQ": ["quality", "coding", "reasoning", "large"],
        "mlx-community/Mistral-Nemo-Instruct-2407-4bit": ["quality", "long_context", "general", "medium"],
        "GGorman/DeepSeek-Coder-V2-Instruct-Q4-mlx": ["coding", "quality", "reasoning"],
        "inferencelabs/GLM-4.7-MLX-6.5bit": ["quality", "reasoning", "general"],
        "inferencerlabs/LongCat-Flash-Thinking-2601-MLX-5.5bit": ["reasoning", "long_context", "fast"],
        # GGUF Models
        "DeepSeek/DeepSeek-R1-Distill-Llama-70B-GGUF/DeepSeek-R1-Distill-Llama-70B-Q8_0.gguf": ["quality", "reasoning", "large"],
        "DeepSeek/DeepSeek-V3-0324-UD-Q4_K_XL/DeepSeek-V3-0324-UD-Q4_K_XL-merged.gguf": ["quality", "reasoning", "coding", "large"],
        "DevQuasar/nvidia.Llama-3_1-Nemotron-Ultra-253B-v1-GGUF/nvidia.Llama-3_1-Nemotron-Ultra-253B-v1.Q4_K_M_Dima.gguf": ["quality", "reasoning", "large"],
        "Gemma/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q4_K_M.gguf": ["quality", "general", "medium"],
        "Llama/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-Q8_0.gguf": ["quality", "general", "reasoning", "large"],
        "OpenAi/gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4_Dima.gguf": ["quality", "reasoning", "coding", "large"],
        "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/qwen2.5-coder-32b-instruct-fp16.gguf": ["coding", "quality", "medium"],
        "mradermacher/Qwen3-72B-Instruct-2-i1-GGUF/Qwen3-72B-Instruct-2.i1-Q5_K_M.gguf": ["quality", "reasoning", "large"],
        "yinghy2018/CLIENS_PHI4_14B_BF16-Q8_0-GGUF/cliens_phi4_14b_bf16-q8_0.gguf": ["fast", "general", "medium"],
        # Claude Models
        "claude-opus-4-5-20251101": ["quality", "reasoning", "coding", "creative", "large"],
        "claude-opus-4.5": ["quality", "reasoning", "coding", "creative", "large"],
        "claude-opus-4-20250514": ["quality", "reasoning", "coding", "creative", "large"],
        "claude-opus-4.1": ["quality", "reasoning", "coding", "creative", "large"],
        "claude-opus-4-1-20250805": ["quality", "reasoning", "coding", "creative", "large"],
        "claude-sonnet-4.5": ["quality", "reasoning", "coding", "creative"],
        "claude-sonnet-4-5-20250929": ["quality", "reasoning", "coding", "creative"],
        "claude-sonnet-4": ["quality", "reasoning", "coding"],
        "claude-sonnet-4-20250514": ["quality", "reasoning", "coding"],
        "claude-3-7-sonnet-20250219": ["quality", "reasoning", "coding"],
        "claude-haiku-4.5": ["fast", "general", "coding"],
        "claude-haiku-4-5-20251001": ["fast", "general", "coding"],
        "claude-3-5-haiku-20241022": ["fast", "general"],
        # GPT Models
        "gpt-5": ["quality", "reasoning", "general", "large"],
        "gpt-5.1": ["quality", "reasoning", "general", "large"],
        "gpt-5.2": ["quality", "reasoning", "general", "large"],
        "gpt-5-mini": ["fast", "general"],
        "gpt-5-codex": ["coding", "quality"],
        "gpt-5.1-codex": ["coding", "quality"],
        "gpt-5.1-codex-mini": ["coding", "fast"],
        "gpt-5.1-codex-max": ["coding", "quality", "large"],
        "gpt-5.2-codex": ["coding", "quality"],
        "gpt-4.1": ["quality", "reasoning", "general"],
        "gpt-oss-120b-medium": ["quality", "reasoning", "large"],
        # Gemini Models
        "gemini-2.5-pro": ["quality", "reasoning", "long_context"],
        "gemini-2.5-flash": ["fast", "general"],
        "gemini-2.5-flash-lite": ["fast", "general"],
        "gemini-3-pro-preview": ["quality", "reasoning"],
        "gemini-3-flash-preview": ["fast", "general"],
        "gemini-3-pro-image-preview": ["quality", "vision"],
        # Hybrid/Other Models
        "gemini-claude-sonnet-4-5": ["quality", "reasoning", "coding"],
        "gemini-claude-sonnet-4-5-thinking": ["quality", "reasoning", "coding"],
        "gemini-claude-opus-4-5-thinking": ["quality", "reasoning", "coding", "large"],
        "grok-code-fast-1": ["coding", "fast"],
        "oswe-vscode-prime": ["coding", "quality"],
    }
}


def get_model_tags(model_id: str) -> List[str]:
    """
    Get tags for a model based on its ID.
    Tags help with smart model selection.
    """
    tags = set()
    model_lower = model_id.lower()

    # Check manual overrides first
    if model_id in MODEL_TAGS["overrides"]:
        return MODEL_TAGS["overrides"][model_id]

    # Check patterns
    for tag, patterns in MODEL_TAGS["patterns"].items():
        for pattern in patterns:
            if re.search(pattern, model_lower):
                tags.add(tag)
                break

    # Add size-based tags (check larger sizes first to avoid false matches)
    if any(size in model_lower for size in ["70b", "72b", "80b", "120b", "235b", "253b", "405b"]):
        tags.add("large")
    elif any(size in model_lower for size in ["13b", "14b", "27b", "32b", "34b"]):
        tags.add("medium")
    elif any(size in model_lower for size in ["2b", "3b", "4b", "7b", "8b"]):
        tags.add("small")

    # Default to general if no specific tags
    if not tags:
        tags.add("general")

    return list(tags)


def find_models_by_tag(tag: str, models: List[dict] = None) -> List[dict]:
    """
    Find models matching a specific tag.
    If models not provided, fetches from cache or API.
    """
    if models is None:
        cached = get_cached_models()
        if cached:
            models = cached.get("models", [])
        else:
            # Fetch fresh
            services = get_available_service_ports()
            models = []
            for service_name, service_info in services.items():
                if service_info["available"]:
                    response = make_api_request("/v1/models", port=service_info["port"])
                    if response.get("success"):
                        data = response.get("data", {})
                        if isinstance(data, dict) and "data" in data:
                            for m in data["data"]:
                                m["_service"] = service_name
                                m["_port"] = service_info["port"]
                            models.extend(data["data"])

    matching = []
    for model in models:
        model_id = model.get("id", "")
        model_tags = get_model_tags(model_id)
        if tag in model_tags:
            model["_tags"] = model_tags
            matching.append(model)

    return matching


__all__ = [
    "MODEL_TAGS",
    "get_model_tags",
    "find_models_by_tag"
]
