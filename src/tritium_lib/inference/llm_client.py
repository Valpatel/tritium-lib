# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""LLM client functions — pure urllib, no external dependencies.

Extracted from tritium-sc vision.py. Supports both llama-server
(OpenAI-compatible) and ollama (legacy) APIs.
"""

from __future__ import annotations

import json
import urllib.request

# Default LLM host — overridden by set_ollama_host()
_ollama_host: str = "http://localhost:8081"


def set_ollama_host(host: str) -> None:
    """Set the LLM API host (called during initialization)."""
    global _ollama_host
    _ollama_host = host


def _is_llama_server(url: str) -> bool:
    """Check if the host is llama-server (not ollama) by port heuristic."""
    port = url.rstrip("/").split(":")[-1]
    return port in ("8081", "8082", "8083")


def ollama_chat(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    base_url: str | None = None,
) -> dict:
    """Call LLM chat API with optional tools and images.

    Automatically detects llama-server vs ollama by port and uses
    the correct API format:
    - llama-server: /v1/chat/completions (OpenAI-compatible)
    - ollama: /api/chat (legacy)
    """
    url = base_url or _ollama_host
    use_openai = _is_llama_server(url)

    payload: dict = {
        "model": model,
        "messages": messages,
    }
    if use_openai:
        payload["max_tokens"] = 2048
    else:
        payload["stream"] = False
    if tools:
        payload["tools"] = tools

    endpoint = "/v1/chat/completions" if use_openai else "/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        if use_openai:
            # Convert OpenAI format to ollama format for backward compat
            choices = result.get("choices", [])
            if choices:
                return {"message": choices[0].get("message", {})}
            return {"message": {"content": ""}}
        return result


def llama_server_chat(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    base_url: str | None = None,
    max_tokens: int = 2048,
) -> dict:
    """Call llama-server directly using OpenAI-compatible API.

    Unlike ollama_chat(), this always uses the /v1/chat/completions
    endpoint without auto-detection.
    """
    url = base_url or _ollama_host

    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        choices = result.get("choices", [])
        if choices:
            return {"message": choices[0].get("message", {})}
        return {"message": {"content": ""}}
