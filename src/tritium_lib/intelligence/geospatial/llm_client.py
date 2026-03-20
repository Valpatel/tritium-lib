# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""LLM client for geospatial intelligence — uses llama-server.

Discovers running llama-server instances on standard ports (8081-8089)
and uses the best available for terrain classification and terrain
brief generation.

Does NOT use ollama — llama-server only.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Standard ports to check for llama-server instances
_LLAMA_SERVER_PORTS = [8081, 8082, 8083, 8084, 8085, 8086, 8087, 8088, 8089]

# Cache of discovered servers
_discovered: dict[int, dict] = {}


def discover_llama_servers(
    ports: Optional[list[int]] = None,
    timeout: float = 1.0,
) -> list[dict]:
    """Discover running llama-server instances.

    Returns list of dicts with keys:
        port: int
        endpoint: str (http://127.0.0.1:{port})
        model_name: str
        n_params: int
        n_ctx_train: int
        capabilities: list[str]

    Results are cached until clear_discovery_cache() is called.
    """
    global _discovered

    if _discovered:
        return list(_discovered.values())

    try:
        import requests
    except ImportError:
        return []

    ports = ports or _LLAMA_SERVER_PORTS
    servers = []

    for port in ports:
        endpoint = f"http://127.0.0.1:{port}"
        try:
            health = requests.get(f"{endpoint}/health", timeout=timeout)
            if health.status_code != 200:
                continue

            models = requests.get(f"{endpoint}/v1/models", timeout=timeout)
            if models.status_code != 200:
                continue

            data = models.json()
            model_info = {}

            # Extract model metadata
            model_data = data.get("data", [{}])
            if model_data:
                meta = model_data[0].get("meta", {})
                model_info = {
                    "port": port,
                    "endpoint": endpoint,
                    "model_name": model_data[0].get("id", "unknown"),
                    "n_params": meta.get("n_params", 0),
                    "n_ctx_train": meta.get("n_ctx_train", 0),
                    "capabilities": data.get("models", [{}])[0].get("capabilities", []),
                }
            else:
                model_info = {
                    "port": port,
                    "endpoint": endpoint,
                    "model_name": "unknown",
                    "n_params": 0,
                    "n_ctx_train": 0,
                    "capabilities": [],
                }

            servers.append(model_info)
            _discovered[port] = model_info
            logger.debug(
                "Discovered llama-server on port %d: %s (%d params)",
                port, model_info["model_name"], model_info["n_params"],
            )

        except Exception:
            continue

    if servers:
        logger.info("Found %d llama-server instance(s)", len(servers))

    return servers


def get_best_server(
    prefer_large: bool = False,
    require_vision: bool = False,
) -> Optional[dict]:
    """Get the best available llama-server for a task.

    Args:
        prefer_large: prefer larger models (more params)
        require_vision: require multimodal/vision capability

    Returns:
        Server info dict, or None if no suitable server found.
    """
    servers = discover_llama_servers()
    if not servers:
        return None

    if require_vision:
        vision_servers = [s for s in servers if "vision" in s.get("capabilities", [])]
        if vision_servers:
            servers = vision_servers
        else:
            return None

    if prefer_large:
        servers.sort(key=lambda s: s.get("n_params", 0), reverse=True)
    else:
        # Prefer smaller models for speed
        servers.sort(key=lambda s: s.get("n_params", 0))

    return servers[0]


def llm_complete(
    prompt: str,
    endpoint: Optional[str] = None,
    max_tokens: int = 100,
    temperature: float = 0.1,
    timeout: float = 10.0,
) -> Optional[str]:
    """Send a completion request to llama-server.

    Uses the best available server if no endpoint specified.

    Returns the completion text, or None on failure.
    """
    try:
        import requests
    except ImportError:
        return None

    if endpoint is None:
        server = get_best_server()
        if server is None:
            return None
        endpoint = server["endpoint"]

    try:
        resp = requests.post(
            f"{endpoint.rstrip('/')}/v1/chat/completions",
            json={
                "model": "any",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    except Exception as e:
        logger.debug("LLM completion failed: %s", e)
        return None


def clear_discovery_cache() -> None:
    """Clear the cached server discovery results."""
    global _discovered
    _discovered = {}
