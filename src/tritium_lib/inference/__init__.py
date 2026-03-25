# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Inference modules — LLM fleet management and client functions."""

from .fleet import LLMFleet, FleetHost, OllamaFleet
from .llm_client import ollama_chat, llama_server_chat, set_ollama_host

__all__ = [
    "LLMFleet", "FleetHost", "OllamaFleet",
    "ollama_chat", "llama_server_chat", "set_ollama_host",
]
