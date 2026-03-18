# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""NPC thinking engine — gives simulated entities inner thoughts via small LLMs.

Uses llama-server (preferred) or Ollama for concurrent NPC inference.
Small models (0.5B-1.5B) are fast enough for real-time NPC thoughts.

Usage::

    thinker = NPCThinker(endpoint="http://localhost:8080")
    thought = await thinker.think(npc, situation)
    # → "I should head to the coffee shop before my shift starts"
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Prompt templates for different NPC states
THOUGHT_PROMPTS = {
    "idle": (
        "You are {name}, a {role} in a city. It's {time_of_day}. "
        "You are currently idle at {location}. "
        "In one short sentence, what are you thinking about doing next?"
    ),
    "walking": (
        "You are {name}, walking to {destination}. It's {time_of_day}. "
        "In one short sentence, what's on your mind?"
    ),
    "fleeing": (
        "You are {name}. Something dangerous is happening nearby — {threat}. "
        "In one short sentence, what are you thinking?"
    ),
    "protesting": (
        "You are {name}, part of a protest at the town plaza. "
        "The crowd mood is {mood}. In one short sentence, what are you chanting or thinking?"
    ),
    "driving": (
        "You are {name}, driving a {vehicle} to {destination}. It's {time_of_day}. "
        "In one short sentence, what's on your mind?"
    ),
}


@dataclass
class NPCThought:
    """A single thought from an NPC."""
    npc_id: str
    npc_name: str
    thought: str
    state: str
    timestamp: float = field(default_factory=time.time)
    model: str = ""
    latency_ms: float = 0


class NPCThinker:
    """Generates NPC thoughts using a local LLM server.

    Supports:
    - llama-server (OpenAI-compatible API, handles concurrent requests)
    - Ollama (fallback, serializes requests)

    Args:
        endpoint: LLM server URL (default: try llama-server then Ollama)
        model: Model name (default: smallest available)
        max_concurrent: Max simultaneous inference requests
        cooldown_s: Minimum seconds between thoughts per NPC
    """

    def __init__(
        self,
        endpoint: str = "",
        model: str = "",
        max_concurrent: int = 4,
        cooldown_s: float = 30.0,
    ):
        self.endpoint = endpoint
        self.model = model
        self.max_concurrent = max_concurrent
        self.cooldown_s = cooldown_s
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._last_thought: dict[str, float] = {}
        self._thought_history: list[NPCThought] = []
        self._api_type = ""  # "llama" or "ollama", detected on first call
        self._available = True

    async def _detect_endpoint(self) -> str:
        """Auto-detect available LLM endpoint."""
        import aiohttp

        # Try llama-server first (better concurrency)
        for url in ["http://localhost:8080", "http://localhost:8081"]:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            self._api_type = "llama"
                            logger.info(f"NPCThinker: using llama-server at {url}")
                            return url
            except Exception:
                pass

        # Fall back to Ollama
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        self._api_type = "ollama"
                        logger.info("NPCThinker: using Ollama at localhost:11434")
                        return "http://localhost:11434"
        except Exception:
            pass

        logger.warning("NPCThinker: no LLM server available")
        self._available = False
        return ""

    async def _detect_model(self) -> str:
        """Pick the smallest available model."""
        if self._api_type == "ollama":
            import aiohttp
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.endpoint}/api/tags") as resp:
                        data = await resp.json()
                        models = [m["name"] for m in data.get("models", [])]
                        # Prefer small models for NPC thinking
                        for preferred in ["qwen2.5:0.5b", "qwen2.5:1.5b", "llama3.2:1b", "phi4-mini"]:
                            if preferred in models:
                                return preferred
                        if models:
                            return models[0]
            except Exception:
                pass
        return self.model or "qwen2.5:0.5b"

    async def think(self, npc: dict, situation: dict = None) -> NPCThought | None:
        """Generate a thought for an NPC.

        Args:
            npc: Entity dict with target_id, name, identity, status, etc.
            situation: Context dict with time_of_day, threats, crowd_mood, etc.

        Returns:
            NPCThought or None if on cooldown or unavailable.
        """
        npc_id = npc.get("target_id", npc.get("id", ""))
        now = time.time()

        # Cooldown check
        if now - self._last_thought.get(npc_id, 0) < self.cooldown_s:
            return None

        if not self._available:
            return None

        # Detect endpoint on first call
        if not self.endpoint:
            self.endpoint = await self._detect_endpoint()
            if not self.endpoint:
                return None
        if not self.model:
            self.model = await self._detect_model()

        # Build prompt
        situation = situation or {}
        identity = npc.get("identity", {})
        name = identity.get("first_name", npc.get("name", "Someone"))
        status = npc.get("status", npc.get("fsm_state", "idle"))

        template = THOUGHT_PROMPTS.get(status, THOUGHT_PROMPTS["idle"])
        prompt = template.format(
            name=name,
            role=npc.get("asset_type", "person"),
            time_of_day=situation.get("time_of_day", "daytime"),
            location=situation.get("location", "the city"),
            destination=situation.get("destination", "somewhere"),
            threat=situation.get("threat", "a disturbance"),
            mood=situation.get("crowd_mood", "tense"),
            vehicle=situation.get("vehicle", "car"),
        )

        # Rate limit concurrent requests
        async with self._semaphore:
            start = time.monotonic()
            try:
                thought_text = await self._generate(prompt)
            except Exception as e:
                logger.debug(f"NPCThinker error for {npc_id}: {e}")
                return None
            latency = (time.monotonic() - start) * 1000

        self._last_thought[npc_id] = now
        thought = NPCThought(
            npc_id=npc_id,
            npc_name=name,
            thought=thought_text.strip(),
            state=status,
            model=self.model,
            latency_ms=latency,
        )
        self._thought_history.append(thought)
        if len(self._thought_history) > 100:
            self._thought_history = self._thought_history[-50:]

        return thought

    async def _generate(self, prompt: str) -> str:
        """Call the LLM server."""
        import aiohttp

        if self._api_type == "llama":
            # OpenAI-compatible API
            url = f"{self.endpoint}/v1/chat/completions"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": 0.8,
            }
        else:
            # Ollama API
            url = f"{self.endpoint}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 50, "temperature": 0.8},
            }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()

        if self._api_type == "llama":
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            return data.get("response", "")

    def get_recent_thoughts(self, count: int = 10) -> list[dict]:
        """Return recent thoughts for display."""
        return [
            {
                "npc_id": t.npc_id,
                "name": t.npc_name,
                "thought": t.thought,
                "state": t.state,
                "time": t.timestamp,
                "latency_ms": t.latency_ms,
            }
            for t in self._thought_history[-count:]
        ]
