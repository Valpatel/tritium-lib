# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for NPCThinker — LLM-powered NPC thought generation."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tritium_lib.sim_engine.core.npc_thinker import NPCThinker, NPCThought, THOUGHT_PROMPTS


class TestThoughtPrompts:
    def test_all_states_have_prompts(self):
        for state in ["idle", "walking", "fleeing", "protesting", "driving"]:
            assert state in THOUGHT_PROMPTS

    def test_prompts_have_placeholders(self):
        for state, template in THOUGHT_PROMPTS.items():
            assert "{name}" in template


class TestNPCThought:
    def test_creation(self):
        t = NPCThought(
            npc_id="npc_001",
            npc_name="Sarah",
            thought="I should get coffee",
            state="idle",
        )
        assert t.npc_id == "npc_001"
        assert t.npc_name == "Sarah"
        assert t.thought == "I should get coffee"
        assert t.timestamp > 0


class TestNPCThinker:
    def test_init_defaults(self):
        thinker = NPCThinker()
        assert thinker.max_concurrent == 4
        assert thinker.cooldown_s == 30.0
        assert thinker._available is True

    def test_init_custom(self):
        thinker = NPCThinker(
            endpoint="http://localhost:8080",
            model="test-model",
            max_concurrent=2,
            cooldown_s=10.0,
        )
        assert thinker.endpoint == "http://localhost:8080"
        assert thinker.model == "test-model"

    @pytest.mark.asyncio
    async def test_cooldown_prevents_rapid_thoughts(self):
        thinker = NPCThinker(cooldown_s=60.0)
        thinker._available = True
        thinker._last_thought["npc_001"] = time.time()

        npc = {"target_id": "npc_001", "name": "Test", "status": "idle"}
        result = await thinker.think(npc)
        assert result is None  # On cooldown

    @pytest.mark.asyncio
    async def test_unavailable_returns_none(self):
        thinker = NPCThinker()
        thinker._available = False

        npc = {"target_id": "npc_001", "name": "Test", "status": "idle"}
        result = await thinker.think(npc)
        assert result is None

    def test_get_recent_thoughts_empty(self):
        thinker = NPCThinker()
        assert thinker.get_recent_thoughts() == []

    def test_get_recent_thoughts_with_data(self):
        thinker = NPCThinker()
        thinker._thought_history = [
            NPCThought(npc_id="1", npc_name="A", thought="hi", state="idle"),
            NPCThought(npc_id="2", npc_name="B", thought="bye", state="walking"),
        ]
        recent = thinker.get_recent_thoughts(1)
        assert len(recent) == 1
        assert recent[0]["name"] == "B"

    def test_get_recent_thoughts_serializable(self):
        thinker = NPCThinker()
        thinker._thought_history = [
            NPCThought(npc_id="1", npc_name="A", thought="hi", state="idle", latency_ms=50),
        ]
        recent = thinker.get_recent_thoughts()
        import json
        json.dumps(recent)  # Should not raise
