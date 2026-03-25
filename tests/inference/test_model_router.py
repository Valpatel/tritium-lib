# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.inference.model_router."""

from tritium_lib.inference.model_router import (
    ModelRouter,
    ModelProfile,
    TaskType,
    AllHostsFailedError,
)


def test_model_profile_creation():
    """ModelProfile can be created with defaults."""
    p = ModelProfile(name="test-model")
    assert p.name == "test-model"
    assert p.capabilities == {"text"}
    assert p.speed == "fast"


def test_model_profile_has_capability():
    """ModelProfile.has_capability works."""
    p = ModelProfile(name="test", capabilities={"text", "vision"})
    assert p.has_capability("text")
    assert p.has_capability("vision")
    assert not p.has_capability("code")


def test_model_profile_to_dict():
    """ModelProfile.to_dict returns serializable dict."""
    p = ModelProfile(name="test", capabilities={"text"})
    d = p.to_dict()
    assert d["name"] == "test"
    assert "text" in d["capabilities"]


def test_task_type_enum():
    """TaskType has expected values."""
    assert TaskType.SIMPLE_THINK.value == "simple_think"
    assert TaskType.VISION.value == "vision"
    assert TaskType.CHAT.value == "chat"


def test_router_creation():
    """ModelRouter can be created."""
    r = ModelRouter()
    assert r is not None
    assert r.profiles == []


def test_router_register():
    """ModelRouter.register adds profiles."""
    r = ModelRouter()
    r.register(ModelProfile(name="m1", capabilities={"text"}))
    assert len(r.profiles) == 1
    assert r.get_profile("m1") is not None


def test_router_unregister():
    """ModelRouter.unregister removes profiles."""
    r = ModelRouter()
    r.register(ModelProfile(name="m1"))
    r.unregister("m1")
    assert r.get_profile("m1") is None


def test_classify_task_vision():
    """classify_task returns VISION for images."""
    r = ModelRouter()
    assert r.classify_task(has_images=True) == TaskType.VISION


def test_classify_task_chat():
    """classify_task returns CHAT for chat context."""
    r = ModelRouter()
    assert r.classify_task(context={"is_chat": True}) == TaskType.CHAT


def test_classify_task_complex():
    """classify_task returns COMPLEX_REASON for threats."""
    r = ModelRouter()
    assert r.classify_task(context={"hostile_count": 5}) == TaskType.COMPLEX_REASON


def test_classify_task_default():
    """classify_task returns SIMPLE_THINK by default."""
    r = ModelRouter()
    assert r.classify_task() == TaskType.SIMPLE_THINK


def test_select_chain_filters():
    """select_chain filters by capability."""
    r = ModelRouter()
    r.register(ModelProfile(name="text-only", capabilities={"text"}, priority=1))
    r.register(ModelProfile(name="vision", capabilities={"text", "vision"}, priority=2))

    chain = r.select_chain(TaskType.VISION)
    assert len(chain) == 1
    assert chain[0].name == "vision"


def test_select_chain_quality_order():
    """select_chain prefers quality (high priority) for complex tasks."""
    r = ModelRouter()
    r.register(ModelProfile(name="small", capabilities={"text"}, priority=1))
    r.register(ModelProfile(name="big", capabilities={"text"}, priority=10))

    chain = r.select_chain(TaskType.COMPLEX_REASON)
    assert chain[0].name == "big"


def test_select_chain_speed_order():
    """select_chain prefers speed (low priority) for simple tasks."""
    r = ModelRouter()
    r.register(ModelProfile(name="small", capabilities={"text"}, priority=1))
    r.register(ModelProfile(name="big", capabilities={"text"}, priority=10))

    chain = r.select_chain(TaskType.SIMPLE_THINK)
    assert chain[0].name == "small"


def test_from_static_factory():
    """ModelRouter.from_static creates a pre-configured router."""
    r = ModelRouter.from_static()
    assert len(r.profiles) == 2
    assert r.get_profile("gemma3:4b") is not None


def test_all_hosts_failed_error():
    """AllHostsFailedError contains task type."""
    err = AllHostsFailedError(TaskType.VISION)
    assert err.task_type == TaskType.VISION
    assert "vision" in str(err)
