# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Gateway-backend tests for tritium_lib.inference.fleet.

The fleet integrates with a live-network inference gateway (a load
balancer that fronts a cluster of GPU inference servers) and falls back
to direct llama-server / ollama. These tests stand up an in-process mock
gateway and exercise discovery, preference order, and chat/generate
routing — entirely offline, no real hosts.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tritium_lib.inference.fleet import LLMFleet, FleetHost


# --------------------------------------------------------------------------
# In-process mock gateway: GET /health, GET /v1/models, POST /v1/chat,
# POST /v1/generate — the live-network gateway contract.
# --------------------------------------------------------------------------
class _MockGatewayHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "backends": {"healthy": 3, "total": 4}})
        elif self.path == "/v1/models":
            self._send(200, {"models": ["qwen2.5:7b", "llava:7b"]})
        else:
            self._send(404, {"error": "not_found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        # Echo the X-Source header back so tests can assert it was sent.
        src = self.headers.get("X-Source", "")
        if self.path == "/v1/generate":
            self._send(200, {
                "response": f"gen:{payload.get('prompt', '')}|src={src}",
                "model": payload.get("model", ""),
                "backend": "cluster-node-A", "cached": False, "eval_count": 7,
            })
        elif self.path == "/v1/chat":
            msgs = payload.get("messages", [])
            last = msgs[-1]["content"] if msgs else ""
            self._send(200, {
                "message": {"role": "assistant", "content": f"chat:{last}|src={src}"},
                "model": payload.get("model", ""), "cached": False, "eval_count": 9,
            })
        else:
            self._send(404, {"error": "not_found"})


@pytest.fixture
def mock_gateway():
    srv = HTTPServer(("127.0.0.1", 0), _MockGatewayHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


# --------------------------------------------------------------------------
# FleetHost gateway semantics
# --------------------------------------------------------------------------
def test_gateway_host_serves_any_model():
    """A gateway fronts the whole cluster, so it can serve any model even
    with an empty/unknown model list (it routes internally)."""
    h = FleetHost(url="http://gw:1", name="gw", backend="gateway")
    assert h.has_model("qwen2.5:7b")
    assert h.has_model("anything-at-all")


def test_direct_host_still_needs_the_model():
    """A direct llama-server only 'has' models it actually lists."""
    h = FleetHost(url="http://h:8081", name="h", backend="llama-server",
                  models=["qwen2.5:7b"])
    assert h.has_model("qwen2.5")
    assert not h.has_model("llava")


# --------------------------------------------------------------------------
# Preference: gateway > llama-server > ollama
# --------------------------------------------------------------------------
def test_best_host_prefers_gateway_then_llama_then_ollama():
    fleet = LLMFleet(auto_discover=False)
    fleet._hosts = [
        FleetHost(url="http://o", name="o", backend="ollama", models=["m:1"]),
        FleetHost(url="http://l", name="l", backend="llama-server", models=["m:1"]),
        FleetHost(url="http://g", name="g", backend="gateway"),
    ]
    assert fleet.best_host("m:1").backend == "gateway"
    # Remove the gateway -> llama-server wins over ollama.
    fleet._hosts = [h for h in fleet._hosts if h.backend != "gateway"]
    assert fleet.best_host("m:1").backend == "llama-server"


# --------------------------------------------------------------------------
# Live discovery + routing against the mock gateway
# --------------------------------------------------------------------------
def test_discovers_gateway_from_env(mock_gateway, monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_URL", mock_gateway)
    # Only discover the gateway, not LAN/localhost (keep the test hermetic).
    fleet = LLMFleet(auto_discover=False)
    fleet.discover_gateways()
    gws = [h for h in fleet.hosts if h.backend == "gateway"]
    assert len(gws) == 1
    assert gws[0].backends_connected == 3


def test_gateway_generate_and_chat_round_trip(mock_gateway, monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_URL", mock_gateway)
    monkeypatch.setenv("LLM_SOURCE", "tritium-test")
    fleet = LLMFleet(auto_discover=False)
    fleet.discover_gateways()

    gen = fleet.generate("qwen2.5:7b", "hello")
    assert gen.startswith("gen:hello")
    assert "src=tritium-test" in gen  # X-Source header propagated

    chat = fleet.chat("qwen2.5:7b", "hi there")
    assert chat.startswith("chat:hi there")
    assert "src=tritium-test" in chat


def test_gateway_in_conf_file(mock_gateway, monkeypatch, tmp_path):
    """A `gateway = URL` line in llm-fleet.conf is discovered, no env."""
    conf = tmp_path / "llm-fleet.conf"
    conf.write_text(f"# fleet\ngateway = {mock_gateway}\n")
    monkeypatch.setenv("LLM_FLEET_CONF", str(conf))
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    fleet = LLMFleet(auto_discover=False)
    fleet.discover_gateways()
    assert any(h.backend == "gateway" for h in fleet.hosts)


def test_no_gateway_configured_is_graceful(monkeypatch):
    """No gateway env/conf -> discover_gateways finds nothing, no crash."""
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    monkeypatch.delenv("LLM_FLEET_CONF", raising=False)
    fleet = LLMFleet(auto_discover=False)
    fleet.discover_gateways()
    assert all(h.backend != "gateway" for h in fleet.hosts)
