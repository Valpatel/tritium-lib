# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Production LLM Fleet — multi-host inference with auto-discovery.

Uses llama-server (OpenAI-compatible API) instead of ollama.
llama-server runs on ports 8081-8083 with /v1/chat/completions.

Discovers llama-server instances from:
1. conf/llm-fleet.conf (gitignored — no host info in repo)
2. LLM_HOSTS env var (comma-separated)
3. OLLAMA_HOSTS env var (legacy compat)
4. Tailscale network scan (auto-discover peers)
5. Localhost fallback (ports 8081-8083)

All host references are dynamic — no IPs or hostnames baked into source.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONF_PATH = Path(__file__).parent.parent / "conf" / "llm-fleet.conf"
LEGACY_CONF_PATH = Path(__file__).parent.parent / "conf" / "ollama-fleet.conf"
# llama-server default ports (multiple instances for different models)
DEFAULT_PORTS = [8081, 8082, 8083]
# Legacy ollama port — still probed for backward compatibility
LEGACY_PORT = 11434
PROBE_TIMEOUT = 3  # seconds

# --- Inference gateway (the live-network front-end) -----------------------
# The production path is an inference gateway: one HTTP service that load-
# balances a cluster of GPU inference servers and exposes a stable API
# (GET /health -> {"backends": {"healthy": N}}, POST /v1/generate,
# POST /v1/chat). When a gateway is configured it is preferred over direct
# llama-server / ollama (it has the whole cluster behind it). Everything is
# config-driven — NO hosts, ports, or tokens are baked into source.
#
#   LLM_GATEWAY_URL   comma-separated gateway base URLs (http://host:port)
#   LLM_FLEET_CONF    override path to llm-fleet.conf (a `gateway = URL` line)
#   LLM_SOURCE        X-Source identifier sent to the gateway (default tritium)
#   LLM_PRIORITY      optional X-Priority QoS hint
GATEWAY_ENV = "LLM_GATEWAY_URL"
CONF_ENV = "LLM_FLEET_CONF"
SOURCE_ENV = "LLM_SOURCE"
PRIORITY_ENV = "LLM_PRIORITY"
DEFAULT_SOURCE = "tritium"


def _conf_paths() -> list[Path]:
    """Config files to read, honoring the LLM_FLEET_CONF override."""
    paths: list[Path] = []
    override = os.environ.get(CONF_ENV)
    if override:
        paths.append(Path(override))
    paths.extend([CONF_PATH, LEGACY_CONF_PATH])
    return paths


@dataclass
class FleetHost:
    """A reachable LLM inference instance on the fleet."""
    url: str
    name: str
    models: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    backend: str = "llama-server"  # 'gateway' | 'llama-server' | 'ollama'
    backends_connected: int = 0  # gateway: healthy cluster nodes behind it

    def has_model(self, model: str) -> bool:
        """Check if this host can serve a model (prefix match).

        A gateway fronts the whole cluster and routes internally, so it
        can serve any model regardless of what it lists — the cluster
        behind it owns the model inventory.
        """
        if self.backend == "gateway":
            return True
        prefix = model.split(":")[0]
        return any(m.startswith(prefix) for m in self.models)


class LLMFleet:
    """Manages a pool of LLM hosts for production inference.

    Supports both llama-server (OpenAI-compatible) and ollama (legacy).
    Auto-discovers hosts from conf file, env vars, and Tailscale.
    Provides host selection by model capability and latency.
    """

    def __init__(self, auto_discover: bool = True):
        self._hosts: list[FleetHost] = []
        self._discover(auto_discover)

    @property
    def hosts(self) -> list[FleetHost]:
        return list(self._hosts)

    @property
    def count(self) -> int:
        return len(self._hosts)

    def hosts_with_model(self, model: str) -> list[FleetHost]:
        """Return hosts that have a specific model, sorted by latency."""
        return [h for h in self._hosts if h.has_model(model)]

    # Backend preference: the gateway fronts the whole cluster, so it wins;
    # then a direct llama-server tier; ollama is the legacy fallback.
    _BACKEND_RANK = {"gateway": 0, "llama-server": 1, "ollama": 2}

    def best_host(self, model: str) -> FleetHost | None:
        """Return the preferred host that can serve this model, or None.

        Order: gateway > llama-server > ollama, then lowest latency.
        """
        hosts = self.hosts_with_model(model)
        if not hosts:
            return None
        return min(hosts, key=lambda h: (
            self._BACKEND_RANK.get(h.backend, 9), h.latency_ms))

    def refresh(self) -> None:
        """Re-discover hosts (useful after network changes)."""
        self._hosts.clear()
        self._discover(auto_discover=True)

    def _discover(self, auto_discover: bool) -> None:
        """Build host list from all sources."""
        # Gateway first — the configured live-network front-end.
        self.discover_gateways()

        candidates: set[str] = set()

        # Always include localhost llama-server ports
        for port in DEFAULT_PORTS:
            candidates.add(f"localhost:{port}")
        # Also probe legacy ollama port
        candidates.add(f"localhost:{LEGACY_PORT}")

        # 1. Conf file (new llm-fleet.conf or legacy ollama-fleet.conf)
        for conf in _conf_paths():
            if conf.exists():
                for line in conf.read_text().splitlines():
                    line = line.strip()
                    if (line and not line.startswith("#")
                            and not line.lower().startswith("gateway")):
                        if ":" not in line:
                            for port in DEFAULT_PORTS:
                                candidates.add(f"{line}:{port}")
                        else:
                            candidates.add(line)

        # 2. LLM_HOSTS env var (preferred) and OLLAMA_HOSTS (legacy)
        for env_key in ["LLM_HOSTS", "OLLAMA_HOSTS"]:
            env_hosts = os.environ.get(env_key, "")
            for h in env_hosts.split(","):
                h = h.strip()
                if h:
                    if ":" not in h:
                        for port in DEFAULT_PORTS:
                            candidates.add(f"{h}:{port}")
                    else:
                        candidates.add(h)

        # 3. Tailscale auto-discovery
        if auto_discover:
            candidates.update(self._scan_tailscale())

        # 4. LAN subnet scan
        if auto_discover:
            candidates.update(self._scan_lan_subnet())

        # Probe all candidates in parallel
        if not candidates:
            return

        with ThreadPoolExecutor(max_workers=min(len(candidates), 10)) as pool:
            futures = {pool.submit(self._probe, c): c for c in candidates}
            for f in as_completed(futures, timeout=PROBE_TIMEOUT + 2):
                try:
                    host = f.result()
                    if host is not None:
                        self._hosts.append(host)
                except Exception:
                    pass

        # Sort by backend preference (gateway > llama-server > ollama),
        # then latency. Gateways were already added by discover_gateways().
        self._hosts.sort(key=lambda h: (
            self._BACKEND_RANK.get(h.backend, 9), h.latency_ms))

    def _scan_tailscale(self) -> set[str]:
        """Discover LLM servers on Tailscale peers."""
        hosts: set[str] = set()
        try:
            result = subprocess.run(
                ["tailscale", "status", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for peer_id, peer in data.get("Peer", {}).items():
                    if not peer.get("Online", False):
                        continue
                    name = peer.get("HostName", "")
                    if name:
                        for port in DEFAULT_PORTS:
                            hosts.add(f"{name}:{port}")
        except (subprocess.TimeoutExpired, FileNotFoundError,
                json.JSONDecodeError, OSError):
            pass
        return hosts

    def _scan_lan_subnet(self) -> set[str]:
        """Discover LLM servers on LAN by scanning the local /24 subnet."""
        import socket
        hosts: set[str] = set()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.1)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            subnet = ".".join(local_ip.split(".")[:3])

            probe_ips = [f"{subnet}.{i}" for i in list(range(1, 21)) + list(range(200, 255))]
            my_last = int(local_ip.split(".")[-1])
            for offset in range(-5, 6):
                ip_last = my_last + offset
                if 1 <= ip_last <= 254:
                    probe_ips.append(f"{subnet}.{ip_last}")

            probe_ips = list(set(probe_ips) - {local_ip})

            import urllib.request

            def _quick_check(ip: str) -> str | None:
                # Try llama-server ports first, then ollama
                for port in DEFAULT_PORTS + [LEGACY_PORT]:
                    try:
                        endpoint = "/health" if port != LEGACY_PORT else "/api/tags"
                        req = urllib.request.Request(
                            f"http://{ip}:{port}{endpoint}",
                            headers={"Accept": "application/json"},
                        )
                        with urllib.request.urlopen(req, timeout=1) as resp:
                            if resp.status == 200:
                                return f"{ip}:{port}"
                    except Exception:
                        continue
                return None

            with ThreadPoolExecutor(max_workers=30) as pool:
                for result in pool.map(_quick_check, probe_ips[:60]):
                    if result:
                        hosts.add(result)
        except Exception:
            pass
        return hosts

    def _probe(self, host_port: str) -> FleetHost | None:
        """Check if a host has an LLM server running and list its models."""
        import urllib.request
        url = f"http://{host_port}"
        name = host_port.split(":")[0]
        port = int(host_port.split(":")[1]) if ":" in host_port else DEFAULT_PORTS[0]

        # Try llama-server first (/v1/models), then ollama (/api/tags)
        for endpoint, backend, model_key in [
            ("/v1/models", "llama-server", "data"),
            ("/health", "llama-server", None),
            ("/api/tags", "ollama", "models"),
        ]:
            try:
                t0 = time.monotonic()
                req = urllib.request.Request(
                    f"{url}{endpoint}",
                    headers={"Accept": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as resp:
                    latency = (time.monotonic() - t0) * 1000
                    body = resp.read().decode()

                    models = []
                    if model_key and body:
                        data = json.loads(body)
                        if backend == "llama-server":
                            # OpenAI format: {"data": [{"id": "model_name"}]}
                            models = [m.get("id", m.get("model", "")) for m in data.get("data", [])]
                        else:
                            # Ollama format: {"models": [{"name": "model:tag"}]}
                            models = [m["name"] for m in data.get("models", [])]
                    elif endpoint == "/health":
                        # Health endpoint doesn't list models — mark as available
                        models = ["unknown"]

                    if models or endpoint == "/health":
                        return FleetHost(
                            url=url, name=name, models=models,
                            latency_ms=latency, backend=backend,
                        )
            except Exception:
                continue

        return None

    # ----- Gateway (live-network front-end) -------------------------------

    def _gateway_urls(self) -> list[str]:
        """Configured gateway base URLs, from env + conf. No defaults —
        a gateway is used only when explicitly configured."""
        urls: list[str] = []
        env = os.environ.get(GATEWAY_ENV, "")
        for u in env.split(","):
            u = u.strip().rstrip("/")
            if u:
                urls.append(u)
        for conf in _conf_paths():
            if not conf.exists():
                continue
            for line in conf.read_text().splitlines():
                line = line.strip()
                if line.lower().startswith("gateway"):
                    # `gateway = URL` or `gateway=URL`
                    _, _, val = line.partition("=")
                    val = val.strip().rstrip("/")
                    if val:
                        urls.append(val)
        # De-dup, preserve order.
        seen: set[str] = set()
        return [u for u in urls if not (u in seen or seen.add(u))]

    def _source_headers(self) -> dict[str, str]:
        """Identification / QoS hints for the gateway (never secrets)."""
        h = {"X-Source": os.environ.get(SOURCE_ENV, DEFAULT_SOURCE)}
        prio = os.environ.get(PRIORITY_ENV, "")
        if prio:
            h["X-Priority"] = prio
        return h

    def discover_gateways(self) -> int:
        """Probe configured gateways and add the reachable ones as hosts.

        Returns the number added. Idempotent-ish: never adds a duplicate
        URL already present. Safe to call with nothing configured (no-op).
        """
        existing = {h.url for h in self._hosts}
        added = 0
        for url in self._gateway_urls():
            if url in existing:
                continue
            host = self._probe_gateway(url)
            if host is not None:
                self._hosts.append(host)
                existing.add(url)
                added += 1
        # Keep preference order stable after a late add.
        self._hosts.sort(key=lambda h: (
            self._BACKEND_RANK.get(h.backend, 9), h.latency_ms))
        return added

    def _probe_gateway(self, url: str) -> FleetHost | None:
        """GET /health -> a gateway FleetHost with its healthy-node count."""
        import urllib.request
        try:
            t0 = time.monotonic()
            req = urllib.request.Request(
                f"{url}/health", headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as resp:
                if resp.status != 200:
                    return None
                latency = (time.monotonic() - t0) * 1000
                body = resp.read().decode()
        except Exception:
            return None

        healthy = 0
        try:
            data = json.loads(body) if body else {}
            healthy = int(data.get("backends", {}).get("healthy", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Optional: list models the cluster advertises (best-effort).
        models: list[str] = []
        try:
            req = urllib.request.Request(
                f"{url}/v1/models", headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as resp:
                mdata = json.loads(resp.read().decode())
                models = mdata.get("models", []) or [
                    m.get("id", "") for m in mdata.get("data", [])]
        except Exception:
            pass

        name = url.split("//", 1)[-1]
        return FleetHost(url=url, name=name, models=models,
                         latency_ms=latency, backend="gateway",
                         backends_connected=healthy)

    def chat(
        self, model: str, prompt: str, images: list[str] | None = None,
        timeout: float = 30.0,
    ) -> str:
        """Send a chat request, optionally with images (base64-encoded).

        Automatically uses the correct API format based on host backend:
        - llama-server: /v1/chat/completions (OpenAI-compatible)
        - ollama: /api/chat (legacy)

        Args:
            model: Model name (e.g. "qwen2.5:7b").
            prompt: The text prompt to send.
            images: Optional list of base64-encoded image strings.
            timeout: Request timeout in seconds.

        Returns:
            Response text from the model, or empty string on failure.
        """
        import urllib.request

        host = self.best_host(model)
        if host is None:
            # Fallback: try any available host
            host = self._hosts[0] if self._hosts else None
        if host is None:
            return ""

        message: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            message["images"] = images

        if host.backend == "gateway":
            return self._gateway_chat(host, model, [message], timeout)
        if host.backend == "llama-server":
            # OpenAI-compatible API
            payload = json.dumps({
                "model": model,
                "messages": [message],
                "max_tokens": 1024,
            }).encode()
            endpoint = "/v1/chat/completions"
        else:
            # Legacy ollama API
            payload = json.dumps({
                "model": model,
                "messages": [message],
                "stream": False,
            }).encode()
            endpoint = "/api/chat"

        try:
            req = urllib.request.Request(
                f"{host.url}{endpoint}",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                if host.backend == "llama-server":
                    # OpenAI format: {"choices": [{"message": {"content": "..."}}]}
                    choices = data.get("choices", [])
                    return choices[0]["message"]["content"] if choices else ""
                else:
                    # Ollama format: {"message": {"content": "..."}}
                    return data.get("message", {}).get("content", "")
        except Exception:
            return ""

    def generate(
        self, model: str, prompt: str, timeout: float = 30.0,
    ) -> str:
        """Send a generate request to the best host.

        For llama-server: converts prompt to chat format (/v1/chat/completions).
        For ollama: uses /api/generate (legacy).
        """
        import urllib.request

        host = self.best_host(model)
        if host is None:
            host = self._hosts[0] if self._hosts else None
        if host is None:
            return ""

        if host.backend == "gateway":
            return self._gateway_generate(host, model, prompt, timeout)
        if host.backend == "llama-server":
            # llama-server doesn't have /api/generate — use chat completions
            return self.chat(model, prompt, timeout=timeout)

        # Legacy ollama /api/generate
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
        }).encode()

        try:
            req = urllib.request.Request(
                f"{host.url}/api/generate",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                return data.get("response", "")
        except Exception:
            return ""

    def _gateway_chat(self, host: FleetHost, model: str,
                      messages: list[dict[str, Any]], timeout: float) -> str:
        """POST /v1/chat — gateway chat API ({"message": {"content": ...}})."""
        import urllib.request
        payload = json.dumps({
            "model": model, "messages": messages, "stream": False,
        }).encode()
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json", **self._source_headers()}
        try:
            req = urllib.request.Request(
                f"{host.url}/v1/chat", data=payload, headers=headers,
                method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            if "error" in data:
                return ""
            msg = data.get("message", {})
            return msg.get("content", "") if isinstance(msg, dict) else str(msg)
        except Exception:
            return ""

    def _gateway_generate(self, host: FleetHost, model: str,
                          prompt: str, timeout: float) -> str:
        """POST /v1/generate — gateway completion API ({"response": ...})."""
        import urllib.request
        payload = json.dumps({
            "model": model, "prompt": prompt, "stream": False,
        }).encode()
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json", **self._source_headers()}
        try:
            req = urllib.request.Request(
                f"{host.url}/v1/generate", data=payload, headers=headers,
                method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            if "error" in data:
                return ""
            return data.get("response", "")
        except Exception:
            return ""

    def status(self) -> str:
        """Return a human-readable status string."""
        if not self._hosts:
            return "0 hosts (no LLM servers found)"
        parts = []
        for h in self._hosts:
            models_str = ", ".join(h.models[:3])
            if len(h.models) > 3:
                models_str += f" +{len(h.models) - 3}"
            parts.append(f"{h.name}[{h.backend}]({models_str})")
        return f"{len(self._hosts)} host(s): {'; '.join(parts)}"


# Backward compatibility alias
OllamaFleet = LLMFleet
