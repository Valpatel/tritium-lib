# tritium_lib.inference

Shared LLM inference for the whole platform — Amy's cognition, robot
thinking, sitrep narration, classification, anomaly detection. One fleet
abstraction, three backend tiers, graceful degradation.

## Backends (preference order)

| Tier | Backend | API | When |
|------|---------|-----|------|
| 1 | **gateway** | `POST /v1/generate`, `POST /v1/chat`, `GET /health` | The live-network path — an inference gateway that load-balances a cluster of GPU servers behind one endpoint. Preferred whenever configured. |
| 2 | **llama-server** | `POST /v1/chat/completions` (OpenAI-compatible) | Direct llama-server tiers, ports 8081-8083. |
| 3 | **ollama** | `POST /api/chat`, `POST /api/generate` | Legacy fallback / quick local tests. |

`LLMFleet` (`fleet.py`) discovers all reachable backends, then
`best_host(model)` picks **gateway > llama-server > ollama**, lowest
latency within a tier. A gateway can serve any model (it routes
internally); direct hosts only serve the models they list. Nothing is
required — with no gateway and no llama-server, it degrades to ollama;
with nothing at all, calls return `""` and callers fall back gracefully.

## Configuration

No hosts, ports, or tokens are baked into source. Configure via env
(wins) or `conf/llm-fleet.conf` (gitignored — copy from
`conf/llm-fleet.conf.example`):

```bash
LLM_GATEWAY_URL=http://gateway-host:PORT   # the production gateway(s)
LLM_HOSTS=host-a,host-b:8082               # direct llama-server hosts
LLM_SOURCE=tritium                         # X-Source header to the gateway
LLM_PRIORITY=normal                        # optional X-Priority QoS hint
LLM_FLEET_CONF=/path/to/llm-fleet.conf     # override the conf path
```

Discovery also probes localhost (8081-8083, 11434), the conf file,
`OLLAMA_HOSTS` (legacy), Tailscale peers, and the local /24 subnet.

## Usage

```python
from tritium_lib.inference.fleet import LLMFleet

fleet = LLMFleet()                      # auto-discovers gateway + hosts
print(fleet.status())                   # what's reachable
reply = fleet.chat("qwen2.5:7b", "Sitrep?")
text  = fleet.generate("qwen2.5:7b", "One line summary:")
```

The gateway sends an `X-Source` header (default `tritium`) so the
upstream can attribute / prioritize traffic — it is an identifier, never
a secret.

## Files

| File | Purpose |
|------|---------|
| `fleet.py` | `LLMFleet` — multi-backend discovery, selection, chat/generate |
| `llm_client.py` | Single-host helpers (`ollama_chat`, `llama_server_chat`) |
| `model_router.py` | Model selection by task/capability |
