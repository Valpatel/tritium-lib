# tritium_lib.conf

**One example config file — no code.** This directory ships a single template,
`llm-fleet.conf.example`, documenting how to point Tritium at your LLM
inference hosts. It is a reference artifact, not a Python package (there is no
`__init__.py`, nothing imports from here).

**Where you are:** `tritium-lib/src/tritium_lib/conf/`
**Parent:** [`../`](../) — the tritium-lib package map

## What it's for

`tritium_lib.inference` (`LLMFleet`/`OllamaFleet`, host pools + chat clients)
needs to know where your GPU/inference hosts are. Rather than hardcode private
LAN hostnames into the repo, the loader reads a **gitignored** `llm-fleet.conf`
that you create by copying this `.example`. Config can also come from
environment variables, which win for ephemeral/CI use.

## The file

| File | What it documents |
|------|-------------------|
| `llm-fleet.conf.example` | The resolution order and syntax. Copy to `llm-fleet.conf` (same dir) and edit; the real file is gitignored so your hosts never land in the repo. |

**Resolution order (best first), per the template:**

1. `gateway = URL` — the live-network inference gateway (preferred; it
   load-balances the whole GPU cluster behind one endpoint;
   `/health`, `/v1/generate`, `/v1/chat`).
2. Direct `host` / `host:port` lines — llama-server tiers (OpenAI-compatible,
   ports 8081–8083).
3. Ollama (`host:11434`) — legacy fallback for small/local tests.

**Environment overrides** (win for CI): `LLM_GATEWAY_URL`, `LLM_HOSTS`,
`LLM_SOURCE`, `LLM_PRIORITY`.

This matches the project rule: use llama-server directly, never hardcode IPs,
graceful degradation — the LLM enhances, never blocks.

## How it's consumed (verified 2026-07-11)

No Python imports from here (it holds no code). The `.example` is copied by an
operator to the gitignored `llm-fleet.conf`, which `tritium_lib.inference`
reads at runtime. Reference-only.

## Related

- [../inference/](../inference/) — the package that reads `llm-fleet.conf` (LLM host pools, `ModelRouter`)
- [../config/](../config/) — the typed settings package (`TritiumBaseSettings`, env/`.env`/TOML) for everything *other* than the LLM fleet
