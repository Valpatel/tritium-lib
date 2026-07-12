# AGENTS.md — tritium-lib

A **submodule** of the [`Valpatel/tritium`](https://github.com/Valpatel/tritium)
superproject. Active branch: `dev`. The parent repo pins this repo via a gitlink
and bumps it after your commits land here.

> **Read the parent's cross-repo manual first:**
> <https://github.com/Valpatel/tritium/blob/main/AGENTS.md> — commit order
> (lib → edge → sc → addons), push order (submodules before the parent), the
> pre-push privacy + markdown gate, and the merge hazards. Getting those wrong
> breaks the superproject for everyone who clones it.

## What belongs here

Reusable models / algorithms / wire-contracts that import cleanly on a bare
aarch64 Jetson with only light deps (numpy / pydantic / opencv). **Not** a web
framework, **not** heavy simulator/tool runtimes, **not** on-robot ROS2. If it
needs `import isaacsim` / `pxr` / `rospy` / a hard `torch`, it does not belong
here — see the placement rules in the parent AGENTS.md.

Keeping lib importable on the robot brain (light + framework-free) is the
invariant that lets the fleet scale. Actively extract reusable code from sc into
lib; never add framework deps to do it.

## Test

```bash
pytest tests/            # needs the `testing` extra (httpx2)
```

## Style

No Co-Authored-By. AGPL-3.0 / Matthew Valancy / Valpatel Software LLC. Work on
`dev`; `main` advances only via a reviewed dev→main PR.
