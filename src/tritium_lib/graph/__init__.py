"""Tritium graph database layer — embedded KuzuDB ontology store.

Status (Wave B item B-2 — truth-in-advertising)
-----------------------------------------------
This package is **shelfware**.  ``TritiumGraph`` (see ``store.py``) is a
working KuzuDB wrapper with schema creation, node/edge CRUD, and Cypher
query support, and it is exercised by the unit tests in
``tritium-lib/tests/graph/`` and the standalone demos in
``tritium_lib.graph.demos`` (e.g. ``graph_demo.py``).

It is **not** wired to the live Tritium SC ontology API.  The runtime
``/api/v1/ontology/*`` endpoints in ``tritium-sc`` are served by an
in-memory adapter over ``TargetTracker``, ``DossierStore``, and
``BleStore`` — see
``tritium-sc/src/app/routers/ontology.py`` for the actual backend.

Integrating ``TritiumGraph`` with the live API — replacing the in-memory
adapter with a persistent property graph that survives restarts and
supports Cypher traversals — is a **separate, not-yet-scheduled
workstream**.  Until that work happens:

* Importing this module is safe; ``TritiumGraph`` is usable directly.
* ``kuzu`` is an optional dependency — install with
  ``pip install 'tritium-lib[graph]'`` if you need the wrapper.
* Anything reading the SC ontology API is reading in-memory store state,
  not a graph database.  Do not assume Cypher semantics.
"""

try:
    from tritium_lib.graph.store import TritiumGraph
    __all__ = ["TritiumGraph"]
except ImportError:
    __all__ = []
