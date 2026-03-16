"""Tritium graph database layer — embedded KuzuDB ontology store."""

try:
    from tritium_lib.graph.store import TritiumGraph
    __all__ = ["TritiumGraph"]
except ImportError:
    __all__ = []
