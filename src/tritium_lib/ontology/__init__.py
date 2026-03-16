# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.ontology — Formal ontology schema and registry.

Defines typed entity types, relationship types, interfaces, and
properties that form the semantic backbone of the Tritium system.
"""

from .registry import OntologyRegistry, OntologyValidationError
from .schema import (
    Cardinality,
    DataType,
    EntityType,
    InterfaceDef,
    OntologySchema,
    PropertyDef,
    RelationshipType,
    TRITIUM_ONTOLOGY,
    TypeStatus,
)

__all__ = [
    "Cardinality",
    "DataType",
    "EntityType",
    "InterfaceDef",
    "OntologyRegistry",
    "OntologySchema",
    "OntologyValidationError",
    "PropertyDef",
    "RelationshipType",
    "TRITIUM_ONTOLOGY",
    "TypeStatus",
]
