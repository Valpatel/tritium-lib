# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Ontology registry — runtime lookup and validation for entity and
relationship types defined in the schema."""

from typing import Any

from .schema import (
    Cardinality,
    EntityType,
    InterfaceDef,
    OntologySchema,
    RelationshipType,
)


class OntologyValidationError(Exception):
    """Raised when entity or relationship data fails ontology validation."""


class OntologyRegistry:
    """In-memory registry of ontology types loaded from an OntologySchema.

    Usage::

        from tritium_lib.ontology.schema import TRITIUM_ONTOLOGY

        registry = OntologyRegistry()
        registry.load_schema(TRITIUM_ONTOLOGY)
        device_type = registry.get_entity_type("device")
    """

    def __init__(self) -> None:
        self._entity_types: dict[str, EntityType] = {}
        self._relationship_types: dict[str, RelationshipType] = {}
        self._interfaces: dict[str, InterfaceDef] = {}
        self._version: str = ""

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_schema(self, schema: OntologySchema) -> None:
        """Register all types from an :class:`OntologySchema`."""
        self._entity_types = dict(schema.entity_types)
        self._relationship_types = dict(schema.relationship_types)
        self._interfaces = dict(schema.interfaces)
        self._version = schema.version

    @property
    def version(self) -> str:
        return self._version

    # ------------------------------------------------------------------
    # Entity type queries
    # ------------------------------------------------------------------

    def get_entity_type(self, api_name: str) -> EntityType:
        """Return the :class:`EntityType` for *api_name* or raise KeyError."""
        try:
            return self._entity_types[api_name]
        except KeyError:
            raise KeyError(
                f"Unknown entity type '{api_name}'. "
                f"Available: {sorted(self._entity_types)}"
            )

    def list_entity_types(self) -> list[EntityType]:
        """Return all registered entity types sorted by api_name."""
        return sorted(self._entity_types.values(), key=lambda t: t.api_name)

    # ------------------------------------------------------------------
    # Relationship type queries
    # ------------------------------------------------------------------

    def get_relationship_type(self, api_name: str) -> RelationshipType:
        """Return the :class:`RelationshipType` for *api_name* or raise."""
        try:
            return self._relationship_types[api_name]
        except KeyError:
            raise KeyError(
                f"Unknown relationship type '{api_name}'. "
                f"Available: {sorted(self._relationship_types)}"
            )

    def list_relationship_types(self) -> list[RelationshipType]:
        """Return all registered relationship types sorted by api_name."""
        return sorted(
            self._relationship_types.values(), key=lambda t: t.api_name
        )

    # ------------------------------------------------------------------
    # Interface queries
    # ------------------------------------------------------------------

    def get_interface(self, api_name: str) -> InterfaceDef:
        """Return the :class:`InterfaceDef` for *api_name* or raise."""
        try:
            return self._interfaces[api_name]
        except KeyError:
            raise KeyError(
                f"Unknown interface '{api_name}'. "
                f"Available: {sorted(self._interfaces)}"
            )

    def get_interface_implementors(self, interface_name: str) -> list[EntityType]:
        """Return all entity types that implement *interface_name*."""
        # Validate that the interface exists.
        self.get_interface(interface_name)
        return sorted(
            [
                et
                for et in self._entity_types.values()
                if interface_name in et.interfaces
            ],
            key=lambda t: t.api_name,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_entity(
        self, entity_type: str, properties: dict[str, Any]
    ) -> bool:
        """Validate *properties* against the schema for *entity_type*.

        Returns ``True`` on success.  Raises :class:`OntologyValidationError`
        with details on failure.
        """
        et = self.get_entity_type(entity_type)

        # Check required properties are present.
        missing = [
            pdef.name
            for pdef in et.properties.values()
            if pdef.required and pdef.name not in properties
        ]
        if missing:
            raise OntologyValidationError(
                f"Entity '{entity_type}' missing required properties: "
                f"{sorted(missing)}"
            )

        # Check no unknown properties.
        unknown = set(properties) - set(et.properties)
        if unknown:
            raise OntologyValidationError(
                f"Entity '{entity_type}' has unknown properties: "
                f"{sorted(unknown)}"
            )

        return True

    def validate_schema(self) -> list[str]:
        """Validate internal consistency of the loaded schema.

        Checks for:
        - Entity types that reference non-existent interfaces
        - Relationship types that reference non-existent entity types
        - Interfaces whose required_properties don't exist on implementors

        Returns a list of error messages (empty means valid).
        """
        errors: list[str] = []

        # Check interface references on entity types
        for et in self._entity_types.values():
            for iface_name in et.interfaces:
                if iface_name not in self._interfaces:
                    errors.append(
                        f"Entity '{et.api_name}' references unknown "
                        f"interface '{iface_name}'"
                    )
                else:
                    iface = self._interfaces[iface_name]
                    for prop in iface.required_properties:
                        if prop not in et.properties:
                            errors.append(
                                f"Entity '{et.api_name}' implements "
                                f"'{iface_name}' but missing required "
                                f"property '{prop}'"
                            )

        # Check relationship endpoint references
        for rt in self._relationship_types.values():
            if rt.from_type not in self._entity_types:
                errors.append(
                    f"Relationship '{rt.api_name}' from_type "
                    f"'{rt.from_type}' is not a known entity type"
                )
            if rt.to_type not in self._entity_types:
                errors.append(
                    f"Relationship '{rt.api_name}' to_type "
                    f"'{rt.to_type}' is not a known entity type"
                )

        return errors

    def validate_relationship(
        self,
        rel_type: str,
        from_entity_type: str,
        to_entity_type: str,
    ) -> bool:
        """Validate that a relationship of *rel_type* can connect the given
        entity types.

        Returns ``True`` on success.  Raises :class:`OntologyValidationError`
        on failure.
        """
        rt = self.get_relationship_type(rel_type)

        if rt.from_type != from_entity_type:
            raise OntologyValidationError(
                f"Relationship '{rel_type}' expects from_type='{rt.from_type}'"
                f" but got '{from_entity_type}'"
            )
        if rt.to_type != to_entity_type:
            raise OntologyValidationError(
                f"Relationship '{rel_type}' expects to_type='{rt.to_type}'"
                f" but got '{to_entity_type}'"
            )

        return True
