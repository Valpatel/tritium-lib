# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.ontology — schema definitions and registry."""

import pytest

from tritium_lib.ontology import (
    Cardinality,
    DataType,
    EntityType,
    InterfaceDef,
    OntologyRegistry,
    OntologySchema,
    OntologyValidationError,
    PropertyDef,
    RelationshipType,
    TRITIUM_ONTOLOGY,
    TypeStatus,
)


# -----------------------------------------------------------------------
# Schema structure tests
# -----------------------------------------------------------------------

class TestTritiumOntologySchema:
    """Verify the canonical TRITIUM_ONTOLOGY constant is well-formed."""

    def test_version(self) -> None:
        assert TRITIUM_ONTOLOGY.version == "1.0.0"

    def test_has_10_entity_types(self) -> None:
        assert len(TRITIUM_ONTOLOGY.entity_types) == 10

    def test_entity_type_names(self) -> None:
        expected = {
            "device", "mesh_node", "edge_unit",
            "sighting", "track",
            "chat_message", "command",
            "classification", "investigation", "zone",
        }
        assert set(TRITIUM_ONTOLOGY.entity_types.keys()) == expected

    def test_has_12_relationship_types(self) -> None:
        # detected_by, sighting_of, part_of_track,
        # connected_to, mesh_peer, managed_by,
        # located_at, within_zone,
        # classified_as, related_to, subject_of,
        # sent_by
        assert len(TRITIUM_ONTOLOGY.relationship_types) == 12

    def test_has_3_interfaces(self) -> None:
        assert set(TRITIUM_ONTOLOGY.interfaces.keys()) == {
            "trackable", "identifiable", "classifiable",
        }

    def test_all_entity_types_have_primary_key(self) -> None:
        for name, et in TRITIUM_ONTOLOGY.entity_types.items():
            assert et.primary_key_field, f"{name} missing primary_key_field"
            assert et.primary_key_field in et.properties, (
                f"{name}: pk '{et.primary_key_field}' not in properties"
            )

    def test_all_entity_types_active(self) -> None:
        for et in TRITIUM_ONTOLOGY.entity_types.values():
            assert et.status == TypeStatus.ACTIVE

    def test_relationship_endpoints_exist(self) -> None:
        """Every relationship's from_type and to_type must be a known entity."""
        entity_names = set(TRITIUM_ONTOLOGY.entity_types)
        for name, rt in TRITIUM_ONTOLOGY.relationship_types.items():
            assert rt.from_type in entity_names, (
                f"rel '{name}' from_type '{rt.from_type}' not found"
            )
            assert rt.to_type in entity_names, (
                f"rel '{name}' to_type '{rt.to_type}' not found"
            )

    def test_interface_properties_exist_on_implementors(self) -> None:
        """Each entity claiming an interface must have all required props."""
        for et in TRITIUM_ONTOLOGY.entity_types.values():
            for iface_name in et.interfaces:
                iface = TRITIUM_ONTOLOGY.interfaces[iface_name]
                for prop_name in iface.required_properties:
                    assert prop_name in et.properties, (
                        f"Entity '{et.api_name}' implements '{iface_name}' "
                        f"but missing property '{prop_name}'"
                    )

    def test_device_entity(self) -> None:
        device = TRITIUM_ONTOLOGY.entity_types["device"]
        assert device.display_name == "Device"
        assert device.primary_key_field == "mac_address"
        assert "trackable" in device.interfaces
        assert "identifiable" in device.interfaces
        assert "classifiable" in device.interfaces
        assert device.properties["mac_address"].required is True

    def test_sighting_entity(self) -> None:
        s = TRITIUM_ONTOLOGY.entity_types["sighting"]
        assert s.primary_key_field == "sighting_id"
        required = [p.name for p in s.properties.values() if p.required]
        assert "sighting_id" in required
        assert "mac" in required
        assert "timestamp" in required

    def test_detected_by_relationship(self) -> None:
        r = TRITIUM_ONTOLOGY.relationship_types["detected_by"]
        assert r.from_type == "sighting"
        assert r.to_type == "edge_unit"
        assert r.cardinality == Cardinality.ONE


# -----------------------------------------------------------------------
# PropertyDef tests
# -----------------------------------------------------------------------

class TestPropertyDef:
    def test_defaults(self) -> None:
        p = PropertyDef(name="foo", data_type=DataType.STRING)
        assert p.required is False
        assert p.default is None

    def test_all_data_types(self) -> None:
        for dt in DataType:
            p = PropertyDef(name="x", data_type=dt)
            assert p.data_type == dt


# -----------------------------------------------------------------------
# Registry tests
# -----------------------------------------------------------------------

class TestOntologyRegistry:
    @pytest.fixture()
    def registry(self) -> OntologyRegistry:
        r = OntologyRegistry()
        r.load_schema(TRITIUM_ONTOLOGY)
        return r

    def test_version(self, registry: OntologyRegistry) -> None:
        assert registry.version == "1.0.0"

    def test_get_entity_type(self, registry: OntologyRegistry) -> None:
        et = registry.get_entity_type("device")
        assert et.api_name == "device"

    def test_get_entity_type_unknown(self, registry: OntologyRegistry) -> None:
        with pytest.raises(KeyError, match="Unknown entity type"):
            registry.get_entity_type("nonexistent")

    def test_list_entity_types(self, registry: OntologyRegistry) -> None:
        types = registry.list_entity_types()
        assert len(types) == 10
        # Sorted by api_name
        names = [t.api_name for t in types]
        assert names == sorted(names)

    def test_get_relationship_type(self, registry: OntologyRegistry) -> None:
        rt = registry.get_relationship_type("detected_by")
        assert rt.from_type == "sighting"

    def test_get_relationship_type_unknown(self, registry: OntologyRegistry) -> None:
        with pytest.raises(KeyError, match="Unknown relationship type"):
            registry.get_relationship_type("nope")

    def test_list_relationship_types(self, registry: OntologyRegistry) -> None:
        types = registry.list_relationship_types()
        assert len(types) == 12

    def test_get_interface(self, registry: OntologyRegistry) -> None:
        iface = registry.get_interface("trackable")
        assert "latitude" in iface.required_properties

    def test_get_interface_unknown(self, registry: OntologyRegistry) -> None:
        with pytest.raises(KeyError, match="Unknown interface"):
            registry.get_interface("nope")

    def test_get_interface_implementors_trackable(
        self, registry: OntologyRegistry
    ) -> None:
        impls = registry.get_interface_implementors("trackable")
        names = [t.api_name for t in impls]
        assert "device" in names
        assert "mesh_node" in names
        assert "edge_unit" in names
        assert "track" in names
        # chat_message should NOT be trackable
        assert "chat_message" not in names

    def test_get_interface_implementors_classifiable(
        self, registry: OntologyRegistry
    ) -> None:
        impls = registry.get_interface_implementors("classifiable")
        names = [t.api_name for t in impls]
        assert "device" in names
        assert "track" in names
        assert len(names) == 2

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def test_validate_entity_valid(self, registry: OntologyRegistry) -> None:
        assert registry.validate_entity("device", {
            "mac_address": "AA:BB:CC:DD:EE:FF",
            "name": "Test",
        })

    def test_validate_entity_missing_required(
        self, registry: OntologyRegistry
    ) -> None:
        with pytest.raises(OntologyValidationError, match="missing required"):
            registry.validate_entity("device", {"name": "No MAC"})

    def test_validate_entity_unknown_property(
        self, registry: OntologyRegistry
    ) -> None:
        with pytest.raises(OntologyValidationError, match="unknown properties"):
            registry.validate_entity("device", {
                "mac_address": "AA:BB:CC:DD:EE:FF",
                "bogus_field": 42,
            })

    def test_validate_entity_all_optional(
        self, registry: OntologyRegistry
    ) -> None:
        # zone only requires zone_id and name
        assert registry.validate_entity("zone", {
            "zone_id": "z1",
            "name": "HQ",
        })

    def test_validate_relationship_valid(
        self, registry: OntologyRegistry
    ) -> None:
        assert registry.validate_relationship(
            "detected_by", "sighting", "edge_unit"
        )

    def test_validate_relationship_wrong_from(
        self, registry: OntologyRegistry
    ) -> None:
        with pytest.raises(OntologyValidationError, match="from_type"):
            registry.validate_relationship(
                "detected_by", "device", "edge_unit"
            )

    def test_validate_relationship_wrong_to(
        self, registry: OntologyRegistry
    ) -> None:
        with pytest.raises(OntologyValidationError, match="to_type"):
            registry.validate_relationship(
                "detected_by", "sighting", "device"
            )

    # ------------------------------------------------------------------
    # Empty registry
    # ------------------------------------------------------------------

    def test_empty_registry(self) -> None:
        r = OntologyRegistry()
        assert r.version == ""
        assert r.list_entity_types() == []
        assert r.list_relationship_types() == []

    def test_load_custom_schema(self) -> None:
        """Registry can load a minimal custom schema."""
        schema = OntologySchema(
            version="0.1.0",
            entity_types={
                "widget": EntityType(
                    api_name="widget",
                    display_name="Widget",
                    primary_key_field="widget_id",
                    properties={
                        "widget_id": PropertyDef(
                            name="widget_id",
                            data_type=DataType.STRING,
                            required=True,
                        ),
                    },
                ),
            },
        )
        r = OntologyRegistry()
        r.load_schema(schema)
        assert len(r.list_entity_types()) == 1
        assert r.get_entity_type("widget").display_name == "Widget"


# -----------------------------------------------------------------------
# Serialization round-trip
# -----------------------------------------------------------------------

class TestSerialization:
    def test_schema_round_trip(self) -> None:
        """Schema can serialize to JSON and back."""
        data = TRITIUM_ONTOLOGY.model_dump()
        restored = OntologySchema.model_validate(data)
        assert restored.version == TRITIUM_ONTOLOGY.version
        assert set(restored.entity_types) == set(TRITIUM_ONTOLOGY.entity_types)
        assert (
            set(restored.relationship_types)
            == set(TRITIUM_ONTOLOGY.relationship_types)
        )

    def test_schema_json_round_trip(self) -> None:
        """Schema can serialize to JSON string and back."""
        json_str = TRITIUM_ONTOLOGY.model_dump_json()
        restored = OntologySchema.model_validate_json(json_str)
        assert len(restored.entity_types) == 10
