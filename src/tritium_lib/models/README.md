# Tritium-Lib Models

250+ Pydantic v2 models that serve as the canonical data contracts across the entire Tritium ecosystem.

## Model Files

All models are exported from `__init__.py`. Key domains:

- `device.py` — DeviceInfo, DeviceStatus, HeartbeatPayload
- `firmware.py` — FirmwareVersion, OTARequest
- `mesh.py` — MeshPeer, MeshMessage
- `ble.py` — BLESighting, BLEDevice
- `alert.py` — Alert, AlertLevel, AlertRule
- `cot.py` — CursorOnTarget models
- `topology.py` — NetworkTopology, NetworkLink, NetworkNode, PeerQuality
- `sensor.py` — SensorReading, SensorConfig
- `acoustic.py` — AcousticEvent, AcousticClassification
- `dossier.py` — Dossier, DossierEntry
- `federation.py` — FederatedSite, SharedTarget
- `behavior.py` — BehaviorPattern, BehaviorAnomaly
- `terrain.py` — TerrainPoint, CoverageGrid, RFPropagation
- `scenario.py` — TacticalScenario, ScenarioActor, ScenarioEvent
- `comms.py` — CommChannel, ChannelType, ChannelStatus
- `capability.py` — DeviceCapability, CapabilityAdvertisement

## Convention

- All models use Pydantic v2 (`model_config` not `class Config`)
- All public models are listed in `__all__` in `__init__.py`
- Changing a model here requires updating consumers in tritium-edge and tritium-sc
