// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Sensor Data Bridge — generates synthetic sensor data from simulation entities.
 *
 * "The fake city IS the real city" — every simulated entity produces the same
 * sensor data format as real sensors. NPC phones emit BLE advertisements,
 * vehicles emit TPMS signals, camera zones produce YOLO detections.
 *
 * This bridges the city sim to the tritium-sc command center via WebSocket or MQTT,
 * so simulated entities appear as real targets on the tactical map.
 *
 * Data formats match:
 * - BLE sightings: { target_id: "ble_{mac}", rssi, manufacturer, device_class }
 * - TPMS signals: { target_id: "tpms_{id}", tire_pressure, temperature }
 * - Camera detections: { target_id: "det_{class}_{n}", bbox, confidence, class }
 *
 * This is a pure data module — no rendering dependencies.
 */

// ============================================================
// SENSOR BRIDGE
// ============================================================

export class SensorBridge {
    /**
     * @param {Object} config
     * @param {string} config.deviceId - Simulated sensor device ID (default 'city3d_sim')
     * @param {string} config.site - Site identifier (default 'sim')
     * @param {number} config.bleInterval - BLE scan interval in seconds (default 2)
     * @param {number} config.cameraInterval - Camera detection interval (default 1)
     * @param {function} [config.onSighting] - Callback for BLE sightings
     * @param {function} [config.onDetection] - Callback for camera detections
     * @param {WebSocket} [config.ws] - WebSocket connection to SC
     */
    constructor(config = {}) {
        this.deviceId = config.deviceId || 'city3d_sim';
        this.site = config.site || 'sim';
        this.bleInterval = config.bleInterval || 2;
        this.cameraInterval = config.cameraInterval || 1;
        this.onSighting = config.onSighting || null;
        this.onDetection = config.onDetection || null;
        this.ws = config.ws || null;
        this.bleTimer = 0;
        this.cameraTimer = 0;
        this.enabled = false;
    }

    /**
     * Enable the sensor bridge.
     */
    enable() { this.enabled = true; }

    /**
     * Disable the sensor bridge.
     */
    disable() { this.enabled = false; }

    /**
     * Connect to tritium-sc via WebSocket.
     *
     * @param {string} url - WebSocket URL (e.g., 'ws://localhost:8000/ws')
     */
    connect(url) {
        try {
            this.ws = new WebSocket(url);
            this.ws.onopen = () => console.log('[SensorBridge] Connected to', url);
            this.ws.onclose = () => console.log('[SensorBridge] Disconnected');
            this.ws.onerror = (e) => console.log('[SensorBridge] Error:', e);
            this.enabled = true;
        } catch (e) {
            console.log('[SensorBridge] Connection failed:', e);
        }
    }

    /**
     * Tick the bridge — generate sensor data from current entity state.
     *
     * @param {number} dt - Time step (seconds)
     * @param {Array} npcs - NPC entities with { id, x, z, bleMac, type }
     * @param {Array} vehicles - Vehicle entities with { id, x, z, tpmsId, vehicleType }
     */
    tick(dt, npcs, vehicles) {
        if (!this.enabled) return;

        this.bleTimer += dt;
        this.cameraTimer += dt;

        // BLE sightings (from NPC phones)
        if (this.bleTimer >= this.bleInterval) {
            this.bleTimer = 0;
            for (const npc of npcs) {
                if (!npc.bleMac) continue;
                const sighting = {
                    type: 'ble_sighting',
                    device_id: this.deviceId,
                    timestamp: new Date().toISOString(),
                    target_id: `ble_${npc.bleMac}`,
                    mac: npc.bleMac,
                    rssi: -40 - Math.floor(Math.random() * 30), // -40 to -70
                    manufacturer: 'Apple',
                    device_class: 'phone',
                    position: { x: npc.x, z: npc.z },
                };
                this._emit('sighting', sighting);
            }
        }

        // Camera detections (YOLO-style)
        if (this.cameraTimer >= this.cameraInterval) {
            this.cameraTimer = 0;

            // Detect NPCs as "person"
            for (let i = 0; i < npcs.length; i++) {
                const npc = npcs[i];
                const detection = {
                    type: 'detection',
                    device_id: this.deviceId,
                    camera_id: `${this.site}_cam_0`,
                    timestamp: new Date().toISOString(),
                    target_id: `det_person_${i}`,
                    class: 'person',
                    confidence: 0.85 + Math.random() * 0.14,
                    bbox: { x: npc.x - 0.5, z: npc.z - 0.5, w: 1, h: 2 },
                    position: { x: npc.x, z: npc.z },
                };
                this._emit('detection', detection);
            }

            // Detect vehicles
            for (let i = 0; i < vehicles.length; i++) {
                const v = vehicles[i];
                const detection = {
                    type: 'detection',
                    device_id: this.deviceId,
                    camera_id: `${this.site}_cam_0`,
                    timestamp: new Date().toISOString(),
                    target_id: `det_vehicle_${i}`,
                    class: v.vehicleType === 'police' ? 'police_car' :
                           v.vehicleType === 'ambulance' ? 'ambulance' : 'car',
                    confidence: 0.90 + Math.random() * 0.09,
                    bbox: { x: v.x - 1, z: v.z - 2, w: 2, h: 4 },
                    position: { x: v.x, z: v.z },
                };
                this._emit('detection', detection);
            }

            // TPMS from vehicles
            for (const v of vehicles) {
                if (!v.tpmsId) continue;
                const tpms = {
                    type: 'tpms_signal',
                    device_id: this.deviceId,
                    timestamp: new Date().toISOString(),
                    target_id: `tpms_${v.tpmsId}`,
                    tire_pressure: 32 + Math.random() * 4, // PSI
                    temperature: 20 + Math.random() * 15, // Celsius
                    position: { x: v.x, z: v.z },
                };
                this._emit('sighting', tpms);
            }
        }
    }

    _emit(eventType, data) {
        // Callback
        if (eventType === 'sighting' && this.onSighting) {
            this.onSighting(data);
        }
        if (eventType === 'detection' && this.onDetection) {
            this.onDetection(data);
        }

        // WebSocket
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }

    /**
     * Get stats for debug display.
     */
    getStats() {
        return {
            enabled: this.enabled,
            connected: this.ws && this.ws.readyState === WebSocket.OPEN,
            deviceId: this.deviceId,
        };
    }
}

/**
 * Generate a deterministic BLE MAC address from an NPC ID.
 *
 * @param {string} npcId - NPC identifier
 * @returns {string} MAC address in format "AA:BB:CC:DD:EE:FF"
 */
export function generateBleMac(npcId) {
    let hash = 0;
    for (let i = 0; i < npcId.length; i++) {
        hash = ((hash << 5) - hash + npcId.charCodeAt(i)) | 0;
    }
    const bytes = [];
    for (let i = 0; i < 6; i++) {
        bytes.push(((hash >>> (i * 4)) & 0xff).toString(16).padStart(2, '0'));
    }
    return bytes.join(':').toUpperCase();
}

/**
 * Generate a deterministic TPMS sensor ID from a vehicle ID.
 *
 * @param {string} vehicleId
 * @returns {string} TPMS ID like "TPMS_A1B2C3"
 */
export function generateTpmsId(vehicleId) {
    let hash = 0x5A3C;
    for (let i = 0; i < vehicleId.length; i++) {
        hash = ((hash << 3) + hash + vehicleId.charCodeAt(i)) | 0;
    }
    return `TPMS_${(hash >>> 0).toString(16).toUpperCase().slice(0, 6)}`;
}
