// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * CityWeather — time-of-day cycle and weather effects for city sim.
 *
 * Controls sky color, ambient/directional light intensity, fog density,
 * window emissive glow, and provides weather state for vehicle behavior.
 */

/**
 * @typedef {'clear'|'cloudy'|'rain'|'fog'} WeatherState
 */

// Sky gradient keyframes (hour → hex color)
const SKY_COLORS = [
    { hour: 0,   color: 0x050510 },  // midnight
    { hour: 4,   color: 0x050510 },  // pre-dawn
    { hour: 5.5, color: 0x1a1a3e },  // first light
    { hour: 6.5, color: 0xc4956a },  // dawn
    { hour: 8,   color: 0x87CEEB },  // morning
    { hour: 17,  color: 0x87CEEB },  // afternoon
    { hour: 18.5, color: 0xc4856a }, // dusk
    { hour: 19.5, color: 0x3a2a4e }, // twilight
    { hour: 21,  color: 0x0a0a18 },  // night
    { hour: 24,  color: 0x050510 },  // midnight wrap
];

function _lerpColor(a, b, t) {
    const ar = (a >> 16) & 0xFF, ag = (a >> 8) & 0xFF, ab = a & 0xFF;
    const br = (b >> 16) & 0xFF, bg = (b >> 8) & 0xFF, bb = b & 0xFF;
    const r = Math.round(ar + (br - ar) * t);
    const g = Math.round(ag + (bg - ag) * t);
    const bl = Math.round(ab + (bb - ab) * t);
    return (r << 16) | (g << 8) | bl;
}

function _getSkyColor(hour) {
    // Normalize hour to 0-24 range to handle wrapping
    hour = hour % 24;

    for (let i = 0; i < SKY_COLORS.length - 1; i++) {
        if (hour >= SKY_COLORS[i].hour && hour < SKY_COLORS[i + 1].hour) {
            const t = (hour - SKY_COLORS[i].hour) / (SKY_COLORS[i + 1].hour - SKY_COLORS[i].hour);
            return _lerpColor(SKY_COLORS[i].color, SKY_COLORS[i + 1].color, t);
        }
    }
    // Default to midnight color if no match (shouldn't happen after normalization)
    return SKY_COLORS[0].color;
}

export class CityWeather {
    constructor() {
        this.hour = 7;           // simulation hour (0-24)
        this.weather = 'clear';  // current weather state
        this.isNight = false;
        this.isDusk = false;
        this.isDawn = false;

        // Computed values (updated each tick)
        this.skyColor = 0x87CEEB;
        this.ambientIntensity = 0.7;
        this.sunIntensity = 1.0;
        this.fogDensity = 0.0015;
        this.windowEmissive = 0.15;
        this.headlightsOn = false;
        this.streetLightsOn = false;

        // Vehicle behavior modifiers
        this.speedMultiplier = 1.0;
        this.headwayMultiplier = 1.0;

        // Weather transition
        this._weatherTimer = 0;
        this._weatherDuration = 300 + Math.random() * 600;  // 5-15 min between changes
    }

    /**
     * Update weather state for this tick.
     * @param {number} hour — simulation hour (0-24)
     * @param {number} dt — delta time in seconds
     */
    update(hour, dt) {
        this.hour = hour;

        // Time of day classification
        this.isNight = hour >= 21 || hour < 5.5;
        this.isDusk = hour >= 17 && hour < 21;
        this.isDawn = hour >= 5.5 && hour < 8;

        // Sky color
        this.skyColor = _getSkyColor(hour);

        // Lighting
        const dayFactor = this.isNight ? 0.15 : this.isDusk || this.isDawn ? 0.5 : 1.0;
        const rainDim = this.weather === 'rain' ? 0.7 : this.weather === 'cloudy' ? 0.85 : 1.0;

        this.ambientIntensity = (0.2 + dayFactor * 0.5) * rainDim;
        this.sunIntensity = dayFactor * 1.2 * rainDim;

        // Window emissive
        this.windowEmissive = this.isNight ? 0.8 : this.isDusk ? 0.4 : 0.1;

        // Headlights and street lights
        this.headlightsOn = this.isNight || this.isDusk || (this.weather === 'fog');
        this.streetLightsOn = this.isNight || (this.isDusk && hour > 19);

        // Fog
        if (this.weather === 'fog') {
            this.fogDensity = 0.008;
        } else if (this.weather === 'rain') {
            this.fogDensity = 0.004;
        } else if (this.isNight) {
            this.fogDensity = 0.003;
        } else {
            this.fogDensity = 0.0015;
        }

        // Vehicle behavior
        if (this.weather === 'rain') {
            this.speedMultiplier = 0.8;   // 20% speed reduction
            this.headwayMultiplier = 1.3; // 30% more following distance
        } else if (this.weather === 'fog') {
            this.speedMultiplier = 0.7;
            this.headwayMultiplier = 1.5;
        } else {
            this.speedMultiplier = 1.0;
            this.headwayMultiplier = 1.0;
        }

        // Random weather transitions
        this._weatherTimer += dt;
        if (this._weatherTimer >= this._weatherDuration) {
            this._weatherTimer = 0;
            this._weatherDuration = 300 + Math.random() * 600;
            this._transitionWeather();
        }
    }

    /**
     * Random weather state transition.
     */
    _transitionWeather() {
        const transitions = {
            clear: ['clear', 'clear', 'cloudy'],         // mostly stays clear
            cloudy: ['clear', 'cloudy', 'rain', 'fog'],  // can go anywhere
            rain: ['rain', 'cloudy', 'clear'],            // rain usually clears
            fog: ['fog', 'cloudy', 'clear'],              // fog usually clears
        };
        const options = transitions[this.weather] || ['clear'];
        this.weather = options[Math.floor(Math.random() * options.length)];
    }

    /**
     * Apply weather effects to a Three.js scene.
     * @param {THREE.Scene} scene
     * @param {THREE.AmbientLight} ambient
     * @param {THREE.DirectionalLight} sun
     */
    applyToScene(scene, ambient, sun) {
        if (scene.background && typeof scene.background.setHex === 'function') {
            scene.background.setHex(this.skyColor);
        }
        if (scene.fog) {
            scene.fog.density = this.fogDensity;
        }
        if (ambient) {
            ambient.intensity = this.ambientIntensity;
        }
        if (sun) {
            sun.intensity = this.sunIntensity;
        }
    }

    /**
     * Get display string for HUD.
     */
    toString() {
        const timeStr = `${Math.floor(this.hour)}:${String(Math.floor((this.hour % 1) * 60)).padStart(2, '0')}`;
        const period = this.isNight ? 'NIGHT' : this.isDusk ? 'DUSK' : this.isDawn ? 'DAWN' : 'DAY';
        return `${timeStr} ${period} ${this.weather.toUpperCase()}`;
    }
}
