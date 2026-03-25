// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * FloatingText — DOM element that rises and fades (damage numbers, streak text).
 *
 * Creates an absolutely-positioned div at screen coordinates that floats
 * upward and fades out. Pure DOM, no Three.js dependency.
 */

export class FloatingText {
    /**
     * @param {Object} opts
     * @param {number} opts.screenX — initial screen X
     * @param {number} opts.screenY — initial screen Y
     * @param {string} opts.text — display text
     * @param {string} opts.color — CSS color
     * @param {number} opts.duration — ms
     * @param {string} [opts.fontSize='14px']
     * @param {HTMLElement} container — parent element
     */
    constructor(opts, container) {
        this.alive = true;
        this.startTime = performance.now();
        this.endTime = this.startTime + (opts.duration || 1200);
        this.duration = opts.duration || 1200;
        this.mesh = null; // No Three.js mesh

        this.domEl = document.createElement('div');
        this.domEl.style.cssText = `
            position: absolute; left: ${opts.screenX}px; top: ${opts.screenY}px;
            font-family: 'JetBrains Mono', monospace; font-size: ${opts.fontSize || '14px'};
            font-weight: bold; color: ${opts.color || '#ff2a6d'};
            pointer-events: none; z-index: 800;
            text-shadow: 0 0 4px ${opts.color || '#ff2a6d'}44;
            transition: none;
        `;
        this.domEl.textContent = opts.text;
        this.startY = opts.screenY;
        if (container) container.appendChild(this.domEl);
    }

    update(now) {
        const t = Math.min(1, (now - this.startTime) / this.duration);
        this.domEl.style.top = `${this.startY - t * 40}px`;
        this.domEl.style.opacity = String(1 - t);
        if (t >= 1) {
            this.alive = false;
            this.domEl.remove();
        }
    }
}
