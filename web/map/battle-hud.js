// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * BattleHUD — screen effects, wave announcements, kill feed, game-over overlay.
 *
 * Pure DOM manipulation — no MapLibre or Three.js dependency.
 * Listens to game state events and renders tactical HUD elements.
 *
 * Usage:
 *   import { BattleHUD } from '/lib/map/battle-hud.js';
 *   const hud = new BattleHUD(container, eventBus);
 *   hud.onGameStateChange({ state: 'active', wave: 3, ... });
 */

export class BattleHUD {
    /**
     * @param {HTMLElement} container — map container element
     * @param {Object} [options]
     * @param {boolean} [options.showKillFeed=true]
     * @param {boolean} [options.showBanners=true]
     * @param {boolean} [options.showScreenFx=true]
     * @param {number} [options.maxKillFeedEntries=8]
     */
    constructor(container, options = {}) {
        this.container = container;
        this.showKillFeed = options.showKillFeed ?? true;
        this.showBanners = options.showBanners ?? true;
        this.showScreenFx = options.showScreenFx ?? true;
        this.maxKillFeedEntries = options.maxKillFeedEntries ?? 8;

        this._killFeedEl = null;
        this._killFeedEntries = [];
        this._bannerEl = null;
        this._bannerTimeout = null;
        this._vignetteEl = null;
        this._shakeAnimId = null;
    }

    // ── Screen Shake ─────────────────────────────────────────────

    /**
     * Shake the container element.
     * @param {number} [intensity=5] — max pixel displacement
     * @param {number} [duration=300] — ms
     */
    triggerShake(intensity = 5, duration = 300) {
        if (!this.showScreenFx) return;
        const start = performance.now();
        const el = this.container;
        const originalTransform = el.style.transform || '';

        const shake = () => {
            const t = performance.now() - start;
            if (t >= duration) {
                el.style.transform = originalTransform;
                return;
            }
            const decay = 1 - t / duration;
            const dx = (Math.random() - 0.5) * 2 * intensity * decay;
            const dy = (Math.random() - 0.5) * 2 * intensity * decay;
            el.style.transform = `${originalTransform} translate(${dx}px, ${dy}px)`;
            this._shakeAnimId = requestAnimationFrame(shake);
        };
        shake();
    }

    // ── Screen Flash ─────────────────────────────────────────────

    /**
     * Brief full-screen color flash.
     * @param {string} [color='rgba(255,255,255,0.3)']
     * @param {number} [duration=150] — ms
     */
    triggerFlash(color = 'rgba(255,255,255,0.3)', duration = 150) {
        if (!this.showScreenFx) return;
        const flash = document.createElement('div');
        flash.style.cssText = `
            position: absolute; inset: 0; z-index: 700;
            background: ${color}; pointer-events: none;
            opacity: 1; transition: opacity ${duration}ms ease-out;
        `;
        this.container.appendChild(flash);
        requestAnimationFrame(() => { flash.style.opacity = '0'; });
        setTimeout(() => flash.remove(), duration + 50);
    }

    // ── Battle Vignette ──────────────────────────────────────────

    /** Show dark vignette overlay during combat. */
    showVignette() {
        if (this._vignetteEl) return;
        this._vignetteEl = document.createElement('div');
        this._vignetteEl.style.cssText = `
            position: absolute; inset: 0; z-index: 600; pointer-events: none;
            background: radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.4) 100%);
            opacity: 0; transition: opacity 1s ease;
        `;
        this.container.appendChild(this._vignetteEl);
        requestAnimationFrame(() => { if (this._vignetteEl) this._vignetteEl.style.opacity = '1'; });
    }

    /** Remove combat vignette. */
    hideVignette() {
        if (!this._vignetteEl) return;
        this._vignetteEl.style.opacity = '0';
        const el = this._vignetteEl;
        this._vignetteEl = null;
        setTimeout(() => el.remove(), 1000);
    }

    // ── Banner (Wave Announcements) ──────────────────────────────

    /**
     * Show a large centered banner message (e.g., "WAVE 3", "DEFEND THE AREA").
     * @param {string} text
     * @param {string} [color='#00f0ff']
     * @param {number} [duration=3000] — ms before fade out
     */
    showBanner(text, color = '#00f0ff', duration = 3000) {
        if (!this.showBanners) return;
        if (this._bannerEl) this._bannerEl.remove();
        if (this._bannerTimeout) clearTimeout(this._bannerTimeout);

        this._bannerEl = document.createElement('div');
        this._bannerEl.style.cssText = `
            position: absolute; top: 30%; left: 50%; transform: translateX(-50%);
            z-index: 750; pointer-events: none;
            font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: bold;
            color: ${color}; letter-spacing: 4px; text-transform: uppercase;
            text-shadow: 0 0 20px ${color}44, 0 0 40px ${color}22;
            opacity: 0; transition: opacity 0.5s ease;
        `;
        this._bannerEl.textContent = text;
        this.container.appendChild(this._bannerEl);
        requestAnimationFrame(() => { if (this._bannerEl) this._bannerEl.style.opacity = '1'; });

        this._bannerTimeout = setTimeout(() => {
            if (this._bannerEl) {
                this._bannerEl.style.opacity = '0';
                setTimeout(() => { this._bannerEl?.remove(); this._bannerEl = null; }, 500);
            }
        }, duration);
    }

    // ── Kill Feed ────────────────────────────────────────────────

    /**
     * Add an entry to the kill feed.
     * @param {string} text — e.g., "Turret-01 eliminated Hostile-3"
     * @param {string} [color='#ff2a6d']
     */
    addKillFeedEntry(text, color = '#ff2a6d') {
        if (!this.showKillFeed) return;
        this._ensureKillFeedEl();

        const entry = document.createElement('div');
        entry.style.cssText = `
            font-family: 'JetBrains Mono', monospace; font-size: 10px;
            color: ${color}; padding: 2px 6px; margin-bottom: 2px;
            background: rgba(0,0,0,0.6); border-left: 2px solid ${color};
            opacity: 1; transition: opacity 0.5s ease;
        `;
        entry.textContent = text;
        this._killFeedEl.appendChild(entry);
        this._killFeedEntries.push(entry);

        // Cap entries
        while (this._killFeedEntries.length > this.maxKillFeedEntries) {
            const old = this._killFeedEntries.shift();
            old?.remove();
        }

        // Auto-fade after 8s
        setTimeout(() => {
            entry.style.opacity = '0';
            setTimeout(() => entry.remove(), 500);
        }, 8000);
    }

    _ensureKillFeedEl() {
        if (this._killFeedEl) return;
        this._killFeedEl = document.createElement('div');
        this._killFeedEl.style.cssText = `
            position: absolute; top: 60px; right: 8px; z-index: 710;
            max-width: 280px; pointer-events: none;
        `;
        this.container.appendChild(this._killFeedEl);
    }

    // ── Game Over Overlay ────────────────────────────────────────

    /**
     * Show game-over stats overlay.
     * @param {Object} stats — { score, waves, eliminations, accuracy, mvp, ... }
     * @param {string} [outcome='VICTORY'] — 'VICTORY' or 'DEFEAT'
     * @param {Function} [onPlayAgain] — callback for play again button
     * @param {Function} [onClose] — callback for close button
     */
    showGameOver(stats, outcome = 'VICTORY', onPlayAgain, onClose) {
        const color = outcome === 'VICTORY' ? '#05ffa1' : '#ff2a6d';
        const overlay = document.createElement('div');
        overlay.id = 'battle-hud-gameover';
        overlay.style.cssText = `
            position: absolute; inset: 0; z-index: 800;
            background: rgba(2, 2, 6, 0.92); backdrop-filter: blur(12px);
            display: flex; align-items: center; justify-content: center;
        `;
        overlay.innerHTML = `
            <div style="text-align:center;font-family:'JetBrains Mono',monospace;color:#ccc;max-width:500px">
                <div style="font-size:32px;font-weight:bold;color:${color};letter-spacing:4px;margin-bottom:16px">${outcome}</div>
                <div style="display:flex;justify-content:center;gap:24px;margin-bottom:16px">
                    <div><div style="font-size:24px;color:#00f0ff">${stats.score || 0}</div><div style="font-size:9px;color:#666">SCORE</div></div>
                    <div><div style="font-size:24px;color:#00f0ff">${stats.waves || 0}</div><div style="font-size:9px;color:#666">WAVES</div></div>
                    <div><div style="font-size:24px;color:#00f0ff">${stats.eliminations || 0}</div><div style="font-size:9px;color:#666">ELIMINATIONS</div></div>
                </div>
                <div style="display:flex;justify-content:center;gap:8px;margin-top:16px">
                    <button class="bh-btn bh-play-again" style="padding:6px 16px;font-family:monospace;font-size:11px;background:transparent;border:1px solid ${color};color:${color};cursor:pointer;border-radius:3px">PLAY AGAIN</button>
                    <button class="bh-btn bh-close" style="padding:6px 16px;font-family:monospace;font-size:11px;background:transparent;border:1px solid #666;color:#888;cursor:pointer;border-radius:3px">CLOSE</button>
                </div>
            </div>
        `;
        this.container.appendChild(overlay);

        overlay.querySelector('.bh-play-again')?.addEventListener('click', () => {
            overlay.remove();
            if (onPlayAgain) onPlayAgain();
        });
        overlay.querySelector('.bh-close')?.addEventListener('click', () => {
            overlay.remove();
            if (onClose) onClose();
        });
    }

    /** Remove game-over overlay if present. */
    hideGameOver() {
        document.getElementById('battle-hud-gameover')?.remove();
    }

    // ── Cleanup ──────────────────────────────────────────────────

    destroy() {
        if (this._shakeAnimId) cancelAnimationFrame(this._shakeAnimId);
        if (this._bannerTimeout) clearTimeout(this._bannerTimeout);
        this._bannerEl?.remove();
        this._vignetteEl?.remove();
        this._killFeedEl?.remove();
        this.hideGameOver();
    }
}
