// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Panel Utilities -- shared helpers used across all panel modules.
// Import: import { _esc, _timeAgo, _badge, _statusDot, _fetchJson } from './utils.js';

/**
 * HTML-escape a string to prevent XSS when inserting into innerHTML.
 * @param {string} text
 * @returns {string} Escaped HTML string
 */
export function _esc(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

/**
 * Format a Unix timestamp (seconds) as a human-readable relative time.
 * @param {number} ts Unix timestamp in seconds
 * @returns {string} e.g. "just now", "5s ago", "3m ago", "2h ago", "1d ago"
 */
export function _timeAgo(ts) {
    if (!ts) return 'never';
    const secs = Math.floor(Date.now() / 1000 - ts);
    if (secs < 5) return 'just now';
    if (secs < 60) return `${secs}s ago`;
    if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
    if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
    return `${Math.floor(secs / 86400)}d ago`;
}

/**
 * Create a colored badge HTML string.
 * @param {string} label Badge text
 * @param {string} color CSS color value
 * @param {object} [opts] Optional: { title, style }
 * @returns {string} HTML string for a badge span
 */
export function _badge(label, color, opts) {
    const title = (opts && opts.title) ? ` title="${_esc(opts.title)}"` : '';
    const extra = (opts && opts.style) ? `;${opts.style}` : '';
    return `<span class="panel-badge" style="background:${color};color:#0a0a0f;padding:1px 5px;border-radius:3px;font-size:0.5rem;font-weight:bold${extra}"${title}>${_esc(label)}</span>`;
}

/**
 * Create a colored status dot HTML string.
 * @param {string} status One of: "online", "stale", "offline", or any other value
 * @returns {string} HTML string for a small colored dot
 */
export function _statusDot(status) {
    const colors = {
        online: 'var(--green, #05ffa1)',
        stale: 'var(--yellow, #fcee0a)',
        offline: 'var(--magenta, #ff2a6d)',
    };
    const color = colors[status] || 'var(--text-dim, #888)';
    return `<span class="panel-status-dot" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:4px" title="${(status || 'unknown').toUpperCase()}"></span>`;
}

/**
 * Fetch JSON from an API endpoint with error handling.
 * @param {string} url API URL
 * @param {object} [opts] Fetch options (method, body, headers)
 * @returns {Promise<any>} Parsed JSON response
 * @throws {Error} On HTTP error or network failure
 */
export async function _fetchJson(url, opts) {
    const resp = await fetch(url, opts);
    if (!resp.ok) {
        const text = await resp.text().catch(() => '');
        throw new Error(`HTTP ${resp.status}: ${text || resp.statusText}`);
    }
    return resp.json();
}

/**
 * Canonical threat level color mapping.
 * Keys cover all naming conventions used across the codebase:
 * lowercase (low/moderate/high/critical), UPPERCASE, color names (green/yellow/orange/red),
 * and semantic aliases (none/medium/unknown).
 *
 * Cyberpunk palette: green=#05ffa1, yellow=#fcee0a, orange=#ff8c00, magenta=#ff2a6d.
 */
export const THREAT_COLORS = {
    // Lowercase threat levels
    none:     '#888',
    low:      '#05ffa1',
    moderate: '#fcee0a',
    medium:   '#fcee0a',
    high:     '#ff8c00',
    critical: '#ff2a6d',
    unknown:  '#666',
    // UPPERCASE threat levels
    NONE:     '#888',
    LOW:      '#05ffa1',
    MODERATE: '#fcee0a',
    MEDIUM:   '#fcee0a',
    HIGH:     '#ff8c00',
    CRITICAL: '#ff2a6d',
    UNKNOWN:  '#666',
    // Color-name keys (sitaware style)
    green:    '#05ffa1',
    yellow:   '#fcee0a',
    orange:   '#ff8c00',
    red:      '#ff2a6d',
    // Color-name UPPERCASE
    GREEN:    '#05ffa1',
    YELLOW:   '#fcee0a',
    ORANGE:   '#ff8c00',
    RED:      '#ff2a6d',
};

/**
 * Format a Unix timestamp (seconds) as a 24-hour time-only string (HH:MM:SS).
 * Wave 203: hour12:false ensures consistent 24h format across locales —
 * default 2-digit on en-US gives "02:13:20 PM" (12h), but tactical UI
 * panels require 24h with no AM/PM suffix.
 * @param {number} ts — Unix timestamp in seconds
 * @returns {string} e.g. "14:30:05" or "--" if invalid
 */
export function _formatTime(ts) {
    if (!ts) return '--';
    const d = new Date(ts * 1000);
    if (isNaN(d.getTime())) return '--';
    return d.toLocaleTimeString(undefined, {
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: false,
    });
}

/**
 * Format a Unix timestamp (seconds) as a 24-hour date+time string.
 * @param {number} ts — Unix timestamp in seconds
 * @returns {string} e.g. "Mar 25, 14:30:05" or "--" if invalid
 */
export function _formatTimestamp(ts) {
    if (!ts) return '--';
    const d = new Date(ts * 1000);
    if (isNaN(d.getTime())) return '--';
    return d.toLocaleString(undefined, {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: false,
    });
}

/**
 * Terminal target statuses — units in these states are dead/gone and should
 * not be counted as active.  This must agree with the Python source of truth
 * in tritium-sc/src/app/routers/targets_unified.py:_TERMINAL_STATUSES so the
 * header counter and /api/targets agree (Wave 198 mismatch fix).
 *
 * Canonical full status enum (10 values) lives in
 * tritium-lib/src/tritium_lib/sim_engine/core/entity.py:494 — terminal
 * statuses are the subset of entries that mark a unit as no-longer-active.
 */
export const TERMINAL_STATUSES = new Set([
    'eliminated',
    'destroyed',
    'despawned',
    'neutralized',
    'escaped',
]);
