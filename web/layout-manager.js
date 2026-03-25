// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Generic layout manager — save, apply, delete, export, import panel layouts.
// SC provides built-in presets via the constructor; this module handles persistence.

const DEFAULT_STORAGE_KEY = 'tritium-layouts';

export class LayoutManager {
    /**
     * @param {Object} [builtinLayouts={}] - Map of name -> layout definition.
     *   Each layout is { panels: { panelId: { x, y, w, h, visible } } }.
     *   Negative x/y values mean "from right/bottom edge".
     * @param {Object} [opts={}] - Options.
     * @param {string} [opts.storageKey] - localStorage key (default 'tritium-layouts').
     */
    constructor(builtinLayouts = {}, opts = {}) {
        this._builtins = builtinLayouts;
        this._storageKey = opts.storageKey || DEFAULT_STORAGE_KEY;
        this._custom = this._loadFromStorage();
    }

    /**
     * Save the current panel positions as a named layout.
     * @param {string} name
     * @param {Object} panelState - { panelId: { x, y, w, h, visible } }
     */
    saveCurrent(name, panelState) {
        this._custom[name] = {
            panels: panelState,
            savedAt: Date.now(),
        };
        this._saveToStorage();
    }

    /**
     * Get a layout by name (custom overrides builtin).
     * @param {string} name
     * @returns {Object|null} layout definition or null
     */
    apply(name) {
        return this._custom[name] || this._builtins[name] || null;
    }

    /**
     * Delete a custom layout. Cannot delete builtins.
     * @param {string} name
     * @returns {boolean} true if deleted
     */
    delete(name) {
        if (this._custom[name]) {
            delete this._custom[name];
            this._saveToStorage();
            return true;
        }
        return false;
    }

    /**
     * List all available layout names.
     * @returns {Array<{name: string, builtin: boolean}>}
     */
    listAll() {
        const all = new Map();
        for (const name of Object.keys(this._builtins)) {
            all.set(name, { name, builtin: true });
        }
        for (const name of Object.keys(this._custom)) {
            all.set(name, { name, builtin: false });
        }
        return Array.from(all.values());
    }

    /**
     * Export a layout as a JSON string.
     * @param {string} name
     * @returns {string|null}
     */
    exportJSON(name) {
        const layout = this._custom[name] || this._builtins[name];
        if (!layout) return null;
        return JSON.stringify({ name, ...layout }, null, 2);
    }

    /**
     * Import a layout from a JSON string.
     * @param {string} json
     * @returns {string|null} the imported layout name, or null on error
     */
    importJSON(json) {
        try {
            const data = JSON.parse(json);
            const name = data.name;
            if (!name || !data.panels) return null;
            this._custom[name] = {
                panels: data.panels,
                savedAt: data.savedAt || Date.now(),
            };
            this._saveToStorage();
            return name;
        } catch (e) {
            console.warn('[LayoutManager] Import failed:', e);
            return null;
        }
    }

    /** @private */
    _loadFromStorage() {
        if (typeof localStorage === 'undefined') return {};
        try {
            const raw = localStorage.getItem(this._storageKey);
            return raw ? JSON.parse(raw) : {};
        } catch (e) {
            return {};
        }
    }

    /** @private */
    _saveToStorage() {
        if (typeof localStorage === 'undefined') return;
        try {
            localStorage.setItem(this._storageKey, JSON.stringify(this._custom));
        } catch (e) {
            console.warn('[LayoutManager] Save failed:', e);
        }
    }
}
