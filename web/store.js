// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Generic reactive store with dot-path subscriptions and RAF-batched notifications.
// SC extends this with domain-specific schema; this module is schema-free.

export class ReactiveStore {
    constructor() {
        /** @type {Map<string, Set<Function>>} */
        this._listeners = new Map();
        /** @type {Object} */
        this._state = {};
        /** @type {Set<string>} */
        this._pendingNotify = new Set();
        /** @type {number|null} */
        this._rafId = null;
    }

    /**
     * Subscribe to changes at a dot-path.
     * @param {string} path - e.g. 'game.phase'
     * @param {Function} fn - callback(newValue, oldValue)
     * @returns {Function} unsubscribe
     */
    on(path, fn) {
        if (!this._listeners.has(path)) {
            this._listeners.set(path, new Set());
        }
        this._listeners.get(path).add(fn);
        return () => this._listeners.get(path)?.delete(fn);
    }

    /**
     * Set a value at a dot-path. Triggers notification for that path.
     * @param {string} path - e.g. 'game.score'
     * @param {*} value
     */
    set(path, value) {
        const old = this.get(path);
        if (old === value) return;

        const parts = path.split('.');
        let obj = this._state;
        for (let i = 0; i < parts.length - 1; i++) {
            const key = parts[i];
            if (obj[key] === undefined || typeof obj[key] !== 'object') {
                obj[key] = {};
            }
            obj = obj[key];
        }
        obj[parts[parts.length - 1]] = value;

        this._scheduleNotify(path, value, old);
    }

    /**
     * Get a value at a dot-path.
     * @param {string} path
     * @param {*} [defaultValue]
     * @returns {*}
     */
    get(path, defaultValue) {
        const parts = path.split('.');
        let obj = this._state;
        for (const part of parts) {
            if (obj === undefined || obj === null) return defaultValue;
            obj = obj[part];
        }
        return obj !== undefined ? obj : defaultValue;
    }

    /**
     * Schedule a batched notification for the given path.
     * Uses requestAnimationFrame when available, otherwise fires synchronously.
     */
    _scheduleNotify(path, value, oldValue) {
        this._pendingNotify.add(path);
        // Store latest values for pending notifications
        if (!this._pendingValues) this._pendingValues = new Map();
        this._pendingValues.set(path, { value, oldValue });

        if (typeof requestAnimationFrame === 'function') {
            if (!this._rafId) {
                this._rafId = requestAnimationFrame(() => {
                    this._rafId = null;
                    this.flushNotify();
                });
            }
        } else {
            // Node.js / no-RAF environment — flush synchronously
            this.flushNotify();
        }
    }

    /**
     * Immediately fire all pending notifications.
     */
    flushNotify() {
        const pending = new Set(this._pendingNotify);
        const values = this._pendingValues || new Map();
        this._pendingNotify.clear();
        this._pendingValues = new Map();

        for (const path of pending) {
            const info = values.get(path);
            const val = info ? info.value : this.get(path);
            const old = info ? info.oldValue : undefined;
            this._notify(path, val, old);
        }
    }

    /**
     * Fire listeners for a specific path.
     */
    _notify(path, value, oldValue) {
        const fns = this._listeners.get(path);
        if (!fns) return;
        for (const fn of fns) {
            try {
                fn(value, oldValue);
            } catch (e) {
                console.error(`[Store] Listener error on "${path}":`, e);
            }
        }
    }

    /**
     * Remove all listeners and reset state.
     */
    destroy() {
        this._listeners.clear();
        this._state = {};
        this._pendingNotify.clear();
        if (this._pendingValues) this._pendingValues.clear();
        if (this._rafId && typeof cancelAnimationFrame === 'function') {
            cancelAnimationFrame(this._rafId);
        }
        this._rafId = null;
    }
}
