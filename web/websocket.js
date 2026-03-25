// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Generic WebSocket base class — connect, reconnect with exponential backoff,
// ping keepalive, and disconnected banner. SC subclasses _handleMessage().

export class TritiumWebSocket {
    /**
     * @param {string} url - WebSocket URL (e.g. 'ws://localhost:8000/ws/live')
     * @param {Object} [opts]
     * @param {number} [opts.initialDelay=1000] - Initial reconnect delay (ms)
     * @param {number} [opts.maxDelay=16000] - Max reconnect delay (ms)
     * @param {number} [opts.pingInterval=25000] - Ping keepalive interval (ms)
     * @param {Function} [opts.onOpen] - Called when connection opens
     * @param {Function} [opts.onClose] - Called when connection closes
     * @param {Function} [opts.onError] - Called on connection error
     * @param {Function} [opts.onMessage] - Called with parsed JSON message
     */
    constructor(url, opts = {}) {
        this._url = url;
        this._initialDelay = opts.initialDelay || 1000;
        this._maxDelay = opts.maxDelay || 16000;
        this._PING_INTERVAL_MS = opts.pingInterval || 25000;
        this._onOpen = opts.onOpen || null;
        this._onClose = opts.onClose || null;
        this._onError = opts.onError || null;
        this._onMessage = opts.onMessage || null;

        this._ws = null;
        this._reconnectTimer = null;
        this._reconnectDelay = this._initialDelay;
        this._pingTimer = null;
        this._disconnectedBanner = null;
        this._destroyed = false;
    }

    /**
     * Open the WebSocket connection.
     */
    connect() {
        if (this._destroyed) return;
        if (this._ws) {
            try { this._ws.close(); } catch (_) { /* ignore */ }
        }

        try {
            this._ws = new WebSocket(this._url);
        } catch (e) {
            console.warn('[WS] Connection error:', e);
            this._scheduleReconnect();
            return;
        }

        this._ws.onopen = () => {
            this._reconnectDelay = this._initialDelay;
            this._hideDisconnectedBanner();
            this._startPingKeepalive();
            if (this._onOpen) this._onOpen();
        };

        this._ws.onclose = () => {
            this._stopPingKeepalive();
            this._showDisconnectedBanner();
            if (this._onClose) this._onClose();
            this._scheduleReconnect();
        };

        this._ws.onerror = (err) => {
            console.warn('[WS] Error:', err);
            if (this._onError) this._onError(err);
        };

        this._ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (this._onMessage) {
                    this._onMessage(msg);
                }
                this._handleMessage(msg);
            } catch (e) {
                console.warn('[WS] Message parse error:', e);
            }
        };
    }

    /**
     * Send a JSON message.
     * @param {Object} data
     */
    send(data) {
        if (this._ws?.readyState === WebSocket.OPEN) {
            this._ws.send(JSON.stringify(data));
        }
    }

    /**
     * Close the connection and stop reconnecting.
     */
    disconnect() {
        this._destroyed = true;
        clearTimeout(this._reconnectTimer);
        this._stopPingKeepalive();
        this._hideDisconnectedBanner();
        if (this._ws) {
            this._ws.onclose = null;
            this._ws.onerror = null;
            this._ws.onmessage = null;
            try { this._ws.close(); } catch (_) { /* ignore */ }
            this._ws = null;
        }
    }

    /**
     * Override this in subclasses to route incoming messages.
     * @param {Object} msg - parsed JSON message
     */
    _handleMessage(msg) {
        // Default: respond to server pings
        const type = msg.type || msg.event;
        if (type === 'ping') {
            this.send({ type: 'pong' });
        }
    }

    /** @private */
    _scheduleReconnect() {
        if (this._destroyed) return;
        clearTimeout(this._reconnectTimer);
        const delay = this._reconnectDelay;
        this._reconnectTimer = setTimeout(() => {
            this._reconnectDelay = Math.min(this._reconnectDelay * 2, this._maxDelay);
            this.connect();
        }, delay);
    }

    /** @private */
    _showDisconnectedBanner() {
        if (typeof document === 'undefined') return;
        if (this._disconnectedBanner) return;
        const banner = document.createElement('div');
        banner.id = 'ws-disconnected-banner';
        banner.style.cssText = [
            'position: fixed', 'top: 0', 'left: 0', 'right: 0',
            'z-index: 99999', 'background: #ff2a6d', 'color: #fff',
            'text-align: center', 'padding: 6px 12px',
            'font-family: monospace', 'font-size: 13px',
            'font-weight: bold', 'letter-spacing: 2px',
            'text-transform: uppercase',
            'box-shadow: 0 2px 8px rgba(255,42,109,0.5)',
        ].join(';');
        banner.textContent = '// DISCONNECTED -- reconnecting...';
        document.body.appendChild(banner);
        this._disconnectedBanner = banner;
    }

    /** @private */
    _hideDisconnectedBanner() {
        if (this._disconnectedBanner) {
            this._disconnectedBanner.remove();
            this._disconnectedBanner = null;
        }
    }

    /** @private */
    _startPingKeepalive() {
        this._stopPingKeepalive();
        this._pingTimer = setInterval(() => {
            if (this._ws?.readyState === WebSocket.OPEN) {
                this.send({ type: 'ping' });
            }
        }, this._PING_INTERVAL_MS);
    }

    /** @private */
    _stopPingKeepalive() {
        if (this._pingTimer) {
            clearInterval(this._pingTimer);
            this._pingTimer = null;
        }
    }

    /**
     * Whether the WebSocket is currently open.
     * @returns {boolean}
     */
    get isConnected() {
        return this._ws?.readyState === WebSocket.OPEN;
    }
}
