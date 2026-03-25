// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Generic command palette engine — fuzzy search, keyboard nav, Ctrl+K trigger.
// SC provides the command list; this module handles DOM, scoring, and navigation.

/**
 * Fuzzy-score a query against a text string.
 * Returns 0 for no match, higher values for better matches.
 * Supports multi-term matching with word-boundary bonuses.
 * @param {string} query - User search input
 * @param {string} text - Text to match against
 * @returns {number} score (0 = no match)
 */
export function fuzzyScore(query, text) {
    if (!query) return 1;
    const q = query.toLowerCase();
    const t = text.toLowerCase();
    if (t.includes(q)) return 10 + (q.length / t.length) * 5;
    // Multi-term matching
    const terms = q.split(/\s+/);
    let total = 0;
    for (const term of terms) {
        if (!t.includes(term)) return 0;
        // Word-boundary bonus
        const idx = t.indexOf(term);
        const boundary = idx === 0 || /[\s\-_:/]/.test(t[idx - 1]);
        total += boundary ? 3 : 1;
    }
    return total;
}

/**
 * Initialize a command palette on the given container element.
 * @param {HTMLElement} containerEl - Element to append the palette DOM to
 * @param {Function} commandsFn - Returns array of {name, desc, action, category?}
 * @param {Object} [opts] - Options
 * @param {string} [opts.placeholder] - Input placeholder text
 * @param {string} [opts.triggerKey] - Key to open palette (default 'k')
 * @param {boolean} [opts.triggerCtrl] - Require Ctrl/Meta (default true)
 * @returns {{ open, close, isOpen, destroy }}
 */
export function initCommandPalette(containerEl, commandsFn, opts = {}) {
    const placeholder = opts.placeholder || 'Type a command...';
    const triggerKey = opts.triggerKey || 'k';
    const triggerCtrl = opts.triggerCtrl !== false;

    let overlay, input, list;
    let selectedIndex = 0;
    let filtered = [];
    let _open = false;

    // --- DOM ---
    function _createDOM() {
        overlay = document.createElement('div');
        overlay.className = 'cmd-palette-overlay';
        overlay.style.cssText = [
            'position: fixed', 'inset: 0', 'z-index: 99999',
            'background: rgba(0,0,0,0.6)', 'display: none',
            'justify-content: center', 'align-items: flex-start', 'padding-top: 15vh',
        ].join(';');

        const dialog = document.createElement('div');
        dialog.className = 'cmd-palette-dialog';
        dialog.style.cssText = [
            'background: #111', 'border: 1px solid #1a1a2e', 'border-radius: 8px',
            'width: 500px', 'max-width: 90vw', 'max-height: 60vh',
            'display: flex', 'flex-direction: column', 'overflow: hidden',
            'box-shadow: 0 8px 32px rgba(0,0,0,0.6)',
            'font-family: "JetBrains Mono", monospace',
        ].join(';');

        input = document.createElement('input');
        input.type = 'text';
        input.placeholder = placeholder;
        input.className = 'cmd-palette-input';
        input.style.cssText = [
            'background: transparent', 'border: none', 'border-bottom: 1px solid #1a1a2e',
            'color: #ccc', 'font-size: 14px', 'padding: 12px 16px', 'outline: none',
            'font-family: inherit',
        ].join(';');

        list = document.createElement('div');
        list.className = 'cmd-palette-list';
        list.style.cssText = 'overflow-y: auto; flex: 1;';

        dialog.appendChild(input);
        dialog.appendChild(list);
        overlay.appendChild(dialog);
        containerEl.appendChild(overlay);

        // Events
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) close();
        });
        input.addEventListener('input', () => {
            _filterAndRender();
        });
        input.addEventListener('keydown', _onInputKey);
    }

    // --- Fuzzy scoring (delegates to exported fuzzyScore) ---
    function _score(query, text) {
        return fuzzyScore(query, text);
    }

    function _filterAndRender() {
        const query = input.value.trim();
        const commands = commandsFn();
        if (!query) {
            filtered = commands.slice();
        } else {
            filtered = commands
                .map(cmd => ({ cmd, s: _score(query, cmd.name + ' ' + (cmd.desc || '') + ' ' + (cmd.category || '')) }))
                .filter(x => x.s > 0)
                .sort((a, b) => b.s - a.s)
                .map(x => x.cmd);
        }
        selectedIndex = 0;
        _renderList();
    }

    function _renderList() {
        list.innerHTML = '';
        filtered.forEach((cmd, i) => {
            const row = document.createElement('div');
            row.className = 'cmd-palette-item';
            row.style.cssText = [
                'padding: 8px 16px', 'cursor: pointer', 'display: flex',
                'justify-content: space-between', 'align-items: center',
                'border-bottom: 1px solid #0a0a12',
                i === selectedIndex ? 'background: #1a1a2e; color: #00f0ff' : 'color: #999',
            ].join(';');

            const nameSpan = document.createElement('span');
            nameSpan.textContent = cmd.name;
            nameSpan.style.cssText = 'font-size: 13px; font-weight: bold;';
            row.appendChild(nameSpan);

            if (cmd.desc) {
                const descSpan = document.createElement('span');
                descSpan.textContent = cmd.desc;
                descSpan.style.cssText = 'font-size: 10px; color: #666; margin-left: 12px;';
                row.appendChild(descSpan);
            }

            row.addEventListener('click', () => _execute(i));
            row.addEventListener('mouseenter', () => {
                selectedIndex = i;
                _renderList();
            });
            list.appendChild(row);
        });
    }

    // --- Keyboard nav ---
    function _onInputKey(e) {
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            selectedIndex = Math.min(selectedIndex + 1, filtered.length - 1);
            _renderList();
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            selectedIndex = Math.max(selectedIndex - 1, 0);
            _renderList();
        } else if (e.key === 'Enter') {
            e.preventDefault();
            _execute(selectedIndex);
        } else if (e.key === 'Escape') {
            close();
        }
    }

    function _execute(index) {
        const cmd = filtered[index];
        if (cmd && typeof cmd.action === 'function') {
            close();
            cmd.action();
        }
    }

    // --- Global trigger ---
    function _onGlobalKey(e) {
        if (triggerCtrl && (e.ctrlKey || e.metaKey) && e.key === triggerKey) {
            e.preventDefault();
            if (_open) close(); else open();
        }
    }

    // --- Public API ---
    function open() {
        if (!overlay) _createDOM();
        overlay.style.display = 'flex';
        input.value = '';
        _open = true;
        _filterAndRender();
        input.focus();
    }

    function close() {
        if (overlay) overlay.style.display = 'none';
        _open = false;
    }

    function isOpen() {
        return _open;
    }

    function destroy() {
        if (typeof document !== 'undefined') {
            document.removeEventListener('keydown', _onGlobalKey);
        }
        if (overlay && overlay.parentNode) {
            overlay.parentNode.removeChild(overlay);
        }
        overlay = input = list = null;
        _open = false;
    }

    // Bind global trigger
    if (typeof document !== 'undefined') {
        document.addEventListener('keydown', _onGlobalKey);
    }

    return { open, close, isOpen, destroy };
}
