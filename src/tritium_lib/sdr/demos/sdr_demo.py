# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Standalone SDR spectrum analyzer demo — FastAPI app.

Proves tritium-lib SDR simulation + analysis works independently of tritium-sc.

Run with:
    PYTHONPATH=src python3 src/tritium_lib/sdr/demos/sdr_demo.py

Endpoints:
    GET  /spectrum         — latest spectrum sweep data
    GET  /signals          — detected signals with classification
    POST /sweep            — trigger a new sweep (JSON body: start_hz, end_hz, bin_width_hz)
    GET  /waterfall        — waterfall display data (time x frequency matrix)
    GET  /presets          — scan preset frequency ranges
    GET  /bands            — known RF frequency bands
    GET  /status           — analyzer status summary
    GET  /                 — HTML dashboard with live spectrum + waterfall
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from tritium_lib.sdr.simulator import SimulatedSDR
from tritium_lib.sdr.analyzer import SpectrumAnalyzer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEMO_PORT = 9092
SWEEP_INTERVAL = 2.0  # seconds between automatic sweeps

# Default sweep range — FM broadcast band
DEFAULT_START_HZ = 88_000_000
DEFAULT_END_HZ = 108_000_000
DEFAULT_BIN_WIDTH_HZ = 100_000

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

sdr = SimulatedSDR(seed=42, noise_floor_dbm=-95.0)
analyzer = SpectrumAnalyzer(sdr, waterfall_depth=200)

_bg_task: asyncio.Task | None = None
_current_start_hz: int = DEFAULT_START_HZ
_current_end_hz: int = DEFAULT_END_HZ
_current_bin_width_hz: int = DEFAULT_BIN_WIDTH_HZ


# ---------------------------------------------------------------------------
# Background sweep loop
# ---------------------------------------------------------------------------

async def _sweep_loop() -> None:
    """Continuously sweep and detect signals."""
    while True:
        try:
            result = await analyzer.scan(
                _current_start_hz, _current_end_hz, _current_bin_width_hz
            )
            analyzer.detect_signals(result, threshold_dbm=-60.0)
        except Exception as e:
            print(f"Sweep error: {e}")
        await asyncio.sleep(SWEEP_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bg_task
    await analyzer.initialize()
    _bg_task = asyncio.create_task(_sweep_loop())
    print(f"SDR Spectrum Analyzer demo on http://localhost:{DEMO_PORT}")
    print(f"  Sweeping {_current_start_hz/1e6:.1f} - {_current_end_hz/1e6:.1f} MHz")
    print(f"  {len(sdr.signals)} simulated signal sources")
    yield
    if _bg_task:
        _bg_task.cancel()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Tritium SDR Spectrum Analyzer Demo",
    description="Simulated SDR spectrum sweeps, signal detection, waterfall display",
    lifespan=lifespan,
)


class SweepRequest(BaseModel):
    start_hz: int = DEFAULT_START_HZ
    end_hz: int = DEFAULT_END_HZ
    bin_width_hz: int = DEFAULT_BIN_WIDTH_HZ
    threshold_dbm: float = -60.0


@app.get("/spectrum")
async def get_spectrum():
    """Return the latest spectrum sweep data."""
    if analyzer._last_sweep is None:
        return JSONResponse({"error": "No sweep data yet"}, status_code=404)
    sweep = analyzer._last_sweep
    return {
        "sweep": sweep.to_dict(),
        "freq_start_mhz": round(sweep.freq_start_hz / 1e6, 3),
        "freq_end_mhz": round(sweep.freq_end_hz / 1e6, 3),
        "bin_width_khz": round(sweep.bin_width_hz / 1e3, 1),
    }


@app.get("/signals")
async def get_signals():
    """Return detected signals with classification."""
    signals = analyzer._last_signals
    return {
        "signals": [s.to_dict() for s in signals],
        "count": len(signals),
        "sweep_count": analyzer._sweep_count,
        "categories": _signal_category_summary(signals),
    }


@app.post("/sweep")
async def trigger_sweep(req: SweepRequest):
    """Trigger a sweep with custom parameters."""
    global _current_start_hz, _current_end_hz, _current_bin_width_hz

    # Validate range
    if req.end_hz <= req.start_hz:
        return JSONResponse(
            {"error": "end_hz must be greater than start_hz"}, status_code=400
        )
    if req.bin_width_hz < 1000:
        return JSONResponse(
            {"error": "bin_width_hz must be at least 1000"}, status_code=400
        )

    # Update current sweep parameters
    _current_start_hz = req.start_hz
    _current_end_hz = req.end_hz
    _current_bin_width_hz = req.bin_width_hz

    # Run immediate sweep
    result = await analyzer.scan(req.start_hz, req.end_hz, req.bin_width_hz)
    signals = analyzer.detect_signals(result, threshold_dbm=req.threshold_dbm)

    return {
        "sweep": result.to_dict(),
        "signals": [s.to_dict() for s in signals],
        "signal_count": len(signals),
        "freq_start_mhz": round(req.start_hz / 1e6, 3),
        "freq_end_mhz": round(req.end_hz / 1e6, 3),
    }


@app.get("/waterfall")
async def get_waterfall(max_rows: int = 50):
    """Return waterfall display data."""
    return analyzer.get_waterfall(max_rows=max_rows)


@app.get("/presets")
async def get_presets():
    """Return scan preset configurations."""
    return {"presets": SpectrumAnalyzer.get_scan_presets()}


@app.get("/bands")
async def get_bands():
    """Return known RF frequency bands."""
    return {"bands": SpectrumAnalyzer.get_known_bands()}


@app.get("/status")
async def get_status():
    """Return analyzer status."""
    status = analyzer.get_status()
    status["current_sweep"] = {
        "start_hz": _current_start_hz,
        "end_hz": _current_end_hz,
        "bin_width_hz": _current_bin_width_hz,
        "start_mhz": round(_current_start_hz / 1e6, 3),
        "end_mhz": round(_current_end_hz / 1e6, 3),
    }
    status["signal_sources"] = len(sdr.signals)
    return status


def _signal_category_summary(signals) -> dict:
    """Summarize signals by category."""
    cats: dict[str, int] = {}
    for s in signals:
        cats[s.category] = cats.get(s.category, 0) + 1
    return cats


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Tritium SDR Spectrum Analyzer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0a0a; color: #c0c0c0; font-family: 'Courier New', monospace; }
h1 { color: #00f0ff; text-align: center; padding: 12px; font-size: 18px;
     text-shadow: 0 0 10px #00f0ff44; border-bottom: 1px solid #1a1a1a; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 8px; }
.panel { background: #111; border: 1px solid #1a1a1a; border-radius: 4px; padding: 12px; }
.panel h2 { color: #05ffa1; font-size: 13px; margin-bottom: 8px; }
.fullwidth { grid-column: 1 / -1; }
canvas { width: 100%; background: #080808; border: 1px solid #1a1a1a; border-radius: 4px; }
.stat { display: flex; justify-content: space-between; padding: 2px 0;
        border-bottom: 1px solid #0a0a0a; font-size: 12px; }
.stat .val { color: #00f0ff; font-weight: bold; }
table { width: 100%; border-collapse: collapse; font-size: 11px; }
th { color: #ff2a6d; text-align: left; padding: 4px; border-bottom: 1px solid #222; }
td { padding: 3px 4px; border-bottom: 1px solid #111; }
tr:hover { background: #1a1a1a; }
.badge { display: inline-block; padding: 1px 6px; border-radius: 3px;
         font-size: 10px; font-weight: bold; }
.badge-fm { background: #ff2a6d22; color: #ff2a6d; border: 1px solid #ff2a6d44; }
.badge-wifi { background: #00f0ff22; color: #00f0ff; border: 1px solid #00f0ff44; }
.badge-ble { background: #05ffa122; color: #05ffa1; border: 1px solid #05ffa144; }
.badge-lora { background: #fcee0a22; color: #fcee0a; border: 1px solid #fcee0a44; }
.badge-ism { background: #ff8c0022; color: #ff8c00; border: 1px solid #ff8c0044; }
.badge-cellular { background: #9966ff22; color: #9966ff; border: 1px solid #9966ff44; }
.badge-aviation { background: #ff44ff22; color: #ff44ff; border: 1px solid #ff44ff44; }
.badge-unknown { background: #66666622; color: #888; border: 1px solid #44444444; }
select, button { background: #1a1a1a; color: #00f0ff; border: 1px solid #333;
                 padding: 4px 8px; border-radius: 3px; font-family: inherit;
                 font-size: 12px; cursor: pointer; }
button:hover { background: #222; border-color: #00f0ff; }
.controls { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }
</style>
</head>
<body>
<h1>TRITIUM SDR SPECTRUM ANALYZER</h1>
<div class="grid">
  <div class="panel fullwidth">
    <h2>SPECTRUM</h2>
    <div class="controls">
      <select id="preset-select">
        <option value="">Select Preset...</option>
      </select>
      <button onclick="applyPreset()">Apply</button>
      <span id="sweep-info" style="color: #666; font-size: 11px;"></span>
    </div>
    <canvas id="spectrum-canvas" height="200"></canvas>
  </div>
  <div class="panel fullwidth">
    <h2>WATERFALL</h2>
    <canvas id="waterfall-canvas" height="200"></canvas>
  </div>
  <div class="panel">
    <h2>DETECTED SIGNALS</h2>
    <table>
      <thead><tr><th>Frequency</th><th>Power</th><th>SNR</th><th>Category</th><th>Band</th></tr></thead>
      <tbody id="signals-body"></tbody>
    </table>
  </div>
  <div class="panel">
    <h2>STATUS</h2>
    <div id="status">Loading...</div>
  </div>
</div>
<script>
let currentPreset = null;

function catBadge(cat) {
    const cls = {fm:'badge-fm',broadcast:'badge-fm',wifi:'badge-wifi',ble:'badge-ble',
                 lora:'badge-lora',ism:'badge-ism',cellular:'badge-cellular',
                 aviation:'badge-aviation'}[cat] || 'badge-unknown';
    return `<span class="badge ${cls}">${cat}</span>`;
}

async function loadPresets() {
    const data = await (await fetch('/presets')).json();
    const sel = document.getElementById('preset-select');
    data.presets.forEach(p => {
        const opt = document.createElement('option');
        opt.value = JSON.stringify(p);
        opt.textContent = `${p.name} (${p.start_mhz}-${p.end_mhz} MHz)`;
        sel.appendChild(opt);
    });
}

async function applyPreset() {
    const sel = document.getElementById('preset-select');
    if (!sel.value) return;
    const p = JSON.parse(sel.value);
    currentPreset = p;
    await fetch('/sweep', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({start_hz: p.start_hz, end_hz: p.end_hz, bin_width_hz: p.bin_width_hz})
    });
}

function drawSpectrum(sweep) {
    const canvas = document.getElementById('spectrum-canvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.clientWidth;
    const H = 200;
    canvas.width = W;
    canvas.height = H;
    ctx.fillStyle = '#080808';
    ctx.fillRect(0, 0, W, H);

    const pts = sweep.points;
    if (!pts || !pts.length) return;

    const powers = pts.map(p => p.power);
    const minP = -100;
    const maxP = Math.max(-20, ...powers);

    // Grid lines
    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth = 0.5;
    for (let db = -90; db <= -20; db += 10) {
        const y = H - ((db - minP) / (maxP - minP)) * H;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
        ctx.fillStyle = '#444';
        ctx.font = '9px monospace';
        ctx.fillText(db + ' dBm', 2, y - 2);
    }

    // Spectrum line
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 1.5;
    ctx.shadowColor = '#00f0ff';
    ctx.shadowBlur = 4;
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
        const x = (i / pts.length) * W;
        const y = H - ((powers[i] - minP) / (maxP - minP)) * H;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Filled area
    ctx.globalAlpha = 0.15;
    ctx.fillStyle = '#00f0ff';
    ctx.lineTo(W, H);
    ctx.lineTo(0, H);
    ctx.fill();
    ctx.globalAlpha = 1.0;

    // Frequency labels
    ctx.fillStyle = '#666';
    ctx.font = '9px monospace';
    const startMhz = sweep.freq_start / 1e6;
    const endMhz = sweep.freq_end / 1e6;
    for (let i = 0; i <= 10; i++) {
        const freq = startMhz + (endMhz - startMhz) * (i / 10);
        const x = (i / 10) * W;
        ctx.fillText(freq.toFixed(1), x + 2, H - 4);
    }
}

function drawWaterfall(data) {
    const canvas = document.getElementById('waterfall-canvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.clientWidth;
    const H = 200;
    canvas.width = W;
    canvas.height = H;
    ctx.fillStyle = '#080808';
    ctx.fillRect(0, 0, W, H);

    if (!data.rows || !data.rows.length || !data.num_bins) return;

    const rows = data.rows;
    const numBins = data.num_bins;
    const rowH = Math.max(1, H / rows.length);
    const colW = W / numBins;

    for (let r = 0; r < rows.length; r++) {
        for (let c = 0; c < numBins; c++) {
            const power = rows[r][c];
            // Map power -100..-20 to color
            const norm = Math.max(0, Math.min(1, (power + 100) / 80));
            if (norm < 0.01) continue;
            // Blue -> Cyan -> Green -> Yellow -> Red
            let R, G, B;
            if (norm < 0.25) {
                R = 0; G = 0; B = Math.floor(norm * 4 * 255);
            } else if (norm < 0.5) {
                const t = (norm - 0.25) * 4;
                R = 0; G = Math.floor(t * 255); B = 255;
            } else if (norm < 0.75) {
                const t = (norm - 0.5) * 4;
                R = Math.floor(t * 255); G = 255; B = Math.floor((1-t) * 255);
            } else {
                const t = (norm - 0.75) * 4;
                R = 255; G = Math.floor((1-t) * 255); B = 0;
            }
            ctx.fillStyle = `rgb(${R},${G},${B})`;
            ctx.fillRect(Math.floor(c * colW), Math.floor(r * rowH),
                         Math.ceil(colW), Math.ceil(rowH));
        }
    }
}

async function updateSpectrum() {
    try {
        const data = await (await fetch('/spectrum')).json();
        if (data.sweep) {
            drawSpectrum(data.sweep);
            document.getElementById('sweep-info').textContent =
                `${data.freq_start_mhz}-${data.freq_end_mhz} MHz | ${data.sweep.num_points} bins | ${data.sweep.sweep_time_ms.toFixed(1)}ms`;
        }
    } catch(e) {}
}

async function updateWaterfall() {
    try {
        const data = await (await fetch('/waterfall?max_rows=100')).json();
        drawWaterfall(data);
    } catch(e) {}
}

async function updateSignals() {
    try {
        const data = await (await fetch('/signals')).json();
        const tbody = document.getElementById('signals-body');
        tbody.innerHTML = data.signals.map(s => `<tr>
            <td>${s.freq_mhz.toFixed(3)} MHz</td>
            <td>${s.power_dbm.toFixed(1)} dBm</td>
            <td>${s.snr_db.toFixed(1)} dB</td>
            <td>${catBadge(s.category)}</td>
            <td>${s.band_name || '-'}</td>
        </tr>`).join('');
    } catch(e) {}
}

async function updateStatus() {
    try {
        const s = await (await fetch('/status')).json();
        document.getElementById('status').innerHTML = [
            ['Device', s.device.name || 'N/A'],
            ['Serial', s.device.serial || 'N/A'],
            ['Sweeps', s.sweep_count],
            ['Signals Detected', s.total_signals_detected],
            ['Current Signals', s.last_signals_count],
            ['Signal Sources', s.signal_sources],
            ['Waterfall Rows', `${s.waterfall_depth}/${s.waterfall_max}`],
            ['Noise Floor', `${s.noise_floor_dbm} dBm`],
            ['Tracked Freqs', s.tracked_frequencies],
            ['Range', `${s.current_sweep.start_mhz}-${s.current_sweep.end_mhz} MHz`],
        ].map(([k,v]) => `<div class="stat"><span>${k}</span><span class="val">${v}</span></div>`).join('');
    } catch(e) {}
}

async function refresh() {
    await Promise.all([updateSpectrum(), updateWaterfall(), updateSignals(), updateStatus()]);
}

loadPresets();
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the SDR spectrum analyzer dashboard."""
    return _DASHBOARD_HTML


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=DEMO_PORT, log_level="warning")
