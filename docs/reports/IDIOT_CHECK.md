VILLAGE IDIOT REPORT — 2026-03-21

PERSONA: Gary Kowalski, building manager, tech skill 2/5.
Expects dots on a map and some alerts. Confused by anything unlabeled or terminal-y.

TASK TESTED: Sim Engine Demo Game (tritium-lib game_server)

---

STEP 1: Is anything already running on port 9090?
- Tried: curl http://localhost:9090/
- Result: Nothing. Port was empty.
- PASS (nothing to clean up)

STEP 2: Does the server start?
- Command: SIM_PORT=9090 python3 -m tritium_lib.sim_engine.demos.game_server
- Result: Started silently in background with PID 17567. No error output.
- PASS

STEP 3: Does the web page load?
- Tried: curl http://localhost:9090/
- Result: Full HTML page returned — Tritium Sim Engine 3D Tactical viewer.
  Has a HUD overlay, cyberpunk styling (dark background, cyan text).
- PASS

STEP 4: Does /api/status work?
- Tried: curl http://localhost:9090/api/status
- Result immediately after start: {"running":false,"preset":"","tick_count":0}
- The server starts IDLE. You have to separately POST to /api/start to make it go.
- Gary the building manager would have NO IDEA he needs to do this.
  The web page loads but the game is dead until you hit a hidden API endpoint.
- PARTIAL PASS (endpoint works, but default state is confusing)

STEP 5: Does starting the game work?
- Tried: POST /api/start with {"preset":"urban_combat"}
- Result: {"status":"started","preset":"urban_combat","modules":17}
- 17 simulation modules activated.
- PASS

STEP 6: Does /api/status show real data after start?
- Result: running=true, tick_count=26, 14 total units, 3 vehicles,
  50 crowd members, environment="Rain night, calm, 20 degrees C"
- Factions: gov, reb, civ
- PASS — real data coming through

STEP 7: Does the WebSocket stream data?
- Before game started: connected but got NOTHING for 5 seconds. Timeout.
- After game started: connected, immediately got tick frames.
- Frame contents: tick number, time, units (14), projectiles, effects,
  weather, terrain, crowd, vehicles, detection, comms, medical, logistics,
  naval, air_combat, morale, electronic_warfare, supply_routes, objectives,
  influence, territory, stats — 35 keys per frame.
- Units have: id, x, y, z, heading, type, alliance, color, health, status, label
- Example unit: Alpha_0, infantry, friendly (green #05ffa1), health=1.0, status=moving
- Consecutive ticks (71, 72) showed the unit position changing — it is actually moving.
- PASS

STEP 8: Other API endpoints
- /api/stats: Returns leaderboard with per-unit stats (kills, deaths, accuracy,
  distance moved, time alive). Working.
- /api/presets: Returns 5 world presets, 4 scenario presets, 5 campaign presets,
  plus vehicle templates. Working.
- /api/pause (POST): Paused the simulation. Returned {"paused":true}.
  By tick 118 one friendly unit had already died (dead=1). Working.
- Unpause also worked.
- PASS on all tested endpoints

STEP 9: Kill the server
- kill $(lsof -ti:9090) — server killed cleanly.
- PASS

---

WEBSITE: works
MAP: not checked (this is a headless API test, no browser opened for game_server)
TARGETS ON MAP: N/A — confirmed via WebSocket: 14 units streaming with positions
CLICKING: not tested (no browser)
DEMO MODE: game starts idle, requires POST to /api/start — not automatic
APIs: 6 of 6 returned real data

OBVIOUS PROBLEMS:
1. The game server starts in an IDLE state. The web page loads but nothing
   happens until you POST to /api/start. Gary would stare at a frozen screen
   and have no idea why. There is no "Start Game" button visible from the outside
   — you have to know the API endpoint exists.
2. WebSocket sends nothing when game is idle. A confused user would think
   the app is broken, not that they need to hit a start endpoint first.
3. No auto-start. Visiting / in a browser does not start a simulation.
   (The /city route does auto-start CitySim — but the main / route does not.)

THINGS THAT SEEM TO WORK:
1. Server starts cleanly with no errors.
2. HTML page loads correctly on port 9090.
3. WebSocket streams rich frame data at ~10 ticks/second once game is running.
4. All tested API endpoints (/api/status, /api/start, /api/pause, /api/stats,
   /api/presets) return correct data.
5. Simulation actually simulates: units move, units die, environment is dynamic.
6. Pause and unpause work correctly.
7. 17 modules active simultaneously without crash.

MY HONEST IMPRESSION:
The backend engine is genuinely impressive — 35 data channels per frame,
units actually moving and dying, weather and logistics and naval and air all
running at once. But a normal person visiting the web page would see nothing
and assume it is broken, because the game does not start itself and there is
no visible button or prompt telling you what to do next.
