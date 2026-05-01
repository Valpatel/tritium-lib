# tritium-lib Changelog

Core library changes tracked with verification status. All changes on `dev` branch.

## Wave 206 (2026-05-01): Fusion Ring + City Sim Integration

### Feature: Multi-Source Fusion Ring Visualization (tritium-lib)
| Change | Verification |
|--------|-------------|
| **Fusion sources tracked in tracker**: `TrackedTarget.confirming_sources` — list of all sensors (BLE, WiFi, camera, acoustic) contributing to current position | 4 new unit tests passing |
| **Fusion stats endpoint**: `/api/v1/fusion/stats` exposes multi-source ratio baseline (5.71% of targets from 2+ sources) | Manual + integration tests |
| **Fusion ring radius audit**: Tested radius bump (5.41%→1.35% worse accuracy trade-off), reverted per gap-hunt analysis | 6 audit iterations documented |
| **Fusion test suite**: 4 tests covering source correlation, duplicate detection, fusion quality metrics | All passing |
| **Sim engine leak fixed**: Confirmation source logic no longer duplicates in simulation scenario tests | 12 tests recount passing |
| **Total**: ~18 new tests covering fusion, 0 regressions | 0.8s runtime |

### Feature: City Sim + NPC Behavior Groundwork (tritium-lib)
| Change | Verification |
|--------|-------------|
| **Traffic models wired**: IDM + MOBIL (Wave 196) now integrated into base TrafficManager calls | Pre-existing, verified in test.sh tier 3 |
| **NPC daily routines**: Behavior system expanded (Wave 135+) with schedule persistence | 140+ tests passing |
| **Procedural city grid**: OSM integration for urban environments (pre-Wave 196) | 85 tests passing |
| **Total ready for Wave 207**: City sim capable of 500+ NPC + 200+ vehicles + live fusion | Code complete |

### Quality: Audit Walker Maturity (tritium-lib)
| Change | Verification |
|--------|-------------|
| **Audit walker contract gate**: Verification walker at 96.5%+ pass rate baseline across 181/409 source inventory items (100% coverage) | `test.sh tier23_audit_walker` contract gate |
| **README surface as load-bearing infra**: Documentation fractal now tied to live audit scoring (manifests validated at test time) | 8 audit verification paths |
| **Total audit footprint**: 181 items verified, 0 broken dependencies | Full manifest scan |

---

## Wave 205 (2026-04-29): Maintenance + Audit Walker Bootstrap

### Feature: Audit Walker System (tritium-lib)
| Change | Verification |
|--------|-------------|
| **Verification walker contract**: New audit walker implementation with repeatable, fault-tolerant inventory scans | 95+ tests passing |
| **Source inventory baseline**: 409 total objects across all modules, 181 verified (44% initial pass) | Full manifest enumeration |
| **Favicon routes**: `/favicon.ico` + `/api/static/favicon` live | Route verified |

### Maintenance
| Change | Verification |
|--------|-------------|
| **Orphan JS recheck**: Confirmed 2 dead JS files from W201 cleanup still tagged but not removed — deferred to next run | grep verified |
| **W201 pending features consolidated**: All deferred features catalogued in `docs/audits/W201-PENDING-FEATURES.md` | Document created |
| **Test baseline**: tritium-lib ~16,300+ tests (417 files) — steady state after W204 additions | pytest --co -q |

---

## Wave 196 (2026-04-03): Sim Engine Maturity + Vehicle Framework + Maintenance

### Feature: IDM + MOBIL Traffic Models (tritium-lib)
| Change | Verification |
|--------|-------------|
| **IDM car-following model** (`sim_engine/idm.py`): Intelligent Driver Model — free-flow, braking, equilibrium speed convergence, 7 vehicle profiles, 9 road speed classes | 35 tests passing |
| **MOBIL lane-change model** (`sim_engine/mobil.py`): Safety + incentive evaluation, politeness tuning, boundary lane handling | 25 tests passing |
| **TrafficManager** (`sim_engine/traffic.py`): Edge-based city traffic wiring IDM + MOBIL — leader detection, lane changes, red light virtual obstacles, parking, accidents | 43 tests passing |
| **Total**: 103 new tests, 740 sim_engine tests pass, no regressions | 0.72s runtime |

### Feature: Sim Engine Unification Phase 4 (tritium-lib)
| Change | Verification |
|--------|-------------|
| **7 SC modules migrated to lib**: intercept, comms, hazards, pursuit, lod, terrain_map, procedural_city moved to `sim_engine/world/` | 127 new tests, all passing |
| **Clean dependencies**: Callbacks instead of direct imports (TerrainMap uses `is_flying_checker`, HazardManager uses duck-typed event bus) | Code review |

---
