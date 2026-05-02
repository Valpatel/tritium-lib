# Tritium-Lib Changelog

Changes tracked with verification status. All changes on `dev` branch.

## Verification Levels

| Level | Meaning |
|-------|---------|
| **Unit Tested** | Passes `pytest tests/` |
| **Consumer Tested** | Verified working in tritium-edge or tritium-sc imports |
| **Human Verified** | Manually reviewed by a human |

---

## 2026-04-28 — Wave 200 Security: NaN/Inf rejection in update_from_rf_motion

| Change | Verification |
|--------|-------------|
| `target_tracker.update_from_rf_motion`: rejects NaN/Inf positions (slip past (0,0) check because `NaN == 0.0` is False) | Unit Tested |
| Defensive float coercion catches TypeError/ValueError/IndexError for malformed dict/tuple inputs | Unit Tested |
| Test: `tests/tracking/test_security_wave200.py` — 7/7 PASS | Unit Tested |
| Audit: HIGH W200-H2 fix — see `docs/security/wave-200-audit.md` (parent repo) | Documented |

---

## 2026-03-21 — Wave 186: Maintenance

| Change | Verification |
|--------|-------------|
| Created `docs/STATUS.md` with Wave 186 baselines: 227 test files, 146 sim_engine files, 343 total src files | Manual |
| Redundancy scan: `steering.py` vs `steering_np.py` confirmed intentional (pure Python vs NumPy vectorized) | Manual |
| Redundancy scan: zero TODO/FIXME/HACK comments in sim_engine | Manual |
| Agent Trigger Schedule updated: maintenance/feature/visual-testing/quality set to Wave 186 | Manual |
| Wave 185 CHANGELOG entry added: buildings fix, combat balance, kill attribution, weather HUD | Manual |

---


---

_Older entries archived to `docs/local/CHANGELOG-archive.md`._
