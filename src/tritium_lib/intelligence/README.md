# tritium_lib.intelligence

Parent: [tritium_lib](../README.md)

RL-enhanced intelligence pipeline for target correlation, classification, and anomaly detection.

## Key Files

| File | Purpose |
|------|---------|
| `base_learner.py` | Abstract base class for all RL learners |
| `scorer.py` | Target correlation scoring (16-feature model) |
| `feature_engineering.py` | Feature extraction for ML models |
| `anomaly.py` | Anomaly detection module |
| `pattern_learning.py` | Behavioral pattern learning |
| `threat_model.py` | Threat scoring and classification |
| `rl_metrics.py` | RL training metrics tracking (accuracy, feature importance) |
| `fusion_metrics.py` | Per-strategy fusion accuracy tracking |
| `model_registry.py` | Trained model persistence and loading |

## Related

- [tritium_lib.classifier](../classifier/README.md) -- BLE/device classification
- [tritium_lib.ontology](../ontology/README.md) -- Graph entity relationships
- SC intelligence engine: `tritium-sc/src/engine/intelligence/`
