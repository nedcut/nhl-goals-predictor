# Model Registry

Production artifacts are generated locally and promoted through `models/registry.json`:

```bash
python -m src.benchmark --tune-trials 40
```

Binary model files and the mutable registry are intentionally not committed. The
tracked benchmark release under `reports/releases/benchmark-v1/manifest.json`
records the artifact id, registry version, feature-schema hash, data fingerprint,
holdout metrics, and source revision required to reproduce or audit a promotion.

The API loads the promoted registry artifact by default. `NHL_MODEL_PATH` is an
explicit migration override, but the referenced artifact must still satisfy the
release-grade serving contract.
