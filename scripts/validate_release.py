"""Validate benchmark manifest integrity and generated-document synchronization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.release_docs import (  # noqa: E402
    README_END,
    README_START,
    render_model_card,
    render_readme_results,
    render_technical_report,
)


def validate_release(release_dir: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = release_dir / "manifest.json"
    if not manifest_path.exists():
        return [f"missing {manifest_path}"]
    manifest = json.loads(manifest_path.read_text())
    required = {"release_id", "protocol", "data_quality", "cohort", "models", "selection"}
    missing = required - set(manifest)
    if missing:
        errors.append(f"manifest missing keys: {sorted(missing)}")
        return errors
    if not manifest["data_quality"]["ready"]:
        errors.append("data_quality.ready is false")
    champion = manifest["selection"]["champion"]["model"]
    if champion not in manifest["models"]:
        errors.append(f"champion {champion!r} is absent from models")
    if champion not in manifest["selection"]["equivalence_set"]:
        errors.append("champion is absent from equivalence_set")
    artifact = manifest.get("production_artifact")
    if not artifact:
        errors.append("production_artifact is missing")
    elif artifact["data_fingerprint"] != manifest["data_quality"]["data_fingerprint"]:
        errors.append("artifact and release data fingerprints differ")

    model_card = Path("MODEL_CARD.md")
    if not model_card.exists() or model_card.read_text() != render_model_card(manifest):
        errors.append("MODEL_CARD.md is not synchronized with manifest.json")
    technical = release_dir / "technical_report.md"
    if not technical.exists() or technical.read_text() != render_technical_report(manifest):
        errors.append("technical_report.md is not synchronized with manifest.json")
    readme = Path("README.md")
    if not readme.exists():
        errors.append("README.md is missing")
    else:
        text = readme.read_text()
        if README_START not in text or README_END not in text:
            errors.append("README benchmark release markers are missing")
        elif render_readme_results(manifest) not in text:
            errors.append("README benchmark block is not synchronized with manifest.json")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, default=Path("reports/releases/benchmark-v1"))
    args = parser.parse_args()
    errors = validate_release(args.release_dir)
    if errors:
        raise SystemExit("\n".join(f"- {error}" for error in errors))
    print(f"release validation passed: {args.release_dir}")


if __name__ == "__main__":
    main()
