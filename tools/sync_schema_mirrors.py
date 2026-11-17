"""Verify or regenerate the documented external schema mirrors."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


MIRRORED_SCHEMA_NAMES = (
    "target.schema.json",
    "simulation_case.schema.json",
    "sampling_spec.schema.json",
    "label_spec.schema.json",
    "dataset_manifest.schema.json",
)


def synchronize_schema_mirrors(
    repository_root: Path,
    *,
    write: bool = False,
) -> list[str]:
    canonical = repository_root / "damage_engine" / "schemas"
    mirror = repository_root / "schemas"
    drift: list[str] = []
    for name in MIRRORED_SCHEMA_NAMES:
        source = canonical / name
        destination = mirror / name
        if not destination.exists() or destination.read_bytes() != source.read_bytes():
            drift.append(name)
            if write:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
    return drift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    drift = synchronize_schema_mirrors(root, write=args.write)
    if drift and not args.write:
        print("schema mirror drift: " + ", ".join(drift))
        return 1
    if args.write:
        remaining = synchronize_schema_mirrors(root, write=False)
        if remaining:
            print("schema mirror regeneration failed: " + ", ".join(remaining))
            return 1
    print("schema mirrors synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
