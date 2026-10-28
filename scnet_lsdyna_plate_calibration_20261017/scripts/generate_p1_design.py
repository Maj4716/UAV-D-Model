from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "design" / "p1_preregistered_cases.csv"
SEED = 20261017
N_CASES = 24
MANIFEST = ROOT / "manifest.json"


def latin_hypercube(rng: np.random.Generator, n: int, dimensions: int) -> np.ndarray:
    result = np.empty((n, dimensions), dtype=float)
    for dim in range(dimensions):
        result[:, dim] = (rng.permutation(n) + rng.random(n)) / n
    return result


def balanced_categorical_rows(role: str) -> list[tuple[float, str, str]]:
    """Return preregistered categorical strata with exact role-wise margins.

    The 18 fit rows cover every mass/orientation pair twice and every level of
    each categorical factor six times.  The six holdout rows use six distinct
    mass/orientation pairs and contain every level exactly twice.  Material is
    rotated independently so it is not a proxy for mass or orientation.
    """
    masses = [5.0, 8.0, 10.0]
    orientations = ["face", "edge", "corner"]
    materials = ["low", "central", "high"]
    if role == "hf_response_fit":
        rows: list[tuple[float, str, str]] = []
        for repeat in range(2):
            for mass_index, mass in enumerate(masses):
                for orientation_index, orientation in enumerate(orientations):
                    material = materials[(mass_index + 2 * orientation_index + repeat) % 3]
                    rows.append((mass, orientation, material))
        return rows
    if role == "hf_response_holdout":
        pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]
        material_indices = [2, 0, 1, 1, 2, 0]
        return [
            (masses[mass_index], orientations[orientation_index], materials[material_index])
            for (mass_index, orientation_index), material_index in zip(pairs, material_indices, strict=True)
        ]
    raise ValueError(f"unknown role: {role}")


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    next_case_number = 1
    for role, n_role in (("hf_response_fit", 18), ("hf_response_holdout", 6)):
        lhs = latin_hypercube(rng, n_role, 3)
        categorical = balanced_categorical_rows(role)
        order = rng.permutation(n_role)
        for local_index in range(n_role):
            mass, orientation, material = categorical[int(order[local_index])]
            rows.append(
                {
                    "case_id": f"p1_{next_case_number:03d}",
                    "role": role,
                    "fragment_mass_g": f"{mass:.1f}",
                    "impact_speed_mps": f"{1200.0 + 1400.0 * lhs[local_index, 0]:.6f}",
                    "plate_thickness_mm": f"{6.0 + 8.0 * lhs[local_index, 1]:.6f}",
                    "obliquity_deg_from_normal": f"{60.0 * lhs[local_index, 2]:.6f}",
                    "fragment_orientation": orientation,
                    "material_variant": material,
                    "mesh_policy": "p0_selected_after_numerical_qa",
                    "design_seed": SEED,
                    "status": "preregistered_not_submitted"
                }
            )
            next_case_number += 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["p1_design"] = {
            "path": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(OUTPUT),
            "case_count": len(rows),
            "status": "preregistered_not_submitted",
        }
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} preregistered cases to {OUTPUT}")


if __name__ == "__main__":
    main()
