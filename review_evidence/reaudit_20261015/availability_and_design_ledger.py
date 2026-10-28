"""Descriptive storage/denominator checks, without outcome-condition analysis."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).absolute().parents[2]
OUT = Path(__file__).parent
DATA = ROOT / "raw_data/dataset_5000_loitering_impact_fragment_only_v1"


def main():
    formal = DATA / "formal_dataset"
    manifest_path = formal / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_input = manifest["model_input_contract"]
    summaries = defaultdict(Counter)
    role_counts = defaultdict(Counter)
    input_masks, input_values = [], []
    splits, identifiers, system_values, system_masks = [], [], [], []
    repeat_counts, weight_masks = Counter(), Counter()
    fields = ("component", "component_frechet", "component_aux", "armor", "system")
    for shard in sorted((formal / "shards").iterdir()):
        with np.load(shard / "tensors.npz", allow_pickle=False) as arrays:
            names = np.asarray(arrays["split_names"]).astype(str)
            roles = np.asarray(arrays["sampling_role_names"])[arrays["sampling_role_index"]].astype(str)
            for split, role in zip(names, roles):
                role_counts[split][role] += 1
            for stem in fields:
                value = arrays[stem + "_y"]
                available = arrays[stem + "_label_mask"].astype(bool)
                uncertainty = arrays[stem + "_uncertainty_mask"].astype(bool)
                counter = summaries[stem]
                counter["all_positions"] += int(value.size)
                counter["available_positions"] += int(np.count_nonzero(available))
                counter["unavailable_positions"] += int(np.count_nonzero(~available))
                counter["available_exact_zero"] += int(np.count_nonzero(available & (value == 0)))
                counter["available_nonfinite"] += int(np.count_nonzero(available & ~np.isfinite(value)))
                counter["unavailable_zero_placeholders"] += int(np.count_nonzero(~available & (value == 0)))
                counter["uncertainty_estimable_positions"] += int(np.count_nonzero(uncertainty))
                counter["uncertainty_on_unavailable"] += int(np.count_nonzero(uncertainty & ~available))
            values = arrays[model_input["tensor_field"]].copy()
            masks = arrays[model_input["mask_field"]].astype(bool)
            input_values.append(values)
            input_masks.append(masks)
            splits.extend(names.tolist())
            identifiers.extend(arrays["case_ids"].astype(str).tolist())
            system_values.append(arrays["system_y"].copy())
            system_masks.append(arrays["system_label_mask"].astype(bool))
            repeat_counts.update(map(int, arrays["replicate_counts"].tolist()))
            weight_masks.update(map(str, arrays["sampling_weight_mask"].tolist()))
    values = np.concatenate(input_values)
    masks = np.concatenate(input_masks)
    sys_values, sys_masks = np.concatenate(system_values), np.concatenate(system_masks)
    split_array = np.asarray(splits)
    fingerprints = defaultdict(set)
    fingerprint_rows = Counter()
    for i, (value, mask) in enumerate(zip(values, masks)):
        canonical = np.where(mask, value, 0).astype("<f4", copy=False)
        identity = hashlib.sha256(canonical.tobytes() + mask.tobytes()).hexdigest()
        fingerprints[identity].add(splits[i])
        fingerprint_rows[identity] += 1
    train = split_array == "train"
    constant_zero_train = []
    no_available_train = []
    for column in range(sys_values.shape[1]):
        use = train & sys_masks[:, column]
        observed = sys_values[use, column]
        if not observed.size:
            no_available_train.append(column)
        elif np.all(observed == 0):
            constant_zero_train.append(column)
    output = {
        "review_date": "2026-10-15", "scope": "label availability, input identity and declared-role counts only; no outcome-versus-input analysis or effect optimization",
        "python": sys.executable, "dataset_id": manifest["dataset_id"],
        "manifest_hash": "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "case_count": len(identifiers), "unique_case_count": len(set(identifiers)),
        "split_role_counts": {k: dict(v) for k, v in role_counts.items()},
        "label_storage_summaries": {k: dict(v) for k, v in summaries.items()},
        "system_axis_size": int(sys_values.shape[1]),
        "system_train_constant_zero_column_indices": constant_zero_train,
        "system_train_no_available_column_indices": no_available_train,
        "model_input": {"tensor_field": model_input["tensor_field"], "shape": list(values.shape),
                        "available_positions": int(np.count_nonzero(masks)),
                        "available_nonfinite": int(np.count_nonzero(masks & ~np.isfinite(values))),
                        "constant_columns_among_available": sum(bool(np.any(masks[:, j])) and np.unique(values[masks[:, j], j]).size == 1 for j in range(values.shape[1])),
                        "unique_content_fingerprints": len(fingerprints),
                        "duplicate_input_fingerprints": sum(count > 1 for count in fingerprint_rows.values()),
                        "fingerprints_crossing_splits": sum(len(v) > 1 for v in fingerprints.values()),
                        "fingerprint_precision_boundary": "archived float32 values plus availability mask; exact equality only"},
        "replicate_count_histogram": dict(sorted(repeat_counts.items())),
        "sampling_weight_mask_counts": dict(weight_masks),
        "excluded_analyses": ["attack-condition comparisons", "physical causes of zero outcomes", "resampling for effect improvement", "model fitting"],
        "real_world_reference_available": False,
    }
    (OUT / "availability_and_design_ledger.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
