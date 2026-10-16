"""Build the derived fragment-only loitering-impact dataset contracts.

The source target remains untouched.  This script creates a separately
identified target whose damage-tree leaves consume fragment probabilities,
then binds a label specification, collision-triggered base case, and a
5,000-case outcome-blind sampling plan to that derived target.

This script only writes configuration JSON.  It does not initialize or run a
sampling archive.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.engine import DamageEngine
from damage_engine.label_spec import LabelSpec
from damage_engine.normalization import normalize_target
from damage_engine.registry import PhysicsRegistry
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.utils import content_hash
from damage_engine.validation import validate_inputs


RAW_DATA = ROOT / "raw_data"
SOURCE_TARGET = RAW_DATA / "target.rebuilt.json"
SOURCE_LABEL_SPEC = RAW_DATA / "label_spec.loitering_impact.v1.json"
SOURCE_CASE = RAW_DATA / "case.loitering_impact.v1.json"
SOURCE_SAMPLING_SPEC = RAW_DATA / "sampling_spec.loitering_impact.smoke.v1.json"
PHYSICS_REGISTRY = ROOT / "examples" / "physics.loitering_impact.json"

OUTPUT_TARGET = RAW_DATA / "target.rebuilt.fragment_only.v1.json"
OUTPUT_LABEL_SPEC = RAW_DATA / "label_spec.loitering_impact.fragment_only.v1.json"
OUTPUT_CASE = RAW_DATA / "case.loitering_impact.fragment_only.v1.json"
OUTPUT_SAMPLING_SPEC = (
    RAW_DATA / "sampling_spec.5000.loitering_impact.fragment_only.v1.json"
)

TARGET_ID = "raw-armored-target-fragment-only-v1"
TARGET_REVISION = "raw-2026-09-28-fragment-only-v1"
DAMAGE_TREE_VERSION = "raw-damage-tree-fragment-only-v1"
STANDARD_ID = "raw-damage-codes-fragment-only-provisional"
PROFILE_IDS = ["small_front", "medium_front", "large_front"]


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _derive_target(source: dict) -> tuple[dict, int]:
    target = deepcopy(source)
    target["target_id"] = TARGET_ID
    target["target_revision"] = TARGET_REVISION

    tree = target.get("damage_tree")
    if not isinstance(tree, dict):
        raise ValueError("source target has no damage_tree object")
    tree["version"] = DAMAGE_TREE_VERSION

    leaf_count = 0
    for node in tree.get("nodes", []):
        if node.get("node_type") != "leaf":
            continue
        if node.get("mechanism") not in {"combined", "fragment"}:
            raise ValueError(
                "fragment-only derivation encountered an unsupported leaf mechanism: "
                f"{node.get('id')}={node.get('mechanism')}"
            )
        node["mechanism"] = "fragment"
        leaf_count += 1
    if leaf_count == 0:
        raise ValueError("source target damage tree contains no leaf nodes")

    standards = tree.get("standards")
    if not isinstance(standards, list) or len(standards) != 1:
        raise ValueError("expected exactly one provisional source standard")
    standards[0]["id"] = STANDARD_ID
    standards[0]["name"] = "原始毁伤代码破片单机制临时映射"
    standards[0]["description"] = (
        "派生配置：系统事件只由破片部件损伤概率驱动；"
        "原始阈值到部件状态的映射仍属临时、未验证定义。"
    )

    metadata = target.setdefault("metadata", {})
    metadata["derived_configuration"] = {
        "derivation_id": "fragment-only-collision-impact-v1",
        "source_target": "raw_data/target.rebuilt.json",
        "change_scope": "damage_tree_leaf_mechanism_only",
        "leaf_mechanism": "fragment",
        "blast_response_retained_for_schema_compatibility": True,
        "scientific_status": "unverified_provisional",
    }
    return target, leaf_count


def _derive_label_spec(source: dict, target: dict) -> dict:
    spec = deepcopy(source)
    spec["label_spec_id"] = "raw-armored-target-fragment-only-damage-events-v1"
    spec["revision"] = "1.0.0-provisional"
    spec["status"] = "provisional"
    spec["source"] = {
        "damage_tree_id": f"{TARGET_ID}/damage-tree",
        "damage_tree_revision": DAMAGE_TREE_VERSION,
        "damage_tree_definition_hash": content_hash(target["damage_tree"]),
    }
    task = spec["tasks"]["raw_damage_events"]
    task["description"] = (
        "Non-exclusive provisional armored-target damage-code events driven "
        "only by fragment component damage probabilities."
    )
    spec["provenance"] = {
        "approval_status": "not_approved",
        "evidence_ref": "raw_data/target.rebuilt.fragment_only.v1.json",
    }
    return spec


def _derive_case(source: dict, target: dict) -> dict:
    case = deepcopy(source)
    case["case_id"] = "loitering-impact-fragment-only-template-v1"
    case["target_ref"] = {
        "target_id": TARGET_ID,
        "target_revision": TARGET_REVISION,
        "content_hash": content_hash(normalize_target(target)),
    }

    encounter = case["encounter"]
    encounter["detonation_trigger"] = "first_armor_contact"
    encounter["detonation_delay_s"] = 0.0
    encounter["trajectory_reference_point"] = "nose_fuze"

    analysis = case["analysis"]
    analysis["mechanisms"] = ["fragment"]
    analysis["coupling_model"] = "none"
    analysis["replicates"] = 2
    analysis["adaptive_ci_half_width"] = None
    analysis.pop("adaptive_physical_ci", None)
    analysis["evaluation_standard_ids"] = [STANDARD_ID]
    analysis["epistemic_sample_id"] = case["case_id"]

    metadata = case.setdefault("metadata", {})
    metadata["split_group"] = case["case_id"]
    metadata["impact_semantics"] = (
        "trajectory_origin_m is the nose-fuze point at t=0; detonation occurs "
        "at the first armor intersection with zero delay"
    )
    metadata["mechanism_scope"] = (
        "fragment-only execution; blast_source is retained only because the "
        "current simulation-case schema requires a complete warhead document"
    )
    metadata["warning"] = (
        "Software verification only; target response, damage-code mapping, and "
        "munition parameters remain unvalidated"
    )
    return case


def _derive_sampling_spec(source: dict) -> dict:
    spec = deepcopy(source)
    spec["schema_version"] = "1.2.0"
    spec["model_input_contract"] = "loitering_impact_state_only@1.0.0"
    spec["sampling_spec_id"] = (
        "loitering-impact-three-profile-fragment-only-5000-v1"
    )
    spec["revision"] = "1.0.0-provisional"
    spec["status"] = "provisional"

    distribution = spec["reference_distribution"]
    distribution["distribution_id"] = (
        "loitering-impact-three-profile-fragment-only-design-v1"
    )
    profile_variable = next(
        item for item in distribution["variables"] if item["name"] == "munition_profile"
    )
    profile_variable["values"] = list(PROFILE_IDS)

    # The learner sees exactly six pre-impact design values: profile one-hot3,
    # trajectory-origin y/z, and positive-x impact speed.  Collision solutions,
    # target/armor state, and solver/teacher controls stay outside the model
    # input contract.
    allowed_feature_ids = {
        "munition_profile",
        "trajectory_origin_y",
        "trajectory_origin_z",
        "impact_speed",
    }
    spec["feature_roles"] = [
        item
        for item in spec["feature_roles"]
        if item["feature_id"] in allowed_feature_ids
    ]

    spec["budgets"] = {
        "train_pilot": 3000,
        "validation": 500,
        "test": 1500,
        "rare_stress_test": 0,
    }
    spec["seeds"] = {
        "design": 2026101411,
        "selection": 2026101412,
        "split": 2026101413,
        "screening": 2026101414,
        "formal": 2026101415,
    }
    spec["formal_policy"] = {
        "replicates": 2,
        "adaptive_ci_half_width": None,
        "epistemic_policy": "unique_per_lineage",
    }
    spec["split_policy"]["registry_id"] = (
        "loitering-impact-three-profile-fragment-only-5000-splits-v1"
    )
    spec["round_budgets"] = [
        {
            "round": 0,
            "allocations": [
                {
                    "role": "reference_validation",
                    "stratum_id": "p0",
                    "proposal_id": distribution["distribution_id"],
                    "count": 500,
                },
                {
                    "role": "reference_test",
                    "stratum_id": "p0",
                    "proposal_id": distribution["distribution_id"],
                    "count": 1500,
                },
                {
                    "role": "reference_train",
                    "stratum_id": "p0",
                    "proposal_id": distribution["distribution_id"],
                    "count": 3000,
                },
            ],
        }
    ]
    spec["replicate_policy"] = {
        "default_replicates": 2,
        "outcome_adaptive": False,
        "overrides": [],
    }
    # Boundary samples can miss the imported armor.  Reject them using only
    # pre-label geometry and deterministically resample within a bounded budget.
    spec["legal_sampling"] = {
        "invalid_candidate_policy": "reject_and_resample",
        "max_attempts_per_allocation": 50000,
    }
    return spec


def main() -> int:
    source_target = _load(SOURCE_TARGET)
    source_label_spec = _load(SOURCE_LABEL_SPEC)
    source_case = _load(SOURCE_CASE)
    source_sampling_spec = _load(SOURCE_SAMPLING_SPEC)

    target, leaf_count = _derive_target(source_target)
    label_spec = _derive_label_spec(source_label_spec, target)
    case = _derive_case(source_case, target)
    sampling_spec = _derive_sampling_spec(source_sampling_spec)

    # Validate all four documents before writing any output.
    label_contract = LabelSpec(label_spec)
    sampling_contract = SamplingSpec(sampling_spec)
    registry = PhysicsRegistry.from_file(PHYSICS_REGISTRY)
    engine = DamageEngine(target, registry)
    label_contract.validate_output_vocabulary(engine.damage_tree.outputs)
    validate_inputs(
        engine.target_data,
        engine.prepare_case(case),
        registry,
        compiled_target=engine.compiled_target,
    )
    if sum(sampling_contract.budgets.values()) != 5000:
        raise ValueError("sampling budgets must sum to exactly 5000")
    if sampling_contract.formal_replicates != 2:
        raise ValueError("formal replicate count must be exactly 2")

    _write(OUTPUT_TARGET, target)
    _write(OUTPUT_LABEL_SPEC, label_spec)
    _write(OUTPUT_CASE, case)
    _write(OUTPUT_SAMPLING_SPEC, sampling_spec)

    print(
        json.dumps(
            {
                "status": "ok",
                "outputs": [
                    str(OUTPUT_TARGET.relative_to(ROOT)),
                    str(OUTPUT_LABEL_SPEC.relative_to(ROOT)),
                    str(OUTPUT_CASE.relative_to(ROOT)),
                    str(OUTPUT_SAMPLING_SPEC.relative_to(ROOT)),
                ],
                "fragment_leaf_count": leaf_count,
                "profiles": PROFILE_IDS,
                "budgets": sampling_contract.budgets,
                "total_cases": sum(sampling_contract.budgets.values()),
                "replicates": sampling_contract.formal_replicates,
                "target_hash": engine.target_hash,
                "damage_tree_definition_hash": content_hash(
                    engine.target_data["damage_tree"]
                ),
                "label_spec_hash": label_contract.hash,
                "sampling_spec_hash": sampling_contract.hash,
                "dataset_executed": False,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
