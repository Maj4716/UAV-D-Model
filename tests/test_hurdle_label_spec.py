from pathlib import Path

from damage_engine.label_spec import LabelSpec


ROOT = Path(__file__).resolve().parents[1]


def test_hurdle_v2_label_spec_preserves_tree_view_and_declares_component_core():
    spec = LabelSpec.from_file(
        ROOT / "configs" / "labels" / "loitering_impact.fragment_only.hurdle.v2.json"
    )
    contract = spec.data["hurdle_contract"]
    assert contract["domain_censoring"]["epsilon"] == 0.01
    assert contract["target_encoder_policy"]["label_mutation"] is False
    assert spec.data["status"] == "provisional"


def test_hurdle_v3_declares_target_specific_censoring_contract():
    spec = LabelSpec.from_file(
        ROOT / "configs" / "labels" / "loitering_impact.fragment_only.hurdle.v3.json"
    )
    contract = spec.data["hurdle_contract"]
    assert contract["target_spec_id"] == "component-fragment-hurdle-targets-v2"
    assert contract["domain_censoring"]["epsilon"] == 0.01
    assert contract["uncertainty_separation"]["domain_censoring"] == (
        "component_specific_causal_worst_case_with_global_fallback"
    )
    assert "domain_identification_interval" in contract["derived_support"][
        "decision_rule"
    ]
