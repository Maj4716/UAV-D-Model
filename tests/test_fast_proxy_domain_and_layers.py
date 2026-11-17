from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from damage_engine.engine import DamageEngine
from damage_engine.fast_sampling_proxy import FastSamplingProxy
from damage_engine.registry import PhysicsRegistry
from damage_engine.utils import content_hash


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_documents() -> tuple[dict, dict, dict]:
    target = _load(ROOT / "configs/targets/target.fragment_only.v1.json")
    registry = _load(ROOT / "configs/physics/loitering_impact.json")
    case = _load(ROOT / "configs/cases/loitering_impact.fragment_only.v1.json")
    case["analysis"]["capture_fragment_audit"] = False
    case["analysis"]["capture_replicate_trace"] = False
    return target, registry, case


def test_fast_proxy_publishes_normalized_domain_censoring_contract():
    target, registry_data, case = _fixture_documents()
    result = FastSamplingProxy(
        DamageEngine(target, PhysicsRegistry(registry_data)), azimuths_per_ring=2
    ).evaluate(case)

    domain = result["proxy_domain_coverage"]
    assert domain["represented_packet_weight"] == result["represented_fragment_count"]
    assert domain["out_of_domain_packet_weight"] == result[
        "out_of_domain_packet_weight"
    ]
    assert 0.0 <= domain["out_of_domain_packet_fraction"] <= 1.0
    expected_status = (
        "complete_proxy_domain"
        if result["out_of_domain_packet_weight"] == 0.0
        else "partial_proxy_domain_censored"
    )
    assert result["proxy_domain_status"] == expected_status
    assert result["formal_label_admissible"] is False


def test_fast_proxy_honors_declared_layer_stack_with_shared_penetration_kernel():
    target, registry_data, case = _fixture_documents()
    base = FastSamplingProxy(
        DamageEngine(deepcopy(target), PhysicsRegistry(deepcopy(registry_data))),
        azimuths_per_ring=2,
    ).evaluate(deepcopy(case))

    strong_model = deepcopy(registry_data["fragment_response_models"]["demo_armor_response"])
    strong_model["ballistic_limit"]["coefficient"] = 1.0e8
    registry_data["fragment_response_models"]["proxy-test-strong"] = strong_model
    registry_data["layer_stack_models"] = {
        "proxy-test-stack": {
            "layers": [
                {
                    "material_id": "demo_steel",
                    "fragment_response_model_id": "demo_armor_response",
                    "thickness_fraction": 0.5,
                },
                {
                    "material_id": "demo_steel",
                    "fragment_response_model_id": "proxy-test-strong",
                    "thickness_fraction": 0.5,
                },
            ],
            "source_id": "TEST-ONLY",
        }
    }
    for armor in target["armor"]:
        armor["response"]["layer_stack_id"] = "proxy-test-stack"
    case["target_ref"]["content_hash"] = content_hash(target)

    layered = FastSamplingProxy(
        DamageEngine(target, PhysicsRegistry(registry_data)),
        azimuths_per_ring=2,
    ).evaluate(case)

    assert layered["penetration_kernel"] == "resolve_penetration_path"
    assert layered["layered_interaction_count"] > 0
    assert (
        layered["expected_total_component_hits"]
        < base["expected_total_component_hits"]
    )
