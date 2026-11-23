import gzip
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluate_proxy_teacher_distribution.py"
)
_SPEC = importlib.util.spec_from_file_location("proxy_teacher_distribution", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
analyze_proxy_teacher_arrays = _MODULE.analyze_proxy_teacher_arrays
load_development_evidence = _MODULE.load_development_evidence


def _case(case_id: str) -> dict:
    return {"case_id": case_id, "munition": {"profile_id": "small_front"}}


def test_loader_skips_test_result_before_json_decoding(tmp_path):
    pilot = tmp_path / "pilot"
    round_dir = pilot / "rounds" / "round-000"
    shard = pilot / "formal_dataset" / "shards" / "part-000000"
    round_dir.mkdir(parents=True)
    shard.mkdir(parents=True)
    (pilot / "formal_dataset" / "manifest.json").write_text(
        json.dumps({"dataset_id": "toy", "engine": {}, "source_contracts": {}}),
        encoding="utf-8",
    )
    cases = [_case("train-1"), _case("validation-1"), _case("test-1")]
    with gzip.open(round_dir / "formal_cases.jsonl.gz", "wt", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case) + "\n")
    with (shard / "tensors.npz").open("wb") as stream:
        np.savez_compressed(
            stream,
            case_ids=np.asarray(["train-1", "validation-1", "test-1"]),
            split_names=np.asarray(["train", "validation", "test"]),
            sampling_design_weight_raw=np.asarray([1.0, 2.0, 3.0]),
        )
    with gzip.open(shard / "results.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"case_id": "train-1"}) + "\n")
        stream.write(json.dumps({"case_id": "validation-1"}) + "\n")
        # This line is deliberately not valid JSON.  A passing test proves the
        # loader branches on frozen split metadata before decoding test labels.
        stream.write("THIS TEST RESULT MUST NOT BE DECODED\n")

    rows, audit = load_development_evidence(pilot)

    assert [row.case["case_id"] for row in rows] == ["train-1", "validation-1"]
    assert [row.design_weight for row in rows] == [1.0, 2.0]
    assert audit["test_row_count"] == 1
    assert audit["test_result_json_lines_decoded"] == 0
    assert audit["test_metrics_computed"] is False
    assert audit["test_open_count_increment"] == 0


def test_hurdle_distribution_metrics_include_strata_and_positive_part():
    proxy_hit = np.asarray([[0.0, 0.5], [0.2, 0.8], [0.4, 0.9]])
    teacher_hit = np.asarray([[0.0, 0.4], [0.3, 0.7], [0.5, 1.0]])
    proxy_energy = np.asarray([[0.0, 4.0], [2.0, 8.0], [3.0, 9.0]])
    teacher_energy = np.asarray([[0.0, 5.0], [1.0, 7.0], [4.0, 10.0]])
    mask = np.ones_like(proxy_hit, dtype=bool)

    report = analyze_proxy_teacher_arrays(
        proxy_hit_probability=proxy_hit,
        teacher_hit_probability=teacher_hit,
        proxy_expected_energy_j=proxy_energy,
        teacher_mean_deposited_energy_j=teacher_energy,
        observed_mask=mask,
        case_weights=np.asarray([1.0, 1.0, 2.0]),
        cells=["top::small", "side::small", "side::small"],
    )

    assert report["identified_case_component_count"] == 6
    assert report["positive_energy"]["paired_positive_count"] == 5
    assert report["positive_energy"]["log1p_wasserstein1"] >= 0.0
    assert report["multivariate_hit_probability"]["mmd2"] >= 0.0
    assert set(report["strata"]) == {"side::small", "top::small"}
    assert 0.0 <= report["hit_probability"]["jensen_shannon_fixed_bins"] <= np.log(2.0)
