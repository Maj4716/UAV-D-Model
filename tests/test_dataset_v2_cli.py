from __future__ import annotations

import json
from pathlib import Path

from damage_engine.offline_tree_candidates import compile_offline_tree_candidate
from test_dataset_v2 import _candidate, make_dataset
from tools.build_dataset_v2 import main as build_main
from tools.select_offline_tree_candidate import main as select_main
from tools.validate_dataset_v2 import main as validate_main


def test_dataset_v2_cli_view_validate_select_and_fallback(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "dataset")
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(_candidate()), encoding="utf-8")
    assert build_main(
        ["add-tree-view", "--dataset", str(dataset), "--candidate", str(candidate_path)]
    ) == 0
    assert validate_main([str(dataset)]) == 0
    report_path = tmp_path / "selection.json"
    assert select_main(
        [
            "--dataset",
            str(dataset),
            "--candidate",
            str(candidate_path),
            "--development-seed",
            "11",
            "--validation-seed",
            "12",
            "--preregistration-id",
            "TREE-CLI-TEST",
            "--output",
            str(report_path),
        ]
    ) == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["promotion_eligible"] is False

    components = tmp_path / "components.json"
    fallback = tmp_path / "fallback.json"
    components.write_text(json.dumps(["1", "23"]), encoding="utf-8")
    assert build_main(
        [
            "identity-fallback",
            "--components",
            str(components),
            "--output",
            str(fallback),
        ]
    ) == 0
    compiled = compile_offline_tree_candidate(
        json.loads(fallback.read_text(encoding="utf-8"))
    )
    assert compiled.authority_status == "not_provided"
