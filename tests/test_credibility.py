from pathlib import Path

import pytest

from damage_engine.credibility import build_credibility_dossier, expert_assertion_template
from damage_engine.errors import InputValidationError


def test_expert_template_reserves_authority_instead_of_inventing_claims():
    assertions = expert_assertion_template()
    assert len(assertions) == 10
    assert all(item["claim"] is None for item in assertions)


def test_c4_cannot_pass_with_blank_assertions(tmp_path: Path):
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("verified", encoding="utf-8")
    categories = [
        {
            "category_id": name,
            "status": "pass" if name != "C4" else "pass",
            "evidence_paths": ["evidence.txt"],
        }
        for name in ("C1", "C2", "C3", "C4", "C5", "C6")
    ]
    plan = {"categories": categories, "expert_assertions": expert_assertion_template()}
    with pytest.raises(InputValidationError, match="ten complete"):
        build_credibility_dossier(plan, workspace_root=tmp_path)
