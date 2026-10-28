from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "48_毁伤域采样稀疏性成因研究报告_项目内生性与人为方法因素_2026-10-21.md"
EVIDENCE = ROOT / "review_evidence" / "causal_sparsity_20261021"
OUTPUT = EVIDENCE / "research_report_validation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


text = REPORT.read_text(encoding="utf-8")
ml = json.loads((EVIDENCE / "ml_stats_audit.json").read_text(encoding="utf-8"))
factorial = json.loads(
    (EVIDENCE / "factorial_direction_source_counterfactual.json").read_text(
        encoding="utf-8"
    )
)
axis = json.loads(
    (EVIDENCE / "axis_geometry_necessity_audit.json").read_text(encoding="utf-8")
)

checks: dict[str, bool] = {}
checks["report_exists_nonempty"] = REPORT.is_file() and REPORT.stat().st_size > 20_000
checks["formal_case_count"] = ml["case_count"] == 5000 and "5000" in text
checks["f1_positive_count"] = (
    ml["labels"]["effect-f1"]["all"]["positive_count"] == 28
    and "F1 仅 `28/5000`" in text
)
checks["constant_zero_tasks"] = all(
    ml["labels"][name]["all"]["positive_count"] == 0
    for name in ("effect-k1", "effect-k2", "effect-c1", "effect-c2")
)
checks["direction_and_pose_unique"] = (
    ml["input_support"]["velocity_direction_unique_rows"] == 1
    and ml["input_support"]["orientation_unique_rows"] == 1
)
checks["formal_all_first_contact"] = (
    ml["input_support"]["first_armor_contact_true_count"] == 5000
)

minus = factorial["aggregate"]["by_condition"]["archived_axis_minus_body_x"]
plus = factorial["aggregate"]["by_condition"]["counterfactual_axis_plus_body_x"]
checks["factorial_case_counts"] = minus["case_count"] == plus["case_count"] == 150
checks["factorial_kmfc_counts"] = (
    minus["category_positive_energy_case_count"]
    == {"C": 9, "F": 13, "K": 15, "M": 48}
    and plus["category_positive_energy_case_count"]
    == {"C": 130, "F": 115, "K": 101, "M": 144}
)
checks["axis_trace_is_negative"] = (
    axis["current_dev3_template_fragment_trace"]["all_direction_x_strictly_negative"]
    and axis["observed_train_only_fragment_trace"][
        "all_observed_direction_x_strictly_negative"
    ]
)
checks["version_boundary"] = (
    axis["software_identity"]["formal_archive_matches_current_engine"] is False
    and "dev3 反事实内部是严格配对的" in text
)

xml_root = ET.parse(EVIDENCE / "pytest_relevant.xml").getroot()
suite = xml_root.find("testsuite")
if suite is None and xml_root.tag == "testsuite":
    suite = xml_root
assert suite is not None
checks["pytest_73_clean"] = (
    int(suite.attrib["tests"]) == 73
    and int(suite.attrib["failures"]) == 0
    and int(suite.attrib["errors"]) == 0
    and int(suite.attrib["skipped"]) == 0
)

relative_links: list[str] = []
missing_links: list[str] = []
for match in re.finditer(r"\]\(([^)]+)\)", text):
    target = match.group(1)
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
        continue
    target = target.split("#", 1)[0]
    relative_links.append(target)
    if not (ROOT / target).exists():
        missing_links.append(target)
checks["all_relative_links_exist"] = not missing_links
checks["code_fences_balanced"] = text.count("```") % 2 == 0
checks["display_math_balanced"] = text.count("$$") % 2 == 0
checks["mermaid_present"] = "```mermaid" in text
checks["no_placeholders"] = not re.search(r"\b(?:TODO|TBD|FIXME)\b", text)

result = {
    "schema_version": "1.0.0",
    "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
    "report_sha256": sha256(REPORT),
    "report_bytes": REPORT.stat().st_size,
    "report_lines": len(text.splitlines()),
    "checks": checks,
    "all_checks_passed": all(checks.values()),
    "relative_link_count": len(relative_links),
    "missing_relative_links": sorted(set(missing_links)),
    "source_hashes": {
        "ml_stats_audit.json": sha256(EVIDENCE / "ml_stats_audit.json"),
        "axis_geometry_necessity_audit.json": sha256(
            EVIDENCE / "axis_geometry_necessity_audit.json"
        ),
        "factorial_direction_source_counterfactual.json": sha256(
            EVIDENCE / "factorial_direction_source_counterfactual.json"
        ),
        "pytest_relevant.xml": sha256(EVIDENCE / "pytest_relevant.xml"),
    },
}
OUTPUT.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
if not result["all_checks_passed"]:
    raise SystemExit(1)
