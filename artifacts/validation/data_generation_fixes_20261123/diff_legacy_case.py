"""Show which fields differ between a rebuilt legacy case and its archived record."""
import gzip, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_aim_basis_fix as v

DATASET = v.DATASET
spec = v.load(DATASET / "plan/sampling_spec.json")
base = v.load(DATASET / "plan/base_case.json")
plan = v.build(base, spec, v.V3_PHYSICS)
with gzip.open(DATASET / "rounds/round-000/formal_cases.jsonl.gz", "rt", encoding="utf-8") as s:
    archived = {c["case_id"]: c for c in (json.loads(l) for l in s if l.strip())}

def diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((path + "/" + k, "missing-in-" + ("rebuilt" if k not in a else "archived")))
            else:
                out += diff(a[k], b[k], path + "/" + k)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append((path, f"{str(a)[:70]} != {str(b)[:70]}"))
    return out

from collections import Counter
counts = Counter()
example = None
for case in plan.cases:
    d = diff(case, archived[case["case_id"]])
    for p, _ in d:
        counts[p] += 1
    if example is None:
        example = d
STORAGE = ("/metadata/sampling/engine_code_hash", "/metadata/sampling/contract_document_references",
           "/metadata/sampling/label_spec_document", "/metadata/sampling/sampling_spec_document",
           "/metadata/sampling/split_registry_document", "/metadata/sampling/split_registry_reference",
           "/metadata/sampling/training_design_document")
def numeric_max(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return max([numeric_max(a[k], b[k]) for k in set(a) & set(b) if k not in ("sampling",)] or [0.0])
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return max([numeric_max(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b))
    return 0.0 if a == b else float("inf")
worst = 0.0
nonnumeric = Counter()
for case in plan.cases:
    arch = archived[case["case_id"]]
    worst = max(worst, numeric_max(case, arch))
    for path, _ in diff(case, arch):
        if path in STORAGE:
            continue
        a, b = case, arch
        ok = True
        nonnumeric[path] += 0
    if numeric_max(case, arch) == float("inf"):
        for path, detail in diff(case, arch):
            if path not in STORAGE and "!=" in detail and not any(ch.isdigit() for ch in detail.split("!=")[0][-1:]):
                nonnumeric[path] += 1
import sys as _s
paths = Counter(path for case in plan.cases for path, _ in diff(case, archived[case["case_id"]]))
print("python", _s.version.split()[0])
print("cases", len(plan.cases), "identical cases ignoring storage fields:",
      sum(1 for case in plan.cases if all(p in STORAGE for p, _ in diff(case, archived[case["case_id"]]))))
print("max abs numeric difference (non-storage fields):", worst)
print("non-storage paths that differ:", len([p for p in paths if p not in STORAGE]))
result = {
    "python": _s.version.split()[0],
    "archived_cases": len(archived),
    "rebuilt_cases": len(plan.cases),
    "identical_ignoring_storage_fields": sum(
        1 for case in plan.cases if all(p in STORAGE for p, _ in diff(case, archived[case["case_id"]]))
    ),
    "max_abs_numeric_difference_non_storage": worst,
    "storage_fields_ignored": list(STORAGE),
    "note": "engine_code_hash changes with any code edit; archives store contract documents as references",
}
(HERE / f"legacy_reproduction_py{_s.version_info.major}{_s.version_info.minor}.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8")
