"""Read-only representation audit of an existing archive.

No domain engine, simulation, label generation, training, response calculation,
or distribution/selection optimization. Numeric labels are compared, not emitted.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "raw_data/dataset_5000_loitering_impact_fragment_only_v1"
ARCHIVE = DATA / "formal_dataset"
OUT = Path(__file__).with_suffix(".json")
STAT = ("mean", "stddev", "ci95_lower", "ci95_upper", "sample_count", "uncertainty_estimable")
META = ("estimand", "stddev_estimand", "conditioning", "random_level", "interval_type", "interval_method", "coverage_status")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def content_hash(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def inventory():
    return {str(path.relative_to(DATA)).replace("\\", "/"): digest(path)
            for path in DATA.rglob("*") if path.is_file()}


def pointer(value, path):
    for token in path.lstrip("/").split("/"):
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


class Audit:
    def __init__(self):
        self.checks, self.elements, self.mismatches = Counter(), Counter(), Counter()
        self.examples = {}

    def compare(self, name, expected, observed, *, float32=False, case_id=None):
        left, right = np.asarray(expected), np.asarray(observed)
        self.checks[name] += 1
        if left.shape != right.shape:
            count = max(left.size, right.size, 1)
        else:
            if float32:
                left = left.astype(np.float32)
            if left.dtype.kind in "fci" or right.dtype.kind in "fci":
                left, right = left.astype(float), right.astype(float)
                equal = (left == right) | (np.isnan(left) & np.isnan(right))
            else:
                equal = left == right
            count = int(equal.size - np.count_nonzero(equal))
        self.elements[name] += max(left.size, right.size)
        if count:
            self.mismatches[name] += count
            self.examples.setdefault(name, [])
            if len(self.examples[name]) < 3:
                self.examples[name].append({"case_id": case_id, "expected_shape": list(left.shape), "observed_shape": list(right.shape)})


def aligned_table(path, case_ids, key_axes, stat_columns=True):
    schema = pq.read_schema(path)
    columns = ["case_id", *[name for name, _ in key_axes]]
    if stat_columns:
        columns += ["available", *STAT]
    else:
        columns += [name for name in ("features", "feature_mask", "replicate_count", "aleatory_seed", "epistemic_sample_id", "split", "case_hash") if name in schema.names]
    table = pq.read_table(path, columns=columns)
    code = pc.index_in(table["case_id"], value_set=pa.array(case_ids)).to_numpy(zero_copy_only=False)
    stride = 1
    for field, names in key_axes:
        code = code * len(names) + pc.index_in(table[field], value_set=pa.array(names)).to_numpy(zero_copy_only=False)
        stride *= len(names)
    if table.num_rows != len(case_ids) * stride or np.any(np.asarray(code, dtype=object) == None):
        raise ValueError(f"{path.name}: incomplete/unknown semantic key")
    order = np.argsort(code, kind="stable")
    if not np.array_equal(np.asarray(code)[order], np.arange(len(case_ids) * stride)):
        raise ValueError(f"{path.name}: duplicated/missing semantic key")
    arrays = {}
    for name in columns:
        if name == "case_id" or any(name == key for key, _ in key_axes):
            continue
        column = table[name].combine_chunks()
        if pa.types.is_list(column.type):
            arrays[name] = column.values.to_numpy(zero_copy_only=False).reshape((len(order), -1))[order]
        else:
            arrays[name] = column.to_numpy(zero_copy_only=False)[order]
    return {name: array.reshape((len(case_ids), stride, *array.shape[1:])) for name, array in arrays.items()}


def statistics_matrix(items):
    return {field: np.asarray([item.get(field) for item in items], dtype=float if field != "uncertainty_estimable" else bool)
            for field in STAT}


def main():
    if OUT.exists():
        raise RuntimeError("Preserve previous evidence and use a new output name.")
    before = inventory()
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
              "python_version": platform.python_version(), "scope": "stored representations only; no domain engine or new labels",
              "numeric_labels_emitted": False, "comparison": "exact float64 JSON/Parquet; NPZ exact JSON-to-float32 casting; mask/count exact; NaN equals NaN",
              "source_files_before": len(before), "shards": [], "not_verified": [
                  "current engine execution equivalence to the historical engine",
                  "physical validity or operational representativeness",
                  "replicate-level aggregation, joint alignment, or CI coverage where raw traces are absent",
                  "full geometric-feature recomputation"]}
    audit = Audit()
    try:
        root_manifest = load(ARCHIVE / "manifest.json")
        formal_path = DATA / "rounds/round-000/formal_cases.jsonl.gz"
        with gzip.open(formal_path, "rt", encoding="utf-8") as stream:
            cases = {row["case_id"]: row for row in map(json.loads, stream)}
        report["formal_case_count"] = len(cases)
        report["archive_engine"] = root_manifest["engine"]
        report["contract_versions"] = {"manifest": root_manifest["schema_version"], "tensor": root_manifest["tensor_contract"]["version"]}
        total, trace_count, conditioning = 0, 0, set()
        for shard in sorted((ARCHIVE / "shards").glob("part-*")):
            manifest = load(shard / "manifest.json")
            with np.load(shard / "tensors.npz", allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            ids = arrays["case_ids"].astype(str).tolist()
            component_ids, mechanisms, states = [arrays[name].astype(str).tolist() for name in ("component_ids", "mechanism_names", "state_names")]
            armor_ids, armor_names, aux_names, outputs, standards, bounds = [arrays[name].astype(str).tolist() for name in ("armor_ids", "armor_label_names", "component_aux_label_names", "system_output_names", "evaluation_standard_ids", "frechet_bound_names")]
            tables = {
                "component": aligned_table(shard/"labels.parquet", ids, [("component_id", component_ids),("mechanism", mechanisms),("damage_state", states)]),
                "component_aux": aligned_table(shard/"component_aux.parquet", ids, [("component_id", component_ids),("metric", aux_names)]),
                "armor": aligned_table(shard/"armor_labels.parquet", ids, [("armor_id", armor_ids),("metric", armor_names)]),
                "system": aligned_table(shard/"tree_outputs.parquet", ids, [("evaluation_standard_id", standards),("output_node", outputs)]),
            }
            features = {
                "case": aligned_table(shard/"cases.parquet", ids, [], False),
                "component": aligned_table(shard/"components.parquet", ids, [("component_id", component_ids)], False),
                "armor": aligned_table(shard/"armor.parquet", ids, [("armor_id", armor_ids)], False),
            }
            for family in ("case", "component", "armor"):
                for field, suffix in (("features", "_x"), ("feature_mask", "_feature_mask")):
                    expected = features[family][field]
                    if family == "case": expected = expected[:,0,:]
                    audit.compare(f"features.{family}.{field}", expected, arrays[family+suffix], float32=field=="features")
            case_names = arrays["case_feature_names"].astype(str).tolist()
            for view, names_field in (("case_physics", "case_physics_feature_names"),("solver_context", "solver_context_feature_names")):
                indices = [case_names.index(name) for name in arrays[names_field].astype(str)]
                audit.compare(f"view.{view}.values", arrays["case_x"][:,indices], arrays[view+"_x"])
                audit.compare(f"view.{view}.mask", arrays["case_feature_mask"][:,indices], arrays[view+"_feature_mask"])
            for family, table in tables.items():
                for field in STAT:
                    if field == "uncertainty_estimable": continue
                    values = table[field]
                    if field == "sample_count":
                        expected = np.where(table["available"], values, 0)
                        actual = arrays[family+"_sample_count"].reshape((len(ids),-1))
                    else:
                        if field == "mean": actual = arrays[family+"_y"].reshape((len(ids),-1))
                        elif field == "stddev": actual = arrays[family+"_y_stddev"].reshape((len(ids),-1))
                        else: actual = arrays[family+"_y_ci95"][...,0 if field.endswith("lower") else 1].reshape((len(ids),-1))
                        valid = table["available"] if field == "mean" else table["available"] & table["uncertainty_estimable"]
                        # Unavailable means remain explicit placeholders, not admitted labels.
                        expected = values if field == "mean" else np.where(valid, values, np.nan)
                    audit.compare(f"parquet_npz.{family}.{field}", expected, actual, float32=field != "sample_count")
                audit.compare(f"parquet_npz.{family}.availability", table["available"], arrays[family+"_label_mask"].reshape((len(ids),-1)))
                audit.compare(f"parquet_npz.{family}.uncertainty_mask", table["available"] & table["uncertainty_estimable"], arrays[family+"_uncertainty_mask"].reshape((len(ids),-1)))
            with gzip.open(shard/"results.jsonl.gz", "rt", encoding="utf-8") as stream:
                for local, raw in enumerate(stream):
                    result=json.loads(raw); case_id=result["case_id"]; case=cases[case_id]
                    audit.compare("identity.case_order", case_id, ids[local], case_id=case_id)
                    for field, tensor in (("case_hash","case_hashes"),("aleatory_seed","aleatory_seeds"),("epistemic_sample_id","epistemic_sample_ids")):
                        audit.compare("identity."+field, result["provenance"][field], arrays[tensor][local], case_id=case_id)
                        if field in features["case"]: audit.compare("parquet_identity."+field, result["provenance"][field], features["case"][field][local,0], case_id=case_id)
                    audit.compare("identity.parameter_feature_hash", content_hash(result["parameter_realization"]), result["provenance"]["parameter_realization_hash"], case_id=case_id)
                    audit.compare("identity.engine_code_hash", manifest["engine"]["code_hash"], result["provenance"]["engine_code_hash"], case_id=case_id)
                    audit.compare("identity.replicate_count", result["replicate_count"], arrays["replicate_counts"][local], case_id=case_id)
                    groups = {
                        "component": [result["components"][component]["damage_states"][mechanism][state] for component in component_ids for mechanism in mechanisms for state in states],
                        "component_aux": [result["components"][component]["fragment_metrics" if metric.startswith("fragment_") else "blast_metrics"][metric.split("_",1)[1]] for component in component_ids for metric in aux_names],
                        "armor": [result["armor"][armor][name] if name.startswith("post_") else result["armor"][armor]["blast_metrics"][{"blast_pre_integrity":"pre_blast_integrity","blast_post_integrity":"post_blast_integrity","blast_failure_probability":"blast_failure_probability"}.get(name,name.removeprefix("blast_"))] for armor in armor_ids for name in armor_names],
                        "system": [result["damage_tree"]["outputs"][name] for standard in standards for name in outputs],
                    }
                    for family, items in groups.items():
                        expected=statistics_matrix(items)
                        for field in STAT:
                            audit.compare(f"json_parquet.{family}.{field}", expected[field], tables[family][field][local], case_id=case_id)
                        for item in items: conditioning.add(item.get("conditioning"))
                    frechet=[result["components"][component]["damage_states"]["combined"][state][f"frechet_union_{bound}_bound"] for component in component_ids for state in states for bound in bounds]
                    frechet_stats=statistics_matrix(frechet)
                    frechet_mask=arrays["component_frechet_label_mask"][local].reshape(-1)
                    for field in ("mean","stddev","ci95_lower","ci95_upper"):
                        expected=frechet_stats[field]
                        actual=(arrays["component_frechet_y"][local] if field=="mean" else arrays["component_frechet_y_stddev"][local] if field=="stddev" else arrays["component_frechet_y_ci95"][local,...,0 if field.endswith("lower") else 1]).reshape(-1)
                        if field!="mean": expected=np.where(frechet_mask & frechet_stats["uncertainty_estimable"],expected,np.nan)
                        audit.compare("json_npz.frechet."+field,expected,actual,float32=True,case_id=case_id)
                    view=[]
                    for path, transform in zip(arrays["loitering_impact_feature_source_paths"].astype(str),arrays["loitering_impact_feature_transforms"].astype(str)):
                        value=pointer(case,path)
                        if transform.startswith("one_hot_equals_"): value=float(value==transform.removeprefix("one_hot_equals_"))
                        elif transform not in {"identity","identity_positive_x_component"}: raise ValueError("unknown archived feature transform")
                        view.append(value)
                    audit.compare("input.training_view",view,arrays["loitering_impact_x"][local],float32=True,case_id=case_id)
                    trace_count += int("replicate_trace" in result)
                    total+=1
            report["shards"].append({"name":shard.name,"cases_checked":len(ids),"manifest_hash":digest(shard/"manifest.json")})
            print(json.dumps({"completed_shard":shard.name,"cases_checked":total,"mismatch_checks":len(audit.mismatches)}),flush=True)
        report.update({"status":"completed","result_count":total,"replicate_trace_result_count":trace_count,"conditioning_values":sorted(conditioning)})
    except Exception as error:
        report.update({"status":"audit_incomplete","fatal_error":{"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}})
    finally:
        after=inventory()
        report.update({"checks":dict(audit.checks),"scalar_comparisons":dict(audit.elements),"mismatch_counts":dict(audit.mismatches),"limited_mismatch_examples":audit.examples,
                       "archive_files_unchanged":before==after,"source_files_after":len(after),"file_inventory_before_hash":content_hash(before),"file_inventory_after_hash":content_hash(after)})
        OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print(json.dumps({"evidence":str(OUT),"status":report["status"],"mismatch_checks":dict(audit.mismatches),"unchanged":before==after},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
