"""Generic harmless arrays/files only; never audit physical input windows."""
from __future__ import annotations
import importlib.util
import json
import math
from pathlib import Path
import sys
import uuid
import warnings

import numpy as np

ROOT = Path(__file__).absolute().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT/file)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def safe(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "nonfinite"
    if isinstance(value, dict):
        return {k:safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [safe(v) for v in value]
    return value


def main():
    from damage_engine.adaptive_sampling import effective_sample_size
    generic = module("io_generic_distribution", "tools/audit_dataset_distribution.py")
    descriptive = module("io_generic_helpers", "tools/audit_loitering_impact_dataset.py")
    results = {"scope":"Artificial arrays and artificial two-shard directory only; no real domain inputs, engine, labels, training or parameter tuning",
               "cases":{}}
    values = np.asarray([[0.],[0.],[1.],[1.],[2.],[2.]])
    splits = np.asarray(["train","train","validation","validation","test","test"])
    drift = generic._split_drift(values, np.ones(values.shape,dtype=bool), ["toy_axis"], splits)
    results["cases"]["distinct_constant_splits"] = {
        "description":"Each split is internally constant, and the three values differ",
        "reported_smds":[v["standardized_mean_difference"] for v in drift[0]["comparisons"]],
        "reported_ks":[v["ks_statistic"] for v in drift[0]["comparisons"]],
        "defect":"zero within-split variance and different means is reported as SMD=0 instead of a degenerate/undefined standardized effect with explicit mean separation"}
    with warnings.catch_warnings(record=True), np.errstate(all="ignore"):
        weights = {}
        for name, scale in (("unit",1.),("large_finite",1e200),("small_finite",1e-200)):
            raw = np.full(3,scale)
            weights[name] = {"expected_ess":3.,"core_ess":effective_sample_size(raw),
                             "stratum_ess":generic._stratum_effective_count(np.ones(3,dtype=bool),raw,np.ones(3,dtype=bool))["design_weight_ess"]}
        results["cases"]["ess_scale_invariance"] = weights
    invalid = descriptive._weight_audit({"sampling_design_weight_raw":np.asarray([1.,-1.,1.]),
                                         "sampling_weight_mask":np.ones(3,dtype=bool),
                                         "sampling_selection_probability":np.asarray([1.,1.1,1.])},
                                        np.asarray(["train","validation","test"]))
    results["cases"]["invalid_eligible_weight_and_probability"] = {
        "eligible_input_count":3,"accepted_weight_count":invalid["all"]["count"],
        "reported_ess_fraction":invalid["all"]["ess_fraction"],
        "reported_probability_violations":invalid["positivity_violations"],
        "invalid_weight_count_field_present":"invalid_weight_count" in invalid,
        "raised_validation_error":False,
        "defect":"negative eligible weight silently filtered; probability above 1 omitted from violation count"}
    collapsed = descriptive._bin_coverage(np.asarray([2.,2.]),2.,2.,bins=4)
    bounded = descriptive._bin_coverage(np.asarray([-.1,0.,.5,1.,1.1,np.nan]),0.,1.,bins=4)
    results["cases"]["bin_contract"] = {
        "constant_axis_declared_range":collapsed["declared_range"],
        "constant_axis_histogram_range":[collapsed["edges"][0],collapsed["edges"][-1]],
        "constant_axis_explicit_not_estimable_status_present":"status" in collapsed,
        "ordinary_boundary_input_count":6,"finite_observed_count":bounded["observed"],
        "out_of_range_count":bounded["out_of_range"],"binned_count":sum(bounded["counts"]),
        "missing_count_field_present":"missing_count" in bounded}
    summary = generic._summary(np.asarray([3.,3.,np.nan,np.inf]))
    results["cases"]["summary_missingness"] = {
        "input_count":4,"reported_count":summary["count"],"reported_unique":summary["unique"],
        "invalid_count_field_present":"invalid_count" in summary,
        "boundary":"helper removes nonfinite values; caller must disclose/check the original eligible denominator"}
    fixture = OUT/("io_toy_uncommitted_shards_"+uuid.uuid4().hex[:8])
    first,extra = fixture/"shards/part-000000",fixture/"shards/part-999999"
    first.mkdir(parents=True)
    extra.mkdir(parents=True)
    np.savez(first/"tensors.npz",case_ids=np.asarray(["toy-kept"]),split_names=np.asarray(["train"]))
    np.savez(extra/"tensors.npz",case_ids=np.asarray(["toy-uncommitted"]),split_names=np.asarray(["test"]))
    (fixture/"manifest.json").write_text(json.dumps({"partitions":[{"name":"tensors","path_glob":"shards/part-000000/tensors.npz"}]}),encoding="utf-8")
    (fixture/"checkpoint.json").write_text(json.dumps({"status":"complete","completed_shards":[{"path":"shards/part-000000"}]}),encoding="utf-8")
    arrays,_,paths = descriptive._load_tensors(fixture)
    results["cases"]["glob_includes_uncommitted_shard"] = {
        "fixture":fixture.relative_to(ROOT).as_posix(),"declared_shard_count":1,"loaded_shard_count":len(paths),
        "loaded_case_count":len(arrays["case_ids"]),"uncommitted_case_loaded":"toy-uncommitted" in arrays["case_ids"],
        "manifest_and_checkpoint_validated":False,
        "boundary":"fixture is intentionally minimal; helper does not consult either file, so it also does not establish a complete archive contract"}
    results["source_hashes"] = {}
    from damage_engine.utils import file_hash
    for name in ("damage_engine/adaptive_sampling.py","damage_engine/sampling_audit.py","tools/audit_dataset_distribution.py","tools/audit_loitering_impact_dataset.py"):
        results["source_hashes"][name] = file_hash(ROOT/name)
    (OUT/"io_audit_algorithms_toy.json").write_text(json.dumps(safe(results),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"toy_case_count":len(results["cases"]),"real_domain_records_read":0,"output":"io_audit_algorithms_toy.json"}))


if __name__ == "__main__":
    main()
